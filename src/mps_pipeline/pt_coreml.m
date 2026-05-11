/*
 * pt_coreml.m - CoreML model management for macOS Apple Silicon.
 *
 * Loads .mlpackage or .mlmodelc models via CoreML, runs inference using
 * MLMultiArray for I/O.  CoreML auto-dispatches to ANE/GPU/CPU.
 */

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#include "pt_coreml.h"
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * Helpers
 * ============================================================================ */

static void pt_coreml_log(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[pt_coreml] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

int pt_coreml_load(PT_CoreMLModel *model, const char *model_path) {
    if (!model || !model_path) return PT_ERR_INVALID_ARGS;

    memset(model, 0, sizeof(*model));
    strncpy(model->model_path, model_path, sizeof(model->model_path) - 1);

    @autoreleasepool {
        NSString *path = [NSString stringWithUTF8String:model_path];
        NSURL *url = [NSURL fileURLWithPath:path];

        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
            pt_coreml_log("Model not found: %s", model_path);
            return PT_ERR_FILE_NOT_FOUND;
        }

        NSError *error = nil;
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;  /* ANE + GPU + CPU */

        /* Determine if this is an mlpackage or mlmodelc */
        MLModel *ml_model = nil;

        if ([path hasSuffix:@".mlmodelc"]) {
            /* Pre-compiled model — fastest load */
            NSURL *compiled_url = url;
            ml_model = [MLModel modelWithContentsOfURL:compiled_url
                                         configuration:config
                                                 error:&error];
        } else if ([path hasSuffix:@".mlpackage"]) {
            /* Compile on first use (CoreML caches the compiled version) */
            NSURL *compiled_url = [MLModel compileModelAtURL:url error:&error];
            if (!compiled_url) {
                pt_coreml_log("Failed to compile model: %s",
                              error.localizedDescription.UTF8String);
                return PT_ERR_ENGINE_BUILD;
            }
            ml_model = [MLModel modelWithContentsOfURL:compiled_url
                                         configuration:config
                                                 error:&error];
        } else {
            pt_coreml_log("Unknown model format (expected .mlpackage or .mlmodelc): %s",
                          model_path);
            return PT_ERR_FILE_NOT_FOUND;
        }

        if (!ml_model) {
            pt_coreml_log("Failed to load model: %s",
                          error.localizedDescription.UTF8String);
            return PT_ERR_ENGINE_BUILD;
        }

        /* Extract input shape from model description */
        MLModelDescription *desc = ml_model.modelDescription;
        NSDictionary *inputs = desc.inputDescriptionsByName;

        /* Expect a single input — iterate to get the first */
        for (NSString *name in inputs) {
            MLFeatureDescription *feat = inputs[name];
            if (feat.type == MLFeatureTypeMultiArray) {
                MLMultiArrayConstraint *constraint = feat.multiArrayConstraint;
                NSArray<NSNumber *> *shape = constraint.shape;
                if (shape.count >= 4) {
                    model->input_batch    = shape[0].intValue;
                    model->input_channels = shape[1].intValue;
                    model->input_height   = shape[2].intValue;
                    model->input_width    = shape[3].intValue;
                }
            }
            break;  /* first input only */
        }

        /* Extract output shape */
        NSDictionary *outputs = desc.outputDescriptionsByName;
        for (NSString *name in outputs) {
            MLFeatureDescription *feat = outputs[name];
            if (feat.type == MLFeatureTypeMultiArray) {
                MLMultiArrayConstraint *constraint = feat.multiArrayConstraint;
                NSArray<NSNumber *> *shape = constraint.shape;
                for (int i = 0; i < (int)shape.count && i < 4; i++) {
                    model->output_dims[i] = shape[i].intValue;
                }
            }
            break;  /* first output only */
        }

        /* Cache feature name strings to avoid per-call dictionary lookup */
        NSString *in_name = desc.inputDescriptionsByName.allKeys.firstObject;
        NSString *out_name = desc.outputDescriptionsByName.allKeys.firstObject;
        if (in_name)  model->cached_input_name  = (__bridge_retained void *)[in_name copy];
        if (out_name) model->cached_output_name = (__bridge_retained void *)[out_name copy];

        /* Retain the model (prevent ARC from releasing it) */
        model->ml_model = (__bridge_retained void *)ml_model;

        pt_coreml_log("Loaded: %s  input=(%d,%d,%d,%d)",
                      model_path,
                      model->input_batch, model->input_channels,
                      model->input_height, model->input_width);
    }

    return PT_OK;
}

void pt_coreml_unload(PT_CoreMLModel *model) {
    if (!model) return;

    @autoreleasepool {
        if (model->cached_input_array) {
            MLMultiArray *arr = (__bridge_transfer MLMultiArray *)model->cached_input_array;
            (void)arr;
        }
        if (model->cached_input_name) {
            NSString *s = (__bridge_transfer NSString *)model->cached_input_name;
            (void)s;
        }
        if (model->cached_output_name) {
            NSString *s = (__bridge_transfer NSString *)model->cached_output_name;
            (void)s;
        }
        if (model->ml_model) {
            MLModel *ml = (__bridge_transfer MLModel *)model->ml_model;
            (void)ml;
        }
    }

    memset(model, 0, sizeof(*model));
}

/* ============================================================================
 * Inference
 * ============================================================================ */

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        f = sign << 31;
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (frac << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
    }
    float v;
    memcpy(&v, &f, sizeof(float));
    return v;
}

int pt_coreml_infer(PT_CoreMLModel *model,
                    const float *input_data,
                    float *output_data,
                    int batch_size) {
    if (!model || !model->ml_model || !input_data || !output_data)
        return PT_ERR_INVALID_ARGS;

    @autoreleasepool {
        MLModel *ml = (__bridge MLModel *)model->ml_model;
        NSError *error = nil;

        int total_elements = batch_size * model->input_channels
                           * model->input_height * model->input_width;

        /* Reuse cached input MLMultiArray when batch size matches */
        MLMultiArray *input_array = nil;
        if (model->cached_input_array && model->cached_input_batch == batch_size) {
            input_array = (__bridge MLMultiArray *)model->cached_input_array;
        } else {
            /* Release old cached array if batch changed */
            if (model->cached_input_array) {
                MLMultiArray *old = (__bridge_transfer MLMultiArray *)model->cached_input_array;
                (void)old;
                model->cached_input_array = NULL;
            }

            NSArray<NSNumber *> *shape = @[
                @(batch_size),
                @(model->input_channels),
                @(model->input_height),
                @(model->input_width)
            ];
            input_array = [[MLMultiArray alloc]
                initWithShape:shape
                dataType:MLMultiArrayDataTypeFloat32
                error:&error];
            if (!input_array) {
                pt_coreml_log("Failed to create input array: %s",
                              error.localizedDescription.UTF8String);
                return PT_ERR_INFERENCE;
            }
            model->cached_input_array = (__bridge_retained void *)input_array;
            model->cached_input_batch = batch_size;
        }

        memcpy(input_array.dataPointer, input_data,
               total_elements * sizeof(float));

        /* Use cached feature names */
        NSString *input_name  = (__bridge NSString *)model->cached_input_name;
        NSString *output_name = (__bridge NSString *)model->cached_output_name;
        if (!input_name || !output_name) {
            pt_coreml_log("Model has no cached input/output feature names");
            return PT_ERR_INFERENCE;
        }

        MLDictionaryFeatureProvider *provider =
            [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{input_name: input_array}
                error:&error];
        if (!provider) {
            pt_coreml_log("Failed to create feature provider: %s",
                          error.localizedDescription.UTF8String);
            return PT_ERR_INFERENCE;
        }

        id<MLFeatureProvider> result = [ml predictionFromFeatures:provider
                                                           error:&error];
        if (!result) {
            pt_coreml_log("Inference failed: %s",
                          error.localizedDescription.UTF8String);
            return PT_ERR_INFERENCE;
        }

        /* Extract output */
        MLFeatureValue *output_feat = [result featureValueForName:output_name];
        if (!output_feat || output_feat.type != MLFeatureTypeMultiArray) {
            pt_coreml_log("Output feature not found or wrong type");
            return PT_ERR_INFERENCE;
        }

        MLMultiArray *output_array = output_feat.multiArrayValue;

        /* MLMultiArray storage may not be contiguous: CoreML often pads
         * inner dimensions for SIMD alignment.  Use nested loops with
         * stride arithmetic instead of per-element divmod. */
        NSArray<NSNumber *> *out_shape = output_array.shape;
        NSArray<NSNumber *> *out_strides = output_array.strides;
        int rank = (int)out_shape.count;
        int dims[4] = {1, 1, 1, 1};
        int strs[4] = {0, 0, 0, 0};
        for (int i = 0; i < rank && i < 4; i++) {
            dims[i] = out_shape[i].intValue;
            strs[i] = out_strides[i].intValue;
        }

        const void *base = output_array.dataPointer;
        int is_fp32 = (output_array.dataType == MLMultiArrayDataTypeFloat32);
        int is_fp16 = (output_array.dataType == MLMultiArrayDataTypeFloat16);
        size_t elem_size = is_fp32 ? 4 : is_fp16 ? 2 : 8;
        int out_idx = 0;

        /* Fast path: FP32 with contiguous innermost dimension — copy
         * whole rows via memcpy instead of element-by-element. */
        if (is_fp32 && strs[rank - 1] == 1) {
            int inner = dims[rank - 1];
            if (rank <= 2) {
                for (int d0 = 0; d0 < dims[0]; d0++) {
                    const float *row = (const float *)base + (size_t)d0 * strs[0];
                    memcpy(output_data + out_idx, row, inner * sizeof(float));
                    out_idx += inner;
                }
            } else if (rank == 3) {
                for (int d0 = 0; d0 < dims[0]; d0++) {
                    for (int d1 = 0; d1 < dims[1]; d1++) {
                        const float *row = (const float *)base
                            + (size_t)d0 * strs[0] + (size_t)d1 * strs[1];
                        memcpy(output_data + out_idx, row, inner * sizeof(float));
                        out_idx += inner;
                    }
                }
            } else {
                for (int d0 = 0; d0 < dims[0]; d0++) {
                    for (int d1 = 0; d1 < dims[1]; d1++) {
                        for (int d2 = 0; d2 < dims[2]; d2++) {
                            const float *row = (const float *)base
                                + (size_t)d0 * strs[0] + (size_t)d1 * strs[1]
                                + (size_t)d2 * strs[2];
                            memcpy(output_data + out_idx, row, inner * sizeof(float));
                            out_idx += inner;
                        }
                    }
                }
            }
        }
        /* FP16 with contiguous innermost — convert per row */
        else if (is_fp16 && strs[rank - 1] == 1) {
            int inner = dims[rank - 1];
            if (rank <= 2) {
                for (int d0 = 0; d0 < dims[0]; d0++) {
                    const uint16_t *row = (const uint16_t *)base + (size_t)d0 * strs[0];
                    for (int w = 0; w < inner; w++)
                        output_data[out_idx++] = fp16_to_fp32(row[w]);
                }
            } else if (rank == 3) {
                for (int d0 = 0; d0 < dims[0]; d0++) {
                    for (int d1 = 0; d1 < dims[1]; d1++) {
                        const uint16_t *row = (const uint16_t *)base
                            + (size_t)d0 * strs[0] + (size_t)d1 * strs[1];
                        for (int w = 0; w < inner; w++)
                            output_data[out_idx++] = fp16_to_fp32(row[w]);
                    }
                }
            } else {
                for (int d0 = 0; d0 < dims[0]; d0++) {
                    for (int d1 = 0; d1 < dims[1]; d1++) {
                        for (int d2 = 0; d2 < dims[2]; d2++) {
                            const uint16_t *row = (const uint16_t *)base
                                + (size_t)d0 * strs[0] + (size_t)d1 * strs[1]
                                + (size_t)d2 * strs[2];
                            for (int w = 0; w < inner; w++)
                                output_data[out_idx++] = fp16_to_fp32(row[w]);
                        }
                    }
                }
            }
        }
        /* General fallback: nested loops with stride indexing */
        else {
            for (int d0 = 0; d0 < dims[0]; d0++) {
                for (int d1 = 0; d1 < dims[1]; d1++) {
                    for (int d2 = 0; d2 < dims[2]; d2++) {
                        for (int d3 = 0; d3 < dims[3]; d3++) {
                            size_t offset = (size_t)d0 * strs[0]
                                          + (size_t)d1 * strs[1]
                                          + (size_t)d2 * strs[2]
                                          + (size_t)d3 * strs[3];
                            offset *= elem_size;
                            const void *p = (const uint8_t *)base + offset;
                            float v;
                            if (is_fp32) {
                                v = *(const float *)p;
                            } else if (is_fp16) {
                                v = fp16_to_fp32(*(const uint16_t *)p);
                            } else {
                                v = (float)[[output_array objectAtIndexedSubscript:out_idx]
                                            floatValue];
                            }
                            output_data[out_idx++] = v;
                        }
                    }
                }
            }
        }
    }

    return PT_OK;
}

int pt_coreml_output_size(const PT_CoreMLModel *model, int batch_size) {
    if (!model) return 0;

    int size = batch_size;
    for (int i = 1; i < 4; i++) {
        if (model->output_dims[i] > 0)
            size *= model->output_dims[i];
    }
    return size;
}
