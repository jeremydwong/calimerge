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
    if (!model || !model->ml_model) return;

    @autoreleasepool {
        /* Release the retained MLModel */
        MLModel *ml = (__bridge_transfer MLModel *)model->ml_model;
        (void)ml;  /* ARC releases it */
    }

    memset(model, 0, sizeof(*model));
}

/* ============================================================================
 * Inference
 * ============================================================================ */

int pt_coreml_infer(PT_CoreMLModel *model,
                    const float *input_data,
                    float *output_data,
                    int batch_size) {
    if (!model || !model->ml_model || !input_data || !output_data)
        return PT_ERR_INVALID_ARGS;

    @autoreleasepool {
        MLModel *ml = (__bridge MLModel *)model->ml_model;
        NSError *error = nil;

        /* Build input MLMultiArray wrapping caller's buffer (zero-copy).
         * Shape: (batch, channels, height, width) */
        NSArray<NSNumber *> *shape = @[
            @(batch_size),
            @(model->input_channels),
            @(model->input_height),
            @(model->input_width)
        ];

        int total_elements = batch_size * model->input_channels
                           * model->input_height * model->input_width;

        /* Create input array — copy data into MLMultiArray */
        MLMultiArray *input_array = [[MLMultiArray alloc]
            initWithShape:shape
            dataType:MLMultiArrayDataTypeFloat32
            error:&error];
        if (!input_array) {
            pt_coreml_log("Failed to create input array: %s",
                          error.localizedDescription.UTF8String);
            return PT_ERR_INFERENCE;
        }

        /* Copy input data */
        memcpy(input_array.dataPointer, input_data,
               total_elements * sizeof(float));

        /* Get input feature name from model description */
        MLModelDescription *desc = ml.modelDescription;
        NSString *input_name = desc.inputDescriptionsByName.allKeys.firstObject;
        NSString *output_name = desc.outputDescriptionsByName.allKeys.firstObject;

        if (!input_name || !output_name) {
            pt_coreml_log("Model has no input/output features");
            return PT_ERR_INFERENCE;
        }

        /* Create feature provider */
        MLDictionaryFeatureProvider *provider =
            [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{input_name: input_array}
                error:&error];
        if (!provider) {
            pt_coreml_log("Failed to create feature provider: %s",
                          error.localizedDescription.UTF8String);
            return PT_ERR_INFERENCE;
        }

        /* Run prediction */
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

        /* Copy output to caller's buffer */
        int out_count = (int)output_array.count;
        if (output_array.dataType == MLMultiArrayDataTypeFloat32) {
            memcpy(output_data, output_array.dataPointer,
                   out_count * sizeof(float));
        } else if (output_array.dataType == MLMultiArrayDataTypeFloat16) {
            /* Convert fp16 -> fp32 */
            const uint16_t *fp16 = (const uint16_t *)output_array.dataPointer;
            for (int i = 0; i < out_count; i++) {
                /* Use vImage or manual conversion */
                uint16_t h = fp16[i];
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp  = (h >> 10) & 0x1F;
                uint32_t frac = h & 0x3FF;
                uint32_t f;
                if (exp == 0) {
                    f = sign << 31;  /* zero or subnormal (treat as zero) */
                } else if (exp == 31) {
                    f = (sign << 31) | 0x7F800000 | (frac << 13);  /* inf/nan */
                } else {
                    f = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
                }
                memcpy(&output_data[i], &f, sizeof(float));
            }
        } else {
            /* Double or other — cast element by element */
            for (int i = 0; i < out_count; i++) {
                output_data[i] = (float)[[output_array objectAtIndexedSubscript:i]
                                         floatValue];
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
