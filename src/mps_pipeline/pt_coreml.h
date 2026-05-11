/*
 * pt_coreml.h - CoreML model management: load compiled models, run inference.
 *
 * macOS equivalent of pt_tensorrt.h.  Uses CoreML for YOLO and VitPose
 * inference on Apple Silicon (ANE + GPU auto-scheduling).
 *
 * Workflow:
 *   1. pt_coreml_load()    -- load a compiled .mlmodelc or .mlpackage
 *   2. pt_coreml_infer()   -- run synchronous inference
 *   3. pt_coreml_unload()  -- free model resources
 *
 * Style: Plain C API (extern "C"), implemented in Objective-C (.m).
 */

#ifndef PT_COREML_H
#define PT_COREML_H

#include "../pt_shared/pt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Data structures
 * ============================================================================ */

typedef struct {
    void *ml_model;         /* MLModel*              */
    int   input_batch;      /* current/fixed batch size */
    int   input_channels;
    int   input_height;
    int   input_width;
    int   output_dims[4];   /* output shape (batch, C, H, W) or (batch, N, 6) */
    char  model_path[512];

    /* Cached across calls to avoid per-inference allocation */
    void *cached_input_array;    /* MLMultiArray* — reused when batch matches */
    int   cached_input_batch;    /* batch size the cached array was allocated for */
    void *cached_input_name;     /* NSString* — input feature name */
    void *cached_output_name;    /* NSString* — output feature name */
} PT_CoreMLModel;

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

/*
 * Load a CoreML model from disk.
 *
 * model_path can be:
 *   - .mlpackage directory (auto-compiled on first use, cached)
 *   - .mlmodelc directory (pre-compiled, fastest startup)
 *
 * Returns PT_OK on success, PT_ERR_FILE_NOT_FOUND or PT_ERR_ENGINE_BUILD.
 */
int pt_coreml_load(PT_CoreMLModel *model, const char *model_path);

/*
 * Unload a CoreML model and free resources.
 * Safe to call on a zeroed struct (no-op).
 */
void pt_coreml_unload(PT_CoreMLModel *model);

/* ============================================================================
 * Inference
 * ============================================================================ */

/*
 * Run synchronous inference.
 *
 * input_data  - host pointer to input tensor (float32, NCHW layout)
 * output_data - host pointer to pre-allocated output buffer
 * batch_size  - actual batch size for this call
 *
 * On Apple Silicon, input/output are in unified memory — no copy overhead.
 * CoreML dispatches to ANE/GPU/CPU automatically.
 *
 * Returns PT_OK on success, PT_ERR_INFERENCE on failure.
 */
int pt_coreml_infer(PT_CoreMLModel *model,
                    const float *input_data,
                    float *output_data,
                    int batch_size);

/*
 * Get total output element count for a given batch size.
 * Returns 0 on error.
 */
int pt_coreml_output_size(const PT_CoreMLModel *model, int batch_size);

#ifdef __cplusplus
}
#endif

#endif /* PT_COREML_H */
