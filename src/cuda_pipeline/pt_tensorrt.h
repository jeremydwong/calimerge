/*
 * pt_tensorrt.h - TensorRT engine management: ONNX parsing, engine caching,
 *                 and inference dispatch.
 *
 * Wraps the TensorRT C++ API behind plain-C free functions.  Internally uses
 * C++ for nvinfer1 calls; every exported symbol is extern "C".
 *
 * Workflow:
 *   1. pt_trt_init()          -- create IRuntime (once at startup)
 *   2. pt_trt_build_engine()  -- load or build a .engine from ONNX
 *   3. pt_trt_infer()         -- enqueue inference on a CUDA stream
 *   4. pt_trt_destroy_engine() -- free per-engine resources
 *   5. pt_trt_shutdown()      -- destroy runtime (once at shutdown)
 */

#ifndef PT_TENSORRT_H
#define PT_TENSORRT_H

#include "pt_common.h"

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Data structures
 * ============================================================================ */

typedef struct {
    void *runtime;      /* nvinfer1::IRuntime*          */
    void *engine;       /* nvinfer1::ICudaEngine*       */
    void *context;      /* nvinfer1::IExecutionContext*  */

    int input_index;
    int output_index;
    int max_batch_size;

    /* Input/output shapes (for validation) -- NCHW or N,D,6 etc. */
    int input_dims[4];
    int output_dims[4];

    char engine_path[512];
    char onnx_path[512];
} PT_TrtEngine;

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

/*
 * pt_trt_init - Create TensorRT IRuntime.  Call once at startup.
 *
 * Returns PT_OK on success, PT_ERR_TENSORRT if TensorRT is unavailable.
 */
int pt_trt_init(void);

/*
 * pt_trt_shutdown - Destroy the global IRuntime.  Call once at exit.
 *
 * Safe to call even if pt_trt_init was never called (no-op).
 */
void pt_trt_shutdown(void);

/* ============================================================================
 * Engine management
 * ============================================================================ */

/*
 * pt_trt_build_engine - Build (or load cached) a TensorRT engine from ONNX.
 *
 * Cache key: {model_name}_{sm_version}_{max_batch}_{fp16}.engine
 * If a cached file exists AND is newer than the ONNX file it is deserialized
 * directly; otherwise the ONNX is parsed and a new engine is built.
 *
 * The optimization profile sets:
 *   min  batch = 1
 *   opt  batch = max_batch / 2  (clamped to >= 1)
 *   max  batch = max_batch
 *
 * Parameters:
 *   engine     - output struct (zeroed by caller)
 *   onnx_path  - path to the ONNX model file
 *   cache_dir  - directory for .engine cache files (created if needed)
 *   max_batch  - maximum batch size for the optimization profile
 *   use_fp16   - 1 to enable FP16 precision, 0 for FP32
 *
 * Returns PT_OK on success, PT_ERR_FILE_NOT_FOUND, PT_ERR_ENGINE_BUILD,
 *         or PT_ERR_TENSORRT on failure.
 */
int pt_trt_build_engine(PT_TrtEngine *engine,
                        const char *onnx_path,
                        const char *cache_dir,
                        int max_batch,
                        int use_fp16);

/*
 * pt_trt_destroy_engine - Destroy execution context and engine.
 *
 * Safe to call on a zeroed PT_TrtEngine (no-op).  Does NOT destroy the
 * global runtime -- that is handled by pt_trt_shutdown.
 */
void pt_trt_destroy_engine(PT_TrtEngine *engine);

/* ============================================================================
 * Inference
 * ============================================================================ */

/*
 * pt_trt_infer - Enqueue inference on the given CUDA stream.
 *
 * The caller provides pre-allocated GPU buffers for input and output.
 * The batch dimension is set on the execution context before enqueue.
 *
 * Parameters:
 *   engine     - a built PT_TrtEngine
 *   input_gpu  - device pointer to input tensor (batch_size * C*H*W elements)
 *   output_gpu - device pointer to output tensor
 *   batch_size - actual batch size for this call (<= max_batch_size)
 *   stream     - CUDA stream for async execution
 *
 * Returns PT_OK on success, PT_ERR_INFERENCE on failure.
 */
int pt_trt_infer(PT_TrtEngine *engine,
                 void *input_gpu,
                 void *output_gpu,
                 int batch_size,
                 cudaStream_t stream);

/* ============================================================================
 * Utilities
 * ============================================================================ */

/*
 * pt_trt_get_binding_size - Total byte count for a binding at a given batch.
 *
 * Returns the number of bytes, or 0 on error.
 */
int pt_trt_get_binding_size(PT_TrtEngine *engine,
                            int binding_index,
                            int batch_size);

#ifdef __cplusplus
}
#endif

#endif /* PT_TENSORRT_H */
