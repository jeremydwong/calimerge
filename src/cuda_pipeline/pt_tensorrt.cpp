/*
 * pt_tensorrt.cpp - TensorRT engine management implementation.
 *
 * Internally uses C++ for TensorRT API calls (nvinfer1 namespace).
 * All exported functions are extern "C" (declared in pt_tensorrt.h).
 *
 * Build note: link against -lnvinfer -lnvonnxparser -lcudart.
 * If TensorRT headers are missing the file still compiles -- every public
 * function returns PT_ERR_TENSORRT with a clear log message.
 */

#include "pt_tensorrt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * We guard the entire TensorRT implementation behind a feature macro.
 * When building without TensorRT the stub path at the bottom provides
 * graceful error returns.
 */
#if __has_include(<NvInfer.h>)
#define PT_HAS_TENSORRT 1
#else
#define PT_HAS_TENSORRT 0
#endif

#if PT_HAS_TENSORRT

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>   /* _mkdir */
#endif

/* ============================================================================
 * TensorRT logger -- just fprintf to stderr with a severity prefix.
 * ============================================================================ */

class PtTrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        const char *prefix = "";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: prefix = "[TRT INTERNAL_ERROR]"; break;
            case Severity::kERROR:          prefix = "[TRT ERROR]";          break;
            case Severity::kWARNING:        prefix = "[TRT WARNING]";        break;
            case Severity::kINFO:           prefix = "[TRT INFO]";           break;
            case Severity::kVERBOSE:        return; /* skip verbose */
        }
        fprintf(stderr, "%s %s\n", prefix, msg);
    }
};

/* ============================================================================
 * Module-level state
 * ============================================================================ */

static PtTrtLogger          g_logger;
static nvinfer1::IRuntime  *g_runtime       = nullptr;
static int                  g_initialized   = 0;

/* ============================================================================
 * Internal helpers
 * ============================================================================ */

/* Get the SM version string for this GPU (e.g. "86" for sm_86). */
static int pt_get_sm_version(char *out, int out_size) {
    int device = 0;
    cudaDeviceProp props;
    if (cudaGetDevice(&device) != cudaSuccess) return -1;
    if (cudaGetDeviceProperties(&props, device) != cudaSuccess) return -1;
    snprintf(out, out_size, "%d%d", props.major, props.minor);
    return 0;
}

/* Extract the base model name from an ONNX path (without directories or extension). */
static void pt_extract_model_name(const char *onnx_path, char *out, int out_size) {
    /* Find last slash or backslash. */
    const char *name = onnx_path;
    for (const char *p = onnx_path; *p; ++p) {
        if (*p == '/' || *p == '\\') name = p + 1;
    }
    /* Copy up to the last '.'. */
    int len = 0;
    for (const char *p = name; *p && *p != '.'; ++p) {
        if (len < out_size - 1) out[len++] = *p;
    }
    out[len] = '\0';
}

/* Return file modification time (seconds since epoch), or 0 on error. */
static time_t pt_file_mtime(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return st.st_mtime;
}

/* Return 1 if the file exists, 0 otherwise. */
static int pt_file_exists(const char *path) {
    struct stat st;
    return (stat(path, &st) == 0) ? 1 : 0;
}

/* Create directory (one level). Returns 0 on success or if it already exists. */
static int pt_mkdir(const char *path) {
#ifdef _WIN32
    /* _mkdir on MSVC, mkdir on POSIX */
    struct stat st;
    if (stat(path, &st) == 0) return 0;
    return _mkdir(path);
#else
    struct stat st;
    if (stat(path, &st) == 0) return 0;
    return mkdir(path, 0755);
#endif
}

/* Read an entire file into malloc'd memory.  Caller must free().
 * Sets *out_size to the byte count.  Returns NULL on error. */
static char *pt_read_file(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return nullptr;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return nullptr; }
    char *buf = (char *)malloc((size_t)sz);
    if (!buf) { fclose(f); return nullptr; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) { free(buf); return nullptr; }
    *out_size = (size_t)sz;
    return buf;
}

/* Write raw bytes to a file.  Returns 0 on success. */
static int pt_write_file(const char *path, const void *data, size_t size) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t wr = fwrite(data, 1, size, f);
    fclose(f);
    return (wr == size) ? 0 : -1;
}

/* Store the first 4 dims of an nvinfer1::Dims into a plain int array. */
static void pt_store_dims(const nvinfer1::Dims &dims, int out[4]) {
    for (int i = 0; i < 4; ++i) {
        out[i] = (i < dims.nbDims) ? dims.d[i] : 0;
    }
}

/* Compute the element count for a binding, replacing dimension 0 with batch_size. */
static int64_t pt_binding_element_count(const nvinfer1::Dims &dims, int batch_size) {
    int64_t count = (int64_t)batch_size;
    for (int i = 1; i < dims.nbDims; ++i) {
        count *= dims.d[i];
    }
    return count;
}

/* Size in bytes of a TensorRT data type. */
static int pt_dtype_size(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        default: return 4;
    }
}

/* ============================================================================
 * Deserialize a cached .engine file.
 * ============================================================================ */

static int pt_deserialize_engine(PT_TrtEngine *eng) {
    size_t size = 0;
    char *data = pt_read_file(eng->engine_path, &size);
    if (!data) {
        fprintf(stderr, "[TRT] Failed to read cached engine: %s\n", eng->engine_path);
        return PT_ERR_ENGINE_BUILD;
    }

    nvinfer1::ICudaEngine *cuda_engine =
        g_runtime->deserializeCudaEngine(data, size);
    free(data);

    if (!cuda_engine) {
        fprintf(stderr, "[TRT] Failed to deserialize engine: %s\n", eng->engine_path);
        return PT_ERR_ENGINE_BUILD;
    }

    nvinfer1::IExecutionContext *ctx = cuda_engine->createExecutionContext();
    if (!ctx) {
        delete cuda_engine;
        fprintf(stderr, "[TRT] Failed to create execution context\n");
        return PT_ERR_ENGINE_BUILD;
    }

    eng->engine  = cuda_engine;
    eng->context = ctx;
    eng->runtime = g_runtime;

    /* Query binding indices and shapes. */
    int nb = cuda_engine->getNbIOTensors();
    eng->input_index  = -1;
    eng->output_index = -1;

    for (int i = 0; i < nb; ++i) {
        const char *name = cuda_engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = cuda_engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            eng->input_index = i;
            nvinfer1::Dims dims = cuda_engine->getTensorShape(name);
            pt_store_dims(dims, eng->input_dims);
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            eng->output_index = i;
            nvinfer1::Dims dims = cuda_engine->getTensorShape(name);
            pt_store_dims(dims, eng->output_dims);
        }
    }

    fprintf(stderr, "[TRT] Deserialized engine: %s\n", eng->engine_path);
    fprintf(stderr, "[TRT]   Input  dims: [%d, %d, %d, %d]\n",
            eng->input_dims[0], eng->input_dims[1],
            eng->input_dims[2], eng->input_dims[3]);
    fprintf(stderr, "[TRT]   Output dims: [%d, %d, %d, %d]\n",
            eng->output_dims[0], eng->output_dims[1],
            eng->output_dims[2], eng->output_dims[3]);

    return PT_OK;
}

/* ============================================================================
 * Build engine from ONNX, serialize to cache file.
 * ============================================================================ */

static int pt_build_from_onnx(PT_TrtEngine *eng, int max_batch, int use_fp16) {
    fprintf(stderr, "[TRT] Building engine from ONNX: %s\n", eng->onnx_path);
    fprintf(stderr, "[TRT]   max_batch=%d  fp16=%d\n", max_batch, use_fp16);

    /* Builder */
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(g_logger);
    if (!builder) {
        fprintf(stderr, "[TRT] Failed to create IBuilder\n");
        return PT_ERR_ENGINE_BUILD;
    }

    /* Network (explicit batch) */
    const uint32_t flags =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(flags);
    if (!network) {
        delete builder;
        fprintf(stderr, "[TRT] Failed to create INetworkDefinition\n");
        return PT_ERR_ENGINE_BUILD;
    }

    /* ONNX parser */
    nvonnxparser::IParser *parser =
        nvonnxparser::createParser(*network, g_logger);
    if (!parser) {
        delete network;
        delete builder;
        fprintf(stderr, "[TRT] Failed to create ONNX parser\n");
        return PT_ERR_ENGINE_BUILD;
    }

    if (!parser->parseFromFile(eng->onnx_path,
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        fprintf(stderr, "[TRT] Failed to parse ONNX file: %s\n", eng->onnx_path);
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            fprintf(stderr, "[TRT]   %s\n", parser->getError(i)->desc());
        }
        delete parser;
        delete network;
        delete builder;
        return PT_ERR_ENGINE_BUILD;
    }

    /* Builder config */
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    if (!config) {
        delete parser;
        delete network;
        delete builder;
        fprintf(stderr, "[TRT] Failed to create IBuilderConfig\n");
        return PT_ERR_ENGINE_BUILD;
    }

    /* 1 GB workspace */
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               (size_t)1 << 30);

    /* FP16 */
    if (use_fp16) {
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            fprintf(stderr, "[TRT]   FP16 enabled\n");
        } else {
            fprintf(stderr, "[TRT]   FP16 requested but not supported on this GPU -- using FP32\n");
        }
    }

    /* Optimization profile with dynamic batch dimension. */
    nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
    if (!profile) {
        delete config;
        delete parser;
        delete network;
        delete builder;
        fprintf(stderr, "[TRT] Failed to create optimization profile\n");
        return PT_ERR_ENGINE_BUILD;
    }

    /* Set min/opt/max for every input tensor. */
    int opt_batch = max_batch / 2;
    if (opt_batch < 1) opt_batch = 1;

    for (int i = 0; i < network->getNbInputs(); ++i) {
        nvinfer1::ITensor *input = network->getInput(i);
        const char *name = input->getName();
        nvinfer1::Dims dims = input->getDimensions();

        /* dims.d[0] is the batch dimension (-1 for dynamic). */
        nvinfer1::Dims min_dims = dims;
        nvinfer1::Dims opt_dims = dims;
        nvinfer1::Dims max_dims = dims;

        min_dims.d[0] = 1;
        opt_dims.d[0] = opt_batch;
        max_dims.d[0] = max_batch;

        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, min_dims);
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, opt_dims);
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, max_dims);

        fprintf(stderr, "[TRT]   Input '%s': min=[%d,...], opt=[%d,...], max=[%d,...]\n",
                name, 1, opt_batch, max_batch);
    }

    config->addOptimizationProfile(profile);

    /* Build serialized engine. */
    fprintf(stderr, "[TRT] Building engine... (this may take a few minutes)\n");

    clock_t t0 = clock();

    nvinfer1::IHostMemory *serialized =
        builder->buildSerializedNetwork(*network, *config);

    clock_t t1 = clock();
    double build_seconds = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    delete parser;
    delete network;
    delete config;
    delete builder;

    if (!serialized || serialized->size() == 0) {
        fprintf(stderr, "[TRT] Engine build failed\n");
        if (serialized) delete serialized;
        return PT_ERR_ENGINE_BUILD;
    }

    fprintf(stderr, "[TRT] Engine built in %.1f seconds (%.1f MB)\n",
            build_seconds, (double)serialized->size() / (1024.0 * 1024.0));

    /* Write to cache -- create the parent directory first. */
    {
        char dir[512];
        strncpy(dir, eng->engine_path, sizeof(dir) - 1);
        dir[sizeof(dir) - 1] = '\0';
        /* Find last separator. */
        char *last_sep = nullptr;
        for (char *p = dir; *p; ++p) {
            if (*p == '/' || *p == '\\') last_sep = p;
        }
        if (last_sep) {
            *last_sep = '\0';
            pt_mkdir(dir);
        }
    }

    if (pt_write_file(eng->engine_path, serialized->data(), serialized->size()) != 0) {
        fprintf(stderr, "[TRT] Warning: failed to write engine cache: %s\n", eng->engine_path);
        /* Non-fatal -- we still have the serialized engine in memory. */
    } else {
        fprintf(stderr, "[TRT] Cached engine to: %s\n", eng->engine_path);
    }

    /* Deserialize into a live engine. */
    nvinfer1::ICudaEngine *cuda_engine =
        g_runtime->deserializeCudaEngine(serialized->data(), serialized->size());
    delete serialized;

    if (!cuda_engine) {
        fprintf(stderr, "[TRT] Failed to deserialize freshly built engine\n");
        return PT_ERR_ENGINE_BUILD;
    }

    nvinfer1::IExecutionContext *ctx = cuda_engine->createExecutionContext();
    if (!ctx) {
        delete cuda_engine;
        fprintf(stderr, "[TRT] Failed to create execution context\n");
        return PT_ERR_ENGINE_BUILD;
    }

    eng->engine  = cuda_engine;
    eng->context = ctx;
    eng->runtime = g_runtime;

    /* Record binding info. */
    int nb = cuda_engine->getNbIOTensors();
    eng->input_index  = -1;
    eng->output_index = -1;

    for (int i = 0; i < nb; ++i) {
        const char *name = cuda_engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = cuda_engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            eng->input_index = i;
            nvinfer1::Dims dims = cuda_engine->getTensorShape(name);
            pt_store_dims(dims, eng->input_dims);
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            eng->output_index = i;
            nvinfer1::Dims dims = cuda_engine->getTensorShape(name);
            pt_store_dims(dims, eng->output_dims);
        }
    }

    fprintf(stderr, "[TRT]   Input  dims: [%d, %d, %d, %d]\n",
            eng->input_dims[0], eng->input_dims[1],
            eng->input_dims[2], eng->input_dims[3]);
    fprintf(stderr, "[TRT]   Output dims: [%d, %d, %d, %d]\n",
            eng->output_dims[0], eng->output_dims[1],
            eng->output_dims[2], eng->output_dims[3]);

    return PT_OK;
}

/* ============================================================================
 * Public API -- TensorRT available
 * ============================================================================ */

extern "C" int pt_trt_init(void) {
    if (g_initialized) return PT_OK;

    g_runtime = nvinfer1::createInferRuntime(g_logger);
    if (!g_runtime) {
        fprintf(stderr, "[TRT] Failed to create TensorRT runtime\n");
        return PT_ERR_TENSORRT;
    }

    /* Log TensorRT version. */
    int ver = getInferLibVersion();
    fprintf(stderr, "[TRT] TensorRT %d.%d.%d initialized\n",
            ver / 1000, (ver / 100) % 10, ver % 100);

    g_initialized = 1;
    return PT_OK;
}

extern "C" void pt_trt_shutdown(void) {
    if (g_runtime) {
        delete g_runtime;
        g_runtime = nullptr;
    }
    g_initialized = 0;
}

extern "C" int pt_trt_build_engine(PT_TrtEngine *engine,
                                   const char *onnx_path,
                                   const char *cache_dir,
                                   int max_batch,
                                   int use_fp16) {
    if (!engine || !onnx_path || !cache_dir) return PT_ERR_INVALID_PARAM;
    if (max_batch < 1) return PT_ERR_INVALID_PARAM;

    if (!g_initialized) {
        fprintf(stderr, "[TRT] Runtime not initialized -- call pt_trt_init() first\n");
        return PT_ERR_NOT_INITIALIZED;
    }

    /* Zero the output struct. */
    memset(engine, 0, sizeof(*engine));
    engine->max_batch_size = max_batch;

    /* Store ONNX path. */
    strncpy(engine->onnx_path, onnx_path, sizeof(engine->onnx_path) - 1);
    engine->onnx_path[sizeof(engine->onnx_path) - 1] = '\0';

    /* Verify ONNX file exists. */
    if (!pt_file_exists(onnx_path)) {
        fprintf(stderr, "[TRT] ONNX file not found: %s\n", onnx_path);
        return PT_ERR_FILE_NOT_FOUND;
    }

    /* Build cache key: {model_name}_{sm}_{max_batch}_{fp16}.engine */
    char model_name[256];
    pt_extract_model_name(onnx_path, model_name, sizeof(model_name));

    char sm[16];
    if (pt_get_sm_version(sm, sizeof(sm)) != 0) {
        fprintf(stderr, "[TRT] Failed to query GPU compute capability\n");
        return PT_ERR_CUDA;
    }

    snprintf(engine->engine_path, sizeof(engine->engine_path),
             "%s/%s_sm%s_b%d_%s.engine",
             cache_dir, model_name, sm, max_batch,
             use_fp16 ? "fp16" : "fp32");

    fprintf(stderr, "[TRT] Engine cache path: %s\n", engine->engine_path);

    /* Check cache: exists AND newer than the ONNX source. */
    time_t onnx_mtime   = pt_file_mtime(onnx_path);
    time_t engine_mtime  = pt_file_mtime(engine->engine_path);

    if (engine_mtime > 0 && engine_mtime >= onnx_mtime) {
        fprintf(stderr, "[TRT] Loading cached engine\n");
        return pt_deserialize_engine(engine);
    }

    /* Cache miss or stale -- build from ONNX. */
    return pt_build_from_onnx(engine, max_batch, use_fp16);
}

extern "C" void pt_trt_destroy_engine(PT_TrtEngine *engine) {
    if (!engine) return;

    if (engine->context) {
        nvinfer1::IExecutionContext *ctx =
            static_cast<nvinfer1::IExecutionContext *>(engine->context);
        delete ctx;
        engine->context = nullptr;
    }

    if (engine->engine) {
        nvinfer1::ICudaEngine *eng =
            static_cast<nvinfer1::ICudaEngine *>(engine->engine);
        delete eng;
        engine->engine = nullptr;
    }

    /* runtime is global -- do not destroy it here. */
    engine->runtime = nullptr;
}

extern "C" int pt_trt_infer(PT_TrtEngine *engine,
                            void *input_gpu,
                            void *output_gpu,
                            int batch_size,
                            cudaStream_t stream) {
    if (!engine || !engine->context || !engine->engine) {
        fprintf(stderr, "[TRT] Infer called on invalid engine\n");
        return PT_ERR_INFERENCE;
    }
    if (batch_size < 1 || batch_size > engine->max_batch_size) {
        fprintf(stderr, "[TRT] Infer: batch_size %d out of range [1, %d]\n",
                batch_size, engine->max_batch_size);
        return PT_ERR_INVALID_PARAM;
    }

    nvinfer1::ICudaEngine *cuda_engine =
        static_cast<nvinfer1::ICudaEngine *>(engine->engine);
    nvinfer1::IExecutionContext *ctx =
        static_cast<nvinfer1::IExecutionContext *>(engine->context);

    /* Set the actual batch dimension on input tensors. */
    int nb = cuda_engine->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char *name = cuda_engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = cuda_engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            nvinfer1::Dims dims = cuda_engine->getTensorShape(name);
            dims.d[0] = batch_size;
            if (!ctx->setInputShape(name, dims)) {
                fprintf(stderr, "[TRT] Failed to set input shape for '%s'\n", name);
                return PT_ERR_INFERENCE;
            }
        }
    }

    /* Set tensor addresses. */
    for (int i = 0; i < nb; ++i) {
        const char *name = cuda_engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = cuda_engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            if (!ctx->setTensorAddress(name, input_gpu)) {
                fprintf(stderr, "[TRT] Failed to set input address for '%s'\n", name);
                return PT_ERR_INFERENCE;
            }
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            if (!ctx->setTensorAddress(name, output_gpu)) {
                fprintf(stderr, "[TRT] Failed to set output address for '%s'\n", name);
                return PT_ERR_INFERENCE;
            }
        }
    }

    /* Enqueue on the CUDA stream. */
    if (!ctx->enqueueV3(stream)) {
        fprintf(stderr, "[TRT] enqueueV3 failed\n");
        return PT_ERR_INFERENCE;
    }

    return PT_OK;
}

extern "C" int pt_trt_get_binding_size(PT_TrtEngine *engine,
                                       int binding_index,
                                       int batch_size) {
    if (!engine || !engine->engine) return 0;
    if (batch_size < 1) return 0;

    nvinfer1::ICudaEngine *cuda_engine =
        static_cast<nvinfer1::ICudaEngine *>(engine->engine);

    int nb = cuda_engine->getNbIOTensors();
    if (binding_index < 0 || binding_index >= nb) return 0;

    const char *name = cuda_engine->getIOTensorName(binding_index);
    nvinfer1::Dims dims = cuda_engine->getTensorShape(name);
    nvinfer1::DataType dtype = cuda_engine->getTensorDataType(name);

    int64_t elems = pt_binding_element_count(dims, batch_size);
    int elem_bytes = pt_dtype_size(dtype);

    return (int)(elems * elem_bytes);
}

#else /* PT_HAS_TENSORRT == 0 */

/* ============================================================================
 * Stub implementation -- TensorRT not available at compile time.
 * Every function logs a clear message and returns an error code.
 * ============================================================================ */

extern "C" int pt_trt_init(void) {
    fprintf(stderr, "[TRT] TensorRT is not available (headers not found at compile time)\n");
    return PT_ERR_TENSORRT;
}

extern "C" void pt_trt_shutdown(void) {
    /* Nothing to do. */
}

extern "C" int pt_trt_build_engine(PT_TrtEngine *engine,
                                   const char *onnx_path,
                                   const char *cache_dir,
                                   int max_batch,
                                   int use_fp16) {
    (void)engine; (void)onnx_path; (void)cache_dir;
    (void)max_batch; (void)use_fp16;
    fprintf(stderr, "[TRT] TensorRT is not available (headers not found at compile time)\n");
    return PT_ERR_TENSORRT;
}

extern "C" void pt_trt_destroy_engine(PT_TrtEngine *engine) {
    (void)engine;
}

extern "C" int pt_trt_infer(PT_TrtEngine *engine,
                            void *input_gpu,
                            void *output_gpu,
                            int batch_size,
                            cudaStream_t stream) {
    (void)engine; (void)input_gpu; (void)output_gpu;
    (void)batch_size; (void)stream;
    fprintf(stderr, "[TRT] TensorRT is not available (headers not found at compile time)\n");
    return PT_ERR_TENSORRT;
}

extern "C" int pt_trt_get_binding_size(PT_TrtEngine *engine,
                                       int binding_index,
                                       int batch_size) {
    (void)engine; (void)binding_index; (void)batch_size;
    return 0;
}

#endif /* PT_HAS_TENSORRT */
