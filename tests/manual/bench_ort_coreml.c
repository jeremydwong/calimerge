/*
 * bench_ort_coreml.c — Compare ONNX Runtime (CoreML EP) vs our manual CoreML
 *                      for VitPose and YOLO inference latency.
 *
 * Build:
 *   clang -O2 -o build/bench_ort_coreml tests/manual/bench_ort_coreml.c \
 *     -I build/onnxruntime/include \
 *     -L build/onnxruntime/lib -lonnxruntime \
 *     -Wl,-rpath,@executable_path/../onnxruntime/lib
 *
 * Run:
 *   build/bench_ort_coreml <vitpose.onnx> [yolo.onnx] [--iterations N]
 */

#include "onnxruntime_c_api.h"
#include "coreml_provider_factory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>

static const OrtApi *g_ort = NULL;

#define ORT_CHECK(expr) do { \
    OrtStatus *_s = (expr); \
    if (_s) { \
        const char *msg = g_ort->GetErrorMessage(_s); \
        fprintf(stderr, "ORT error: %s\n  at %s:%d\n", msg, __FILE__, __LINE__); \
        g_ort->ReleaseStatus(_s); \
        return 1; \
    } \
} while(0)

#define ORT_CHECK_WARN(expr, label) do { \
    OrtStatus *_s = (expr); \
    if (_s) { \
        const char *msg = g_ort->GetErrorMessage(_s); \
        fprintf(stderr, "  [%s] ORT error (non-fatal): %s\n", label, msg); \
        g_ort->ReleaseStatus(_s); \
        goto cleanup; \
    } \
} while(0)

static double now_seconds(void) {
    static mach_timebase_info_data_t tb = {0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)(mach_absolute_time() * tb.numer / tb.denom) * 1e-9;
}

typedef struct {
    double mean_ms;
    double median_ms;
    double p95_ms;
    double min_ms;
    double max_ms;
} Stats;

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static Stats compute_stats(double *times, int n) {
    Stats s = {0};
    if (n <= 0) return s;

    qsort(times, n, sizeof(double), cmp_double);
    double sum = 0;
    for (int i = 0; i < n; i++) sum += times[i];
    s.mean_ms   = sum / n;
    s.median_ms = times[n / 2];
    s.p95_ms    = times[(int)(n * 0.95)];
    s.min_ms    = times[0];
    s.max_ms    = times[n - 1];
    return s;
}

static int bench_model(const char *label, const char *model_path,
                       int batch, int channels, int height, int width,
                       int iterations, int warmup, uint32_t coreml_flags) {
    printf("\n--- %s ---\n", label);
    printf("  model:  %s\n", model_path);
    printf("  input:  (%d, %d, %d, %d)\n", batch, channels, height, width);
    printf("  flags:  0x%x\n", coreml_flags);

    OrtEnv *env = NULL;
    ORT_CHECK(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bench", &env));

    OrtSessionOptions *opts = NULL;
    ORT_CHECK(g_ort->CreateSessionOptions(&opts));

    /* Enable CoreML EP */
    ORT_CHECK(OrtSessionOptionsAppendExecutionProvider_CoreML(opts, coreml_flags));

    /* Set to sequential execution for deterministic timing */
    ORT_CHECK(g_ort->SetIntraOpNumThreads(opts, 1));

    printf("  loading model...\n");
    double t_load = now_seconds();
    OrtSession *session = NULL;
    ORT_CHECK(g_ort->CreateSession(env, model_path, opts, &session));
    printf("  loaded in %.1f ms\n", (now_seconds() - t_load) * 1000.0);

    /* Allocate input */
    int64_t input_shape[] = {batch, channels, height, width};
    size_t input_size = (size_t)batch * channels * height * width;
    float *input_data = (float *)calloc(input_size, sizeof(float));

    /* Fill with small random values to avoid NaN/inf edge cases */
    for (size_t i = 0; i < input_size; i++)
        input_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    OrtMemoryInfo *mem_info = NULL;
    ORT_CHECK(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    OrtValue *input_tensor = NULL;
    ORT_CHECK(g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, input_data, input_size * sizeof(float),
        input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    /* Get input/output names */
    OrtAllocator *allocator = NULL;
    ORT_CHECK(g_ort->GetAllocatorWithDefaultOptions(&allocator));

    char *input_name = NULL;
    char *output_name = NULL;
    ORT_CHECK(g_ort->SessionGetInputName(session, 0, allocator, &input_name));
    ORT_CHECK(g_ort->SessionGetOutputName(session, 0, allocator, &output_name));
    printf("  input_name=%s  output_name=%s\n", input_name, output_name);

    const char *input_names[] = {input_name};
    const char *output_names[] = {output_name};

    /* Warmup */
    int ok = 1;
    double *times = NULL;
    printf("  warming up (%d iterations)...\n", warmup);
    for (int i = 0; i < warmup; i++) {
        OrtValue *output = NULL;
        OrtStatus *ws = g_ort->Run(session, NULL, input_names,
            (const OrtValue *const *)&input_tensor, 1, output_names, 1, &output);
        if (ws) {
            fprintf(stderr, "  warmup failed: %s\n", g_ort->GetErrorMessage(ws));
            g_ort->ReleaseStatus(ws);
            ok = 0;
            break;
        }
        g_ort->ReleaseValue(output);
    }

    if (!ok) goto cleanup;

    /* Timed runs */
    times = (double *)malloc(iterations * sizeof(double));
    printf("  running %d iterations...\n", iterations);

    for (int i = 0; i < iterations; i++) {
        double t0 = now_seconds();
        OrtValue *output = NULL;
        OrtStatus *rs = g_ort->Run(session, NULL, input_names,
            (const OrtValue *const *)&input_tensor, 1, output_names, 1, &output);
        times[i] = (now_seconds() - t0) * 1000.0;
        if (rs) {
            fprintf(stderr, "  run %d failed: %s\n", i, g_ort->GetErrorMessage(rs));
            g_ort->ReleaseStatus(rs);
            ok = 0;
            break;
        }
        g_ort->ReleaseValue(output);
    }

    if (ok) {
        Stats st = compute_stats(times, iterations);
        printf("\n  Results (%d iterations):\n", iterations);
        printf("    mean:   %7.2f ms\n", st.mean_ms);
        printf("    median: %7.2f ms\n", st.median_ms);
        printf("    p95:    %7.2f ms\n", st.p95_ms);
        printf("    min:    %7.2f ms\n", st.min_ms);
        printf("    max:    %7.2f ms\n", st.max_ms);
    } else {
        printf("\n  FAILED — could not complete inference\n");
    }

cleanup:
    free(times);
    free(input_data);
    if (input_tensor) g_ort->ReleaseValue(input_tensor);
    if (mem_info) g_ort->ReleaseMemoryInfo(mem_info);
    if (input_name) g_ort->AllocatorFree(allocator, input_name);
    if (output_name) g_ort->AllocatorFree(allocator, output_name);
    if (session) g_ort->ReleaseSession(session);
    if (opts) g_ort->ReleaseSessionOptions(opts);
    if (env) g_ort->ReleaseEnv(env);

    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <vitpose.onnx> [yolo.onnx] [--iterations N]\n", argv[0]);
        return 1;
    }

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "Failed to get ORT API\n");
        return 1;
    }

    const char *vitpose_path = argv[1];
    const char *yolo_path = NULL;
    int iterations = 100;
    int warmup = 10;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (!yolo_path) {
            yolo_path = argv[i];
        }
    }

    printf("ONNX Runtime + CoreML EP Benchmark\n");
    printf("===================================\n");
    printf("ORT API version: %d\n", ORT_API_VERSION);
    printf("Iterations: %d (warmup: %d)\n", iterations, warmup);

    int rc;

    /* VitPose: batch=16, 3x256x192 — try MLProgram first (better op coverage) */
    rc = bench_model("VitPose (CoreML EP, MLProgram, all units)",
                     vitpose_path, 16, 3, 256, 192,
                     iterations, warmup, COREML_FLAG_CREATE_MLPROGRAM);
    if (rc) {
        printf("  MLProgram failed, trying NeuralNetwork format...\n");
        rc = bench_model("VitPose (CoreML EP, NeuralNetwork, all units)",
                         vitpose_path, 16, 3, 256, 192,
                         iterations, warmup, COREML_FLAG_USE_NONE);
    }

    if (yolo_path) {
        rc = bench_model("YOLO (CoreML EP, MLProgram, all units)",
                         yolo_path, 2, 3, 640, 640,
                         iterations, warmup, COREML_FLAG_CREATE_MLPROGRAM);
        if (rc) {
            rc = bench_model("YOLO (CoreML EP, NeuralNetwork, all units)",
                             yolo_path, 2, 3, 640, 640,
                             iterations, warmup, COREML_FLAG_USE_NONE);
        }
    }

    printf("\n===================================\n");
    printf("Compare these numbers against the current pipeline:\n");
    printf("  Current VitPose (manual CoreML): ~100 ms/frame\n");
    printf("  Current YOLO (manual CoreML):    ~10.8 ms/frame\n");
    printf("===================================\n");

    return 0;
}
