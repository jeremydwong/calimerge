/*
 * pt_arena.cu - Single-allocation GPU arena for the pose tracking pipeline.
 *
 * All GPU memory comes from ONE cudaMalloc.  All pinned host memory comes
 * from ONE cudaMallocHost.  Every pointer in PT_GpuArena is carved out of
 * these two blocks via pointer arithmetic.
 *
 * Alignment: every sub-allocation is rounded up to 256 bytes so that all
 * buffers are naturally aligned for coalesced access and texture loads.
 */

#include "pt_arena.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

#define ARENA_ALIGN 256

static inline size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/* Carve `bytes` out of the arena, advance `cursor`, return the old cursor. */
static inline uint8_t *arena_alloc(uint8_t **cursor, size_t bytes) {
    uint8_t *ptr = *cursor;
    *cursor += align_up(bytes, ARENA_ALIGN);
    return ptr;
}

/* Same, but returns a float*. */
static inline float *arena_alloc_f(uint8_t **cursor, size_t bytes) {
    return (float *)arena_alloc(cursor, bytes);
}

/* Same, but returns an int*. */
static inline int *arena_alloc_i(uint8_t **cursor, size_t bytes) {
    return (int *)arena_alloc(cursor, bytes);
}

/* -------------------------------------------------------------------------
 * Size computation
 * ------------------------------------------------------------------------- */

/*
 * Compute the total GPU bytes and total host bytes required.  The logic is
 * duplicated (once for sizing, once for pointer assignment) to avoid keeping
 * a separate size table.  We just run through the same sequence twice.
 */

static void compute_sizes(int num_cameras,
                          int frame_width,
                          int frame_height,
                          int batch_size,
                          size_t *out_gpu_bytes,
                          size_t *out_host_bytes) {
    int max_images  = batch_size * num_cameras;
    int max_crops   = max_images * PT_MAX_DETECTIONS;

    size_t nv12_bytes = (size_t)frame_width * frame_height * 3 / 2;  /* Y + UV */
    size_t bgr_bytes  = (size_t)frame_width * frame_height * 3;

    size_t gpu = 0;

    /* Decode buffers: double-buffered per camera */
    for (int buf = 0; buf < PT_PIPELINE_DEPTH; buf++) {
        for (int cam = 0; cam < num_cameras; cam++) {
            gpu += align_up(nv12_bytes, ARENA_ALIGN);
            gpu += align_up(bgr_bytes,  ARENA_ALIGN);
        }
    }

    /* YOLO input: (max_images, 3, 640, 640) fp16  --  __half is 2 bytes */
    gpu += align_up((size_t)max_images * 3 * PT_YOLO_INPUT_H * PT_YOLO_INPUT_W * 2, ARENA_ALIGN);

    /* YOLO output: (max_images, 300, 6) fp32 */
    gpu += align_up((size_t)max_images * PT_YOLO_MAX_RAW_DETS * 6 * sizeof(float), ARENA_ALIGN);

    /* Filtered detection boxes: (max_images, MAX_DET, 4) fp32 */
    gpu += align_up((size_t)max_images * PT_MAX_DETECTIONS * 4 * sizeof(float), ARENA_ALIGN);

    /* Filtered detection scores: (max_images, MAX_DET) fp32 */
    gpu += align_up((size_t)max_images * PT_MAX_DETECTIONS * sizeof(float), ARENA_ALIGN);

    /* Detection counts: (max_images,) int */
    gpu += align_up((size_t)max_images * sizeof(int), ARENA_ALIGN);

    /* VitPose input: (max_crops, 3, 256, 192) fp32 */
    gpu += align_up((size_t)max_crops * 3 * PT_VITPOSE_INPUT_H * PT_VITPOSE_INPUT_W * sizeof(float), ARENA_ALIGN);

    /* VitPose affine: (max_crops, 2, 3) fp32 */
    gpu += align_up((size_t)max_crops * 2 * 3 * sizeof(float), ARENA_ALIGN);

    /* VitPose heatmaps: (max_crops, 52, 64, 48) fp32 */
    gpu += align_up((size_t)max_crops * PT_NUM_KEYPOINTS * PT_VITPOSE_HEATMAP_H * PT_VITPOSE_HEATMAP_W * sizeof(float), ARENA_ALIGN);

    /* Keypoints 2D: (max_crops, 52, 3) fp32 */
    gpu += align_up((size_t)max_crops * PT_NUM_KEYPOINTS * 3 * sizeof(float), ARENA_ALIGN);

    *out_gpu_bytes = gpu;

    /* Pinned host memory */
    size_t host = 0;
    host += align_up((size_t)max_crops * PT_NUM_KEYPOINTS * 3 * sizeof(float), ARENA_ALIGN);   /* keypoints_2d */
    host += align_up((size_t)max_images * PT_MAX_DETECTIONS * 4 * sizeof(float), ARENA_ALIGN);  /* detection_boxes */
    host += align_up((size_t)max_images * PT_MAX_DETECTIONS * sizeof(float), ARENA_ALIGN);      /* detection_scores */
    host += align_up((size_t)max_images * sizeof(int), ARENA_ALIGN);                            /* detection_counts */

    *out_host_bytes = host;
}

/* -------------------------------------------------------------------------
 * pt_arena_init
 * ------------------------------------------------------------------------- */

extern "C"
int pt_arena_init(PT_GpuArena *arena,
                  int num_cameras,
                  int frame_width,
                  int frame_height,
                  int batch_size) {
    /* Validate inputs */
    if (!arena) return PT_ERR_INVALID_PARAM;
    if (num_cameras <= 0 || num_cameras > PT_MAX_CAMERAS) return PT_ERR_INVALID_PARAM;
    if (frame_width <= 0 || frame_height <= 0) return PT_ERR_INVALID_PARAM;
    if (batch_size <= 0 || batch_size > PT_BATCH_SIZE_MAX) return PT_ERR_INVALID_PARAM;

    memset(arena, 0, sizeof(PT_GpuArena));

    arena->num_cameras          = num_cameras;
    arena->frame_width          = frame_width;
    arena->frame_height         = frame_height;
    arena->batch_size           = batch_size;
    arena->max_images_per_batch = batch_size * num_cameras;
    arena->max_crops_per_batch  = arena->max_images_per_batch * PT_MAX_DETECTIONS;

    /* Compute total sizes */
    size_t gpu_bytes  = 0;
    size_t host_bytes = 0;
    compute_sizes(num_cameras, frame_width, frame_height, batch_size,
                  &gpu_bytes, &host_bytes);

    /* Single GPU allocation */
    cudaError_t err = cudaMalloc(&arena->gpu_base, gpu_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[pt_arena] cudaMalloc failed for %zu bytes: %s\n",
                gpu_bytes, cudaGetErrorString(err));
        return PT_ERR_OUT_OF_MEMORY;
    }
    arena->gpu_total_bytes = gpu_bytes;

    /* Single pinned host allocation */
    err = cudaMallocHost(&arena->host_base, host_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[pt_arena] cudaMallocHost failed for %zu bytes: %s\n",
                host_bytes, cudaGetErrorString(err));
        cudaFree(arena->gpu_base);
        arena->gpu_base = NULL;
        return PT_ERR_OUT_OF_MEMORY;
    }
    arena->host_total_bytes = host_bytes;

    /* Zero both allocations */
    cudaMemset(arena->gpu_base, 0, gpu_bytes);
    memset(arena->host_base, 0, host_bytes);

    /* --- Assign GPU pointers --- */
    uint8_t *cursor = (uint8_t *)arena->gpu_base;

    size_t nv12_bytes = (size_t)frame_width * frame_height * 3 / 2;
    size_t bgr_bytes  = (size_t)frame_width * frame_height * 3;

    for (int buf = 0; buf < PT_PIPELINE_DEPTH; buf++) {
        for (int cam = 0; cam < num_cameras; cam++) {
            arena->decoded_nv12[buf][cam] = arena_alloc(&cursor, nv12_bytes);
            arena->decoded_bgr[buf][cam]  = arena_alloc(&cursor, bgr_bytes);
        }
    }

    int max_images = arena->max_images_per_batch;
    int max_crops  = arena->max_crops_per_batch;

    /* YOLO input: (max_images, 3, 640, 640) fp16 */
    arena->yolo_input = (void *)arena_alloc(&cursor,
        (size_t)max_images * 3 * PT_YOLO_INPUT_H * PT_YOLO_INPUT_W * 2);

    /* YOLO output: (max_images, 300, 6) fp32 */
    arena->yolo_output = arena_alloc_f(&cursor,
        (size_t)max_images * PT_YOLO_MAX_RAW_DETS * 6 * sizeof(float));

    /* Filtered detection boxes: (max_images, MAX_DET, 4) */
    arena->detection_boxes = arena_alloc_f(&cursor,
        (size_t)max_images * PT_MAX_DETECTIONS * 4 * sizeof(float));

    /* Filtered detection scores: (max_images, MAX_DET) */
    arena->detection_scores = arena_alloc_f(&cursor,
        (size_t)max_images * PT_MAX_DETECTIONS * sizeof(float));

    /* Detection counts: (max_images,) */
    arena->detection_counts = arena_alloc_i(&cursor,
        (size_t)max_images * sizeof(int));

    /* VitPose input: (max_crops, 3, 256, 192) fp32 */
    arena->vitpose_input = arena_alloc_f(&cursor,
        (size_t)max_crops * 3 * PT_VITPOSE_INPUT_H * PT_VITPOSE_INPUT_W * sizeof(float));

    /* VitPose affine: (max_crops, 2, 3) fp32 */
    arena->vitpose_affine = arena_alloc_f(&cursor,
        (size_t)max_crops * 2 * 3 * sizeof(float));

    /* VitPose heatmaps: (max_crops, 52, 64, 48) fp32 */
    arena->vitpose_heatmaps = arena_alloc_f(&cursor,
        (size_t)max_crops * PT_NUM_KEYPOINTS * PT_VITPOSE_HEATMAP_H * PT_VITPOSE_HEATMAP_W * sizeof(float));

    /* Keypoints 2D: (max_crops, 52, 3) fp32 */
    arena->keypoints_2d = arena_alloc_f(&cursor,
        (size_t)max_crops * PT_NUM_KEYPOINTS * 3 * sizeof(float));

    /* --- Assign pinned host pointers --- */
    uint8_t *hcursor = (uint8_t *)arena->host_base;

    arena->host_keypoints_2d = arena_alloc_f(&hcursor,
        (size_t)max_crops * PT_NUM_KEYPOINTS * 3 * sizeof(float));

    arena->host_detection_boxes = arena_alloc_f(&hcursor,
        (size_t)max_images * PT_MAX_DETECTIONS * 4 * sizeof(float));

    arena->host_detection_scores = arena_alloc_f(&hcursor,
        (size_t)max_images * PT_MAX_DETECTIONS * sizeof(float));

    arena->host_detection_counts = arena_alloc_i(&hcursor,
        (size_t)max_images * sizeof(int));

    /* Print summary */
    pt_arena_print_stats(arena);

    return PT_OK;
}

/* -------------------------------------------------------------------------
 * pt_arena_destroy
 * ------------------------------------------------------------------------- */

extern "C"
void pt_arena_destroy(PT_GpuArena *arena) {
    if (!arena) return;

    if (arena->gpu_base) {
        cudaFree(arena->gpu_base);
    }
    if (arena->host_base) {
        cudaFreeHost(arena->host_base);
    }

    memset(arena, 0, sizeof(PT_GpuArena));
}

/* -------------------------------------------------------------------------
 * pt_arena_print_stats
 * ------------------------------------------------------------------------- */

extern "C"
void pt_arena_print_stats(const PT_GpuArena *arena) {
    if (!arena) return;

    double gpu_mb  = (double)arena->gpu_total_bytes  / (1024.0 * 1024.0);
    double host_mb = (double)arena->host_total_bytes / (1024.0 * 1024.0);

    printf("[pt_arena] Session: %d cameras, %dx%d, batch=%d\n",
           arena->num_cameras, arena->frame_width, arena->frame_height,
           arena->batch_size);
    printf("[pt_arena]   max images/batch : %d\n", arena->max_images_per_batch);
    printf("[pt_arena]   max crops/batch  : %d\n", arena->max_crops_per_batch);
    printf("[pt_arena]   GPU allocation   : %zu bytes (%.1f MB)\n",
           arena->gpu_total_bytes, gpu_mb);
    printf("[pt_arena]   Host allocation  : %zu bytes (%.1f MB)\n",
           arena->host_total_bytes, host_mb);
    printf("[pt_arena]   Total            : %.1f MB\n", gpu_mb + host_mb);
}
