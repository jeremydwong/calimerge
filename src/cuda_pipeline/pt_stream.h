/*
 * pt_stream.h - Real-time streaming API for the CUDA pose tracking pipeline.
 *
 * Accepts live BGR camera frames and returns 3D tracked poses synchronously.
 * Designed for use with cm_capture_synced() from the native camera module.
 *
 * Usage:
 *   1. pt_stream_create()        — allocate GPU resources, build TensorRT engines
 *   2. pt_stream_process_frame() — submit one sync frame, get 3D results back
 *   3. pt_stream_destroy()       — free everything
 *
 * All GPU memory is preallocated at create time (single-allocation arena).
 * TensorRT engines are built once and cached to disk.
 * Processing is synchronous: submit frame, get result, ~12-18ms per call.
 *
 * Style: Plain C structs + free functions. No classes, no templates, no STL.
 */

#ifndef PT_STREAM_H
#define PT_STREAM_H

#include "pt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Configuration
 * ============================================================================ */

typedef struct {
    /* Model paths */
    char yolo_onnx_path[512];
    char vitpose_onnx_path[512];
    char engine_cache_dir[512];
    char calibration_toml_path[512];

    /* Camera parameters (all cameras must share dimensions) */
    int num_cameras;
    int frame_width;
    int frame_height;

    /* Processing parameters */
    int   max_persons;          /* max tracked persons (default 2) */
    float person_confidence;    /* YOLO detection threshold (default 0.1) */
    float keypoint_confidence;  /* min keypoint confidence (default 0.1) */
    float epipolar_threshold;   /* max epipolar distance pixels (default 10.0) */
    float max_track_distance;   /* max 3D COM distance for track match (default 0.15) */
    int   track_patience;       /* frames before losing track (default 30) */
    int   use_fp16_yolo;        /* 1=FP16 YOLO input, 0=FP32 (default 1) */

    /* Callbacks (optional) */
    void (*log_callback)(const char *message, void *user_data);
    void *callback_user_data;
} PT_StreamConfig;

/* ============================================================================
 * Input: one camera's BGR frame
 * ============================================================================ */

typedef struct {
    const uint8_t *pixels;      /* BGR8, row-major, host memory */
    int width;
    int height;
    int stride;                 /* bytes per row (typically width * 3) */
    int port;                   /* camera port index (for calibration lookup) */
} PT_StreamFrame;

/* ============================================================================
 * Input: one synchronized frame set from all cameras
 * ============================================================================ */

typedef struct {
    PT_StreamFrame frames[PT_MAX_CAMERAS];
    int num_frames;             /* number of valid camera frames */
    uint64_t sync_index;        /* monotonic counter from cm_capture_synced */
} PT_StreamFrameSet;

/* ============================================================================
 * Output: one person's 3D pose
 * ============================================================================ */

typedef struct {
    int    person_id;
    double keypoints_3d[PT_NUM_KEYPOINTS][3];   /* x, y, z in meters */
    int    keypoints_valid[PT_NUM_KEYPOINTS];    /* 1=triangulated, 0=missing */
    double com_3d[3];                            /* center of mass */
    int    com_valid;                            /* 1 if COM computed */
    int    num_views;                            /* cameras that contributed */
} PT_StreamPerson;

/* ============================================================================
 * Output: full result for one sync frame
 * ============================================================================ */

typedef struct {
    PT_StreamPerson persons[PT_MAX_TRACKS];
    int num_persons;                /* active tracked persons in this frame */
    uint64_t sync_index;
    double processing_time_ms;      /* wall-clock time for this frame */
} PT_StreamResult;

/* ============================================================================
 * Statistics: cumulative timing breakdown
 * ============================================================================ */

typedef struct {
    double upload_ms;
    double yolo_ms;
    double vitpose_ms;
    double matching_ms;
    double triangulation_ms;
    double tracking_ms;
    double total_ms;
    int    frames_processed;
} PT_StreamStats;

/* ============================================================================
 * Opaque handle
 * ============================================================================ */

typedef struct PT_Stream PT_Stream;

/* ============================================================================
 * API
 * ============================================================================ */

/*
 * Create a streaming pipeline. Allocates GPU arena, builds TensorRT engines,
 * loads calibration. This is expensive (~0.5-30s depending on engine cache)
 * but only done once.
 */
int pt_stream_create(PT_Stream **out, const PT_StreamConfig *config);

/*
 * Destroy the streaming pipeline and free all GPU resources.
 */
void pt_stream_destroy(PT_Stream *s);

/*
 * Process one synchronized frame set. Synchronous: blocks until the result
 * is ready (~12-18ms for 3 cameras at 640x480).
 *
 * Returns PT_OK on success, error code on failure.
 */
int pt_stream_process_frame(PT_Stream *s,
                            const PT_StreamFrameSet *input,
                            PT_StreamResult *out_result);

/*
 * Get cumulative timing statistics.
 */
void pt_stream_get_stats(const PT_Stream *s, PT_StreamStats *out);

/*
 * Reset all tracking state (person IDs, track history). Call this when
 * the scene changes or calibration is updated.
 */
void pt_stream_reset_tracks(PT_Stream *s);

/*
 * Export accumulated track history to CSV files (for testing/verification).
 * Delegates to pt_export_csv() using the stream's internal track state.
 *
 * Output files: {output_base_path}_person{id}.csv
 */
int pt_stream_export_csv(const PT_Stream *s, const char *output_base_path);

#ifdef __cplusplus
}
#endif

#endif /* PT_STREAM_H */
