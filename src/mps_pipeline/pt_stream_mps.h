/*
 * pt_stream_mps.h - Streaming API for the MPS pose tracking pipeline.
 *
 * Identical C API contract to pt_stream.h (CUDA pipeline).
 * Accepts live BGR camera frames and returns 3D tracked poses synchronously.
 *
 * Uses CoreML for inference, vImage for preprocessing, and shared C code
 * for matching/triangulation/tracking.
 *
 * Style: Plain C structs + free functions.
 */

#ifndef PT_STREAM_MPS_H
#define PT_STREAM_MPS_H

#include "../pt_shared/pt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Re-use the same config/input/output structs as the CUDA streaming API.
 * These are defined identically so the Python binding can switch backends
 * with only a DLL/dylib name change. */

typedef struct {
    char yolo_model_path[512];       /* .mlpackage or .mlmodelc */
    char vitpose_model_path[512];    /* .mlpackage or .mlmodelc */
    char calibration_toml_path[512];

    int num_cameras;
    int frame_width;
    int frame_height;

    int   max_persons;
    float person_confidence;
    float keypoint_confidence;
    float epipolar_threshold;
    float max_track_distance;
    int   track_patience;

    void (*log_callback)(const char *message, void *user_data);
    void *callback_user_data;
} PT_MPS_StreamConfig;

typedef struct {
    const uint8_t *pixels;   /* BGR8, row-major, host memory */
    int width;
    int height;
    int stride;
    int port;
} PT_MPS_StreamFrame;

typedef struct {
    PT_MPS_StreamFrame frames[PT_MAX_CAMERAS];
    int num_frames;
    uint64_t sync_index;
} PT_MPS_StreamFrameSet;

typedef struct {
    int    person_id;
    double keypoints_3d[PT_NUM_KEYPOINTS][3];
    int    keypoints_valid[PT_NUM_KEYPOINTS];
    double com_3d[3];
    int    com_valid;
    int    num_views;
} PT_MPS_StreamPerson;

typedef struct {
    PT_MPS_StreamPerson persons[PT_MAX_TRACKS];
    int num_persons;
    uint64_t sync_index;
    double processing_time_ms;
} PT_MPS_StreamResult;

typedef struct {
    double coreml_yolo_ms;
    double coreml_vitpose_ms;
    double preprocess_ms;
    double matching_ms;
    double triangulation_ms;
    double tracking_ms;
    double total_ms;
    int    frames_processed;
} PT_MPS_StreamStats;

/* Opaque handle */
typedef struct PT_MPS_Stream PT_MPS_Stream;

/* ============================================================================
 * API — mirrors pt_stream.h exactly
 * ============================================================================ */

int  pt_mps_stream_create(PT_MPS_Stream **out, const PT_MPS_StreamConfig *config);
void pt_mps_stream_destroy(PT_MPS_Stream *s);
int  pt_mps_stream_process_frame(PT_MPS_Stream *s,
                                  const PT_MPS_StreamFrameSet *input,
                                  PT_MPS_StreamResult *out_result);
void pt_mps_stream_get_stats(const PT_MPS_Stream *s, PT_MPS_StreamStats *out);
void pt_mps_stream_reset_tracks(PT_MPS_Stream *s);
int  pt_mps_stream_export_csv(const PT_MPS_Stream *s, const char *output_base_path);

#ifdef __cplusplus
}
#endif

#endif /* PT_STREAM_MPS_H */
