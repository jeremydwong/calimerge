/*
 * pt_offline_mps.h - Offline batch pipeline for macOS Apple Silicon.
 *
 * Mirrors the CUDA pt_pipeline.h API, but tailored for the MPS / CoreML
 * backend. The offline pipeline is "decode + per-frame inference" using the
 * SAME streaming core that pt_stream_mps drives for live cameras — it just
 * pulls frames from AVAssetReader (pt_videodecode) instead of the camera
 * ring buffer.
 *
 * Why mirror pt_pipeline.h instead of inventing a fresh shape?
 *   - The Python binding can switch backends with only the dylib name and
 *     a couple of struct-field renames.
 *   - The output CSV schema (output_3d_poses_tracked.csv_personN.csv) is
 *     identical, so the GUI's _convert_outputs path is platform-blind.
 *
 * Style: Plain C structs + free functions.
 */

#ifndef PT_OFFLINE_MPS_H
#define PT_OFFLINE_MPS_H

#include "../pt_shared/pt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Configuration struct
 *
 * Field order is chosen to match the CUDA PT_PipelineConfig where it can,
 * so the Python binding mirrors the CUDA one closely. NOTE: this struct is
 * NOT identical to the CUDA PT_PipelineConfig — different enough that we
 * give it a distinct name and keep the Python binding separate.
 * ============================================================================ */

typedef struct {
    /* Input paths. video_paths[i] corresponds to ports[i]. */
    char video_paths[PT_MAX_CAMERAS][512];
    int  ports[PT_MAX_CAMERAS];   /* port number per video (parallel arrays) */
    int  num_cameras;

    char yolo_model_path[512];     /* .mlpackage / .mlmodelc */
    char vitpose_model_path[512];  /* .mlpackage / .mlmodelc */
    char calibration_toml_path[512];
    char frame_time_csv_path[512];
    char output_dir[512];

    /* Processing parameters */
    int   batch_size;              /* sync indices per CoreML batch (>=1) */
    int   skip_sync_indices;       /* process every Nth sync index (1 = all) */
    int   max_persons;
    float person_confidence;
    float keypoint_confidence;
    float epipolar_threshold;
    float max_track_distance;
    int   track_patience;

    /* Callbacks (called from the same thread as pt_mps_offline_run) */
    void (*progress_callback)(const char *step, float fraction, void *user_data);
    void (*log_callback)(const char *message, void *user_data);
    void *callback_user_data;
} PT_MPS_OfflineConfig;

/* ============================================================================
 * Statistics (returned by pt_mps_offline_get_stats)
 * ============================================================================ */

typedef struct {
    double total_seconds;
    double decode_seconds;
    double inference_seconds;     /* YOLO + VitPose, lumped — CoreML profiling
                                   * gives per-stage in PT_MPS_StreamStats */
    double matching_seconds;
    double triangulation_seconds;
    double export_seconds;
    int    frames_processed;
    int    persons_tracked;
} PT_MPS_OfflineStats;

/* Opaque handle */
typedef struct PT_MPS_Offline PT_MPS_Offline;

/* ============================================================================
 * API
 * ============================================================================ */

/*
 * Create a new offline pipeline. Models, calibration, and sync table are
 * loaded inside pt_mps_offline_run() so this is cheap.
 *
 * Returns PT_OK on success, error code otherwise.
 */
int  pt_mps_offline_create(PT_MPS_Offline **out, const PT_MPS_OfflineConfig *config);

/*
 * Free all resources held by the pipeline. Safe on NULL.
 */
void pt_mps_offline_destroy(PT_MPS_Offline *p);

/*
 * Run the full offline pipeline:
 *   1. Open AVAssetReader on each input video
 *   2. Load CoreML models + calibration + sync table
 *   3. For each sync index in stride order:
 *        - decode one frame per camera (BGR8)
 *        - call the same per-frame core that pt_stream_mps uses live
 *      (Optionally batched: batch_size > 1 stages multiple sync indices'
 *       decoded frames before running CoreML, for higher throughput.)
 *   4. Export per-track CSVs to <output_dir>/output_3d_poses_tracked.csv_personN.csv
 *
 * Returns PT_OK on success.
 */
int  pt_mps_offline_run(PT_MPS_Offline *p);

/*
 * Read timing stats. Valid after pt_mps_offline_run() returns.
 */
void pt_mps_offline_get_stats(const PT_MPS_Offline *p, PT_MPS_OfflineStats *out);

#ifdef __cplusplus
}
#endif

#endif /* PT_OFFLINE_MPS_H */
