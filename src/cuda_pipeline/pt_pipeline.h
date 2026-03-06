/*
 * pt_pipeline.h - Public C API for the CUDA pose tracking pipeline orchestrator.
 *
 * The pipeline ties together all Phase 1 modules into a single batch-processing
 * loop:  decode -> colorspace -> YOLO -> VitPose -> matching -> triangulation -> tracking.
 *
 * Usage:
 *   PT_Pipeline *p = NULL;
 *   pt_pipeline_create(&p, &config);
 *   pt_pipeline_load_calibration(p, "calibration.toml");
 *   pt_pipeline_load_sync_table(p, "frame_time_history.csv");
 *   pt_pipeline_run(p);
 *   pt_pipeline_destroy(p);
 *
 * Style: Plain C structs + free functions. No classes, no templates, no STL.
 */

#ifndef PT_PIPELINE_H
#define PT_PIPELINE_H

#include "pt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque pipeline handle -- internal struct defined in pt_pipeline.cpp */
typedef struct PT_Pipeline PT_Pipeline;

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

/*
 * pt_pipeline_create - Allocate and configure a new pipeline instance.
 *
 * Copies config into the pipeline struct.  Does NOT allocate GPU resources
 * or load models -- that happens lazily during pt_pipeline_run().
 *
 * Parameters:
 *   out    - receives pointer to newly allocated pipeline
 *   config - pipeline configuration (copied, caller can free after call)
 *
 * Returns PT_OK on success, PT_ERR_INVALID_PARAM if out or config is NULL.
 */
int pt_pipeline_create(PT_Pipeline **out, const PT_PipelineConfig *config);

/*
 * pt_pipeline_destroy - Free all resources held by the pipeline.
 *
 * Closes video decoders, destroys TensorRT engines, frees GPU arena,
 * frees the sync table, and frees the pipeline struct itself.
 *
 * Safe to call on NULL (no-op).  Sets *p to NULL after destruction
 * if the caller passes the address of their pointer.
 */
void pt_pipeline_destroy(PT_Pipeline *p);

/* ============================================================================
 * Configuration (call before pt_pipeline_run)
 * ============================================================================ */

/*
 * pt_pipeline_load_calibration - Load camera calibration from a TOML file.
 *
 * Reads per-camera intrinsics and extrinsics from a TOML file written by
 * save_calibration_to_toml() in the Python package.  Computes projection
 * matrices and fundamental matrices for all camera pairs.
 *
 * Expected TOML format:
 *   [camera_0]
 *   port = 0
 *   serial_number = "ABC123"
 *   camera_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
 *   distortion = [k1, k2, p1, p2, k3]
 *   rotation = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
 *   translation = [tx, ty, tz]
 *
 * Returns PT_OK on success.
 * Returns PT_ERR_FILE_NOT_FOUND if the file cannot be opened.
 * Returns PT_ERR_INVALID_PARAM if parsing fails.
 */
int pt_pipeline_load_calibration(PT_Pipeline *p, const char *toml_path);

/*
 * pt_pipeline_load_sync_table - Load sync table from frame_time_history.csv.
 *
 * Reads the CSV file and builds the sync_to_frame mapping that maps each
 * unique sync_index to a per-camera frame_index.
 *
 * Expected CSV format (header + data rows):
 *   sync_index,port,frame_index,frame_time
 *   73361,0,0,0.033
 *   73361,1,0,0.034
 *   ...
 *
 * Returns PT_OK on success.
 * Returns PT_ERR_FILE_NOT_FOUND if the file cannot be opened.
 * Returns PT_ERR_INVALID_PARAM if parsing fails or no sync indices found.
 */
int pt_pipeline_load_sync_table(PT_Pipeline *p, const char *csv_path);

/* ============================================================================
 * Execution
 * ============================================================================ */

/*
 * pt_pipeline_run - Run the full processing pipeline.
 *
 * This is the main entry point.  It:
 *   1. Initializes the GPU arena
 *   2. Opens all video decoders
 *   3. Loads/builds TensorRT engines (YOLO + VitPose)
 *   4. Processes all sync indices in batches
 *   5. Exports results to CSV
 *
 * Progress is reported via config.progress_callback at the end of each batch.
 * Major events are logged via config.log_callback.
 *
 * Returns PT_OK on success.
 * Returns error code on failure (GPU, model, decode, etc.).
 */
int pt_pipeline_run(PT_Pipeline *p);

/* ============================================================================
 * Statistics
 * ============================================================================ */

/*
 * pt_pipeline_get_stats - Retrieve timing and processing statistics.
 *
 * Valid after pt_pipeline_run() returns.  All times are in seconds.
 */
void pt_pipeline_get_stats(const PT_Pipeline *p, PT_Stats *out);

/* ============================================================================
 * Standalone calibration loader (shared with pt_stream)
 * ============================================================================ */

/*
 * pt_load_calibration - Load camera calibration from a TOML file into a
 * standalone PT_CameraConstants struct.
 *
 * Parses the TOML, then computes projection matrices and fundamental matrices.
 * This is the same operation as pt_pipeline_load_calibration() but does not
 * require a PT_Pipeline instance.
 *
 * Returns PT_OK on success.
 */
int pt_load_calibration(PT_CameraConstants *constants, const char *toml_path);

#ifdef __cplusplus
}
#endif

#endif /* PT_PIPELINE_H */
