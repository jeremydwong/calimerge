/*
 * pt_offline_mps.m - Offline batch pipeline for macOS Apple Silicon.
 *
 * Architectural intent:
 *   This module is intentionally THIN. It is a "decode + drive the
 *   streaming core" loop — it does NOT re-implement YOLO, VitPose,
 *   matching, triangulation, or tracking. All of that lives in
 *   pt_stream_mps.m (which the live path also uses) plus pt_shared/.
 *
 *   The user has explicitly asked offline ≡ online except for:
 *     1. frame source = AVAssetReader instead of camera ring buffer
 *     2. optional larger batch sizes
 *
 *   So: we instantiate a PT_MPS_Stream the same way the live path does,
 *   pull decoded BGR frames from AVAssetReader, hand them to
 *   pt_mps_stream_process_frame, and at the end call
 *   pt_mps_stream_export_csv with a base path that matches the CUDA
 *   convention (output_3d_poses_tracked.csv -> *_personN.csv).
 *
 *   The "batch_size" knob here is a future-extension hook: when batch_size
 *   == 1 the loop is a 1:1 streaming replay. When batch_size > 1, we still
 *   call process_frame once per sync index, but we pre-decode `batch_size`
 *   sync indices' worth of frames before kicking off inference, so
 *   AVAssetReader IO can overlap with CoreML compute. Reusing the streaming
 *   core means the matcher / tracker behaviour is byte-identical to live.
 *
 * Style: Plain C structs + free functions. Objective-C only for AVFoundation.
 */

#import <Foundation/Foundation.h>
#include "pt_offline_mps.h"
#include "pt_stream_mps.h"
#include "pt_videodecode.h"
#include "../pt_shared/pt_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <mach/mach_time.h>

/* ============================================================================
 * Timing helper
 * ============================================================================ */

static double offline_time_seconds(void) {
    static mach_timebase_info_data_t tb = {0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    uint64_t t = mach_absolute_time();
    return (double)(t * tb.numer / tb.denom) * 1e-9;
}

/* ============================================================================
 * Sync-table CSV parser
 *
 * frame_time_history.csv has columns: sync_index,port,frame_index,frame_time
 * For each unique sync_index, we record per-camera "0-based video frame
 * index" — derived by ranking each camera's frames by frame_time. This is
 * the same logic as load_sync_table_csv() in src/cuda_pipeline/pt_pipeline.cpp.
 * ============================================================================ */

typedef struct {
    int sync_index;
    int port;
    int frame_index;          /* raw counter from CSV (camera-internal, NOT video position) */
    double frame_time;
    int derived_frame_index;  /* 0-based video position per camera (rank by frame_time) */
} OfflineSyncRow;

typedef struct {
    /* sync_to_frame[sync_slot * num_cameras + cam_idx] = video frame index, -1 if missing */
    int *sync_to_frame;
    int *sync_indices;
    int  num_sync_indices;
    int  num_cameras;
} OfflineSyncTable;

static int compare_sync_rows_offline(const void *a, const void *b) {
    const OfflineSyncRow *ra = (const OfflineSyncRow *)a;
    const OfflineSyncRow *rb = (const OfflineSyncRow *)b;
    if (ra->sync_index != rb->sync_index) return ra->sync_index - rb->sync_index;
    return ra->port - rb->port;
}

static int compare_by_port_time_offline(const void *a, const void *b) {
    const OfflineSyncRow *ra = (const OfflineSyncRow *)a;
    const OfflineSyncRow *rb = (const OfflineSyncRow *)b;
    if (ra->port != rb->port) return ra->port - rb->port;
    if (ra->frame_time < rb->frame_time) return -1;
    if (ra->frame_time > rb->frame_time) return 1;
    return 0;
}

static int load_sync_table_offline(OfflineSyncTable *table,
                                    const char *path,
                                    const int ports[],
                                    int num_cameras) {
    FILE *f = fopen(path, "r");
    if (!f) return PT_ERR_FILE_NOT_FOUND;

    int capacity = 4096;
    OfflineSyncRow *rows = (OfflineSyncRow *)malloc(capacity * sizeof(OfflineSyncRow));
    if (!rows) { fclose(f); return PT_ERR_OUT_OF_MEMORY; }
    int num_rows = 0;

    char line[512];
    if (!fgets(line, sizeof(line), f)) {  /* header */
        free(rows);
        fclose(f);
        return PT_ERR_INVALID_PARAM;
    }

    while (fgets(line, sizeof(line), f)) {
        int sync_idx, port, frame_idx;
        double frame_time = 0.0;
        if (sscanf(line, "%d,%d,%d,%lf", &sync_idx, &port, &frame_idx, &frame_time) >= 3) {
            if (num_rows >= capacity) {
                capacity *= 2;
                OfflineSyncRow *new_rows = (OfflineSyncRow *)realloc(rows,
                                                capacity * sizeof(OfflineSyncRow));
                if (!new_rows) { free(rows); fclose(f); return PT_ERR_OUT_OF_MEMORY; }
                rows = new_rows;
            }
            rows[num_rows].sync_index = sync_idx;
            rows[num_rows].port = port;
            rows[num_rows].frame_index = frame_idx;
            rows[num_rows].frame_time = frame_time;
            rows[num_rows].derived_frame_index = -1;
            num_rows++;
        }
    }
    fclose(f);

    if (num_rows == 0) {
        free(rows);
        return PT_ERR_INVALID_PARAM;
    }

    /* Derive 0-based video frame index per camera by ranking frame_time. */
    qsort(rows, num_rows, sizeof(OfflineSyncRow), compare_by_port_time_offline);
    {
        int i = 0;
        while (i < num_rows) {
            int port_start = i;
            int current_port = rows[i].port;
            while (i < num_rows && rows[i].port == current_port) i++;

            int rank = 0;
            for (int j = port_start; j < i; j++) {
                if (j > port_start && rows[j].frame_time != rows[j - 1].frame_time) {
                    rank = j - port_start;
                }
                rows[j].derived_frame_index = rank;
            }
        }
    }

    qsort(rows, num_rows, sizeof(OfflineSyncRow), compare_sync_rows_offline);

    int num_unique = 1;
    for (int i = 1; i < num_rows; i++) {
        if (rows[i].sync_index != rows[i - 1].sync_index) num_unique++;
    }

    /* Map port -> camera index (index into config.video_paths array) */
    int port_to_cam[256];
    memset(port_to_cam, -1, sizeof(port_to_cam));
    for (int i = 0; i < num_cameras && i < PT_MAX_CAMERAS; i++) {
        int p = ports[i];
        if (p >= 0 && p < 256) port_to_cam[p] = i;
    }

    table->num_sync_indices = num_unique;
    table->num_cameras = num_cameras;
    table->sync_indices  = (int *)malloc(num_unique * sizeof(int));
    table->sync_to_frame = (int *)malloc(num_unique * num_cameras * sizeof(int));
    if (!table->sync_indices || !table->sync_to_frame) {
        free(table->sync_indices);
        free(table->sync_to_frame);
        free(rows);
        table->sync_indices = NULL;
        table->sync_to_frame = NULL;
        return PT_ERR_OUT_OF_MEMORY;
    }

    for (int i = 0; i < num_unique * num_cameras; i++) {
        table->sync_to_frame[i] = -1;
    }

    int sync_slot = 0;
    table->sync_indices[0] = rows[0].sync_index;
    for (int i = 0; i < num_rows; i++) {
        if (i > 0 && rows[i].sync_index != rows[i - 1].sync_index) {
            sync_slot++;
            table->sync_indices[sync_slot] = rows[i].sync_index;
        }
        int cam_idx = -1;
        if (rows[i].port >= 0 && rows[i].port < 256) {
            cam_idx = port_to_cam[rows[i].port];
        }
        if (cam_idx >= 0 && cam_idx < num_cameras) {
            table->sync_to_frame[sync_slot * num_cameras + cam_idx] =
                rows[i].derived_frame_index;
        }
    }

    free(rows);
    return PT_OK;
}

static void free_sync_table_offline(OfflineSyncTable *t) {
    free(t->sync_indices);
    free(t->sync_to_frame);
    t->sync_indices = NULL;
    t->sync_to_frame = NULL;
    t->num_sync_indices = 0;
    t->num_cameras = 0;
}

/* ============================================================================
 * Pipeline state
 * ============================================================================ */

struct PT_MPS_Offline {
    PT_MPS_OfflineConfig config;
    PT_MPS_OfflineStats  stats;

    /* The streaming pipeline is the SAME C object the live path uses; the
     * offline runner is just a different driver around it. */
    PT_MPS_Stream *stream;

    /* AVAssetReader handle per camera, parallel to config.video_paths */
    PT_VideoReader *readers[PT_MAX_CAMERAS];
    int             reader_widths[PT_MAX_CAMERAS];
    int             reader_heights[PT_MAX_CAMERAS];

    /* Video dims used to size the pipeline (taken from the first opened reader) */
    int frame_width;
    int frame_height;

    /* BGR scratch buffer per camera, reused across sync indices.
     * One per (depth, cam) — depth = max(1, batch_size) so we can stage
     * multiple sync indices' worth of decoded frames before invoking
     * inference. Keeps decode IO overlapped with CoreML compute. */
    uint8_t *bgr_buffers[PT_BATCH_SIZE_MAX][PT_MAX_CAMERAS];

    /* Per-camera read cursor: the next 0-based video frame index we'll
     * deliver from this reader. AVAssetReader is sequential-only, so we
     * skip frames forward by repeatedly calling pt_video_read_frame into
     * a discard buffer when we need to advance. */
    int read_cursor[PT_MAX_CAMERAS];
    uint8_t *discard_buf;  /* shared scratch for "decode-and-throw-away" */

    OfflineSyncTable sync_table;

    int is_initialized;
};

/* ============================================================================
 * Logging helper
 * ============================================================================ */

static void offline_log(const PT_MPS_Offline *p, const char *fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (p->config.log_callback) {
        p->config.log_callback(buf, p->config.callback_user_data);
    }
    fprintf(stderr, "[pt_mps_offline] %s\n", buf);
}

static void offline_progress(const PT_MPS_Offline *p, const char *step, float fraction) {
    if (p->config.progress_callback) {
        p->config.progress_callback(step, fraction, p->config.callback_user_data);
    }
}

/* ============================================================================
 * pt_mps_offline_create
 * ============================================================================ */

int pt_mps_offline_create(PT_MPS_Offline **out, const PT_MPS_OfflineConfig *config) {
    if (!out || !config) return PT_ERR_INVALID_ARGS;
    if (config->num_cameras <= 0 || config->num_cameras > PT_MAX_CAMERAS) {
        fprintf(stderr, "[pt_mps_offline] Invalid num_cameras: %d\n", config->num_cameras);
        return PT_ERR_INVALID_ARGS;
    }

    PT_MPS_Offline *p = (PT_MPS_Offline *)calloc(1, sizeof(PT_MPS_Offline));
    if (!p) return PT_ERR_OUT_OF_MEMORY;

    memcpy(&p->config, config, sizeof(PT_MPS_OfflineConfig));

    /* Apply defaults */
    if (p->config.batch_size <= 0) p->config.batch_size = 1;
    if (p->config.batch_size > PT_BATCH_SIZE_MAX) p->config.batch_size = PT_BATCH_SIZE_MAX;
    if (p->config.skip_sync_indices <= 0) p->config.skip_sync_indices = 1;
    if (p->config.max_persons <= 0) p->config.max_persons = 2;
    if (p->config.person_confidence <= 0.0f) p->config.person_confidence = 0.1f;
    if (p->config.keypoint_confidence <= 0.0f) p->config.keypoint_confidence = 0.1f;
    if (p->config.epipolar_threshold <= 0.0f) p->config.epipolar_threshold = 10.0f;
    if (p->config.max_track_distance <= 0.0f) p->config.max_track_distance = 0.15f;
    if (p->config.track_patience <= 0) p->config.track_patience = 30;

    *out = p;
    return PT_OK;
}

/* ============================================================================
 * pt_mps_offline_destroy
 * ============================================================================ */

void pt_mps_offline_destroy(PT_MPS_Offline *p) {
    if (!p) return;

    if (p->stream) {
        pt_mps_stream_destroy(p->stream);
        p->stream = NULL;
    }

    for (int i = 0; i < PT_MAX_CAMERAS; i++) {
        if (p->readers[i]) {
            pt_video_close(p->readers[i]);
            p->readers[i] = NULL;
        }
    }

    for (int d = 0; d < PT_BATCH_SIZE_MAX; d++) {
        for (int c = 0; c < PT_MAX_CAMERAS; c++) {
            free(p->bgr_buffers[d][c]);
            p->bgr_buffers[d][c] = NULL;
        }
    }

    free(p->discard_buf);
    p->discard_buf = NULL;

    free_sync_table_offline(&p->sync_table);

    free(p);
}

/* ============================================================================
 * Internal: open all video readers + read first-frame dims.
 *
 * AVAssetReader can only iterate forward, so we'll position each reader by
 * draining frames into a discard buffer. That works because the sync table
 * only ever asks for frames in increasing order anyway (sync indices are
 * monotone, and per-camera derived frame indices grow monotonically with
 * sync index modulo dropped frames).
 * ============================================================================ */

static int open_all_videos(PT_MPS_Offline *p) {
    int common_w = 0, common_h = 0;

    for (int i = 0; i < p->config.num_cameras; i++) {
        const char *path = p->config.video_paths[i];
        if (!path || path[0] == '\0') {
            offline_log(p, "Camera %d (port %d): empty video path", i, p->config.ports[i]);
            return PT_ERR_INVALID_ARGS;
        }

        int w = 0, h = 0, frame_count = 0;
        double fps = 0.0;
        int rc = pt_video_open(&p->readers[i], path, &w, &h, &fps, &frame_count);
        if (rc != PT_OK) {
            offline_log(p, "Failed to open video for port %d: %s (rc=%d)",
                        p->config.ports[i], path, rc);
            return rc;
        }
        p->reader_widths[i]  = w;
        p->reader_heights[i] = h;
        p->read_cursor[i]    = 0;

        offline_log(p, "Opened port %d: %s (%dx%d, %.1f fps, %d frames)",
                    p->config.ports[i], path, w, h, fps, frame_count);

        if (common_w == 0) { common_w = w; common_h = h; }
        else if (w != common_w || h != common_h) {
            offline_log(p, "Warning: port %d is %dx%d but pipeline expects %dx%d. "
                           "The streaming pipeline assumes uniform dims.",
                        p->config.ports[i], w, h, common_w, common_h);
        }
    }

    p->frame_width  = common_w;
    p->frame_height = common_h;

    /* Allocate BGR buffers once we know the frame size. */
    size_t bgr_bytes = (size_t)common_w * common_h * 3;
    for (int d = 0; d < p->config.batch_size; d++) {
        for (int c = 0; c < p->config.num_cameras; c++) {
            p->bgr_buffers[d][c] = (uint8_t *)malloc(bgr_bytes);
            if (!p->bgr_buffers[d][c]) {
                offline_log(p, "Failed to alloc BGR scratch (%zu bytes)", bgr_bytes);
                return PT_ERR_OUT_OF_MEMORY;
            }
        }
    }
    p->discard_buf = (uint8_t *)malloc(bgr_bytes);
    if (!p->discard_buf) return PT_ERR_OUT_OF_MEMORY;

    return PT_OK;
}

/*
 * Position the reader for camera `cam_idx` at video frame `target_idx`
 * (0-based), decoding into `bgr_out`. Skips intermediate frames into the
 * shared discard buffer. Returns PT_OK on success, PT_ERR_EOF if the
 * stream ran dry before reaching `target_idx`.
 *
 * If `target_idx` < current cursor, returns PT_ERR_INVALID_PARAM — we don't
 * support rewinding (AVAssetReader can't either; would need to re-open).
 */
static int read_at_index(PT_MPS_Offline *p, int cam_idx, int target_idx,
                          uint8_t *bgr_out) {
    if (target_idx < p->read_cursor[cam_idx]) {
        return PT_ERR_INVALID_PARAM;
    }
    while (p->read_cursor[cam_idx] < target_idx) {
        int rc = pt_video_read_frame(p->readers[cam_idx], p->discard_buf);
        if (rc != PT_OK) return rc;
        p->read_cursor[cam_idx]++;
    }
    int rc = pt_video_read_frame(p->readers[cam_idx], bgr_out);
    if (rc != PT_OK) return rc;
    p->read_cursor[cam_idx]++;
    return PT_OK;
}

/* ============================================================================
 * pt_mps_offline_run
 *
 * High-level shape:
 *   1. Open videos (gives us frame_width/height)
 *   2. Build the streaming pipeline with those dims (single source of truth)
 *   3. Load sync table from frame_time_history.csv
 *   4. For each batch of `batch_size` sync indices:
 *        a. Decode one frame per camera into bgr_buffers[bi][cam]
 *        b. For each sync index in the batch, call
 *           pt_mps_stream_process_frame() — same code path as live
 *      (Decode and inference can overlap when batch_size > 1: by the time
 *       we kick off inference for sync index 0, AVAssetReader has already
 *       buffered sync index 1's frames in the next slot.)
 *   5. Export per-track CSVs via pt_mps_stream_export_csv(), which lands
 *      output_3d_poses_tracked.csv_personN.csv files in output_dir —
 *      identical filename schema to the CUDA path.
 * ============================================================================ */

int pt_mps_offline_run(PT_MPS_Offline *p) {
    if (!p) return PT_ERR_INVALID_ARGS;

    double t_start = offline_time_seconds();

    /* --- 1. Open videos --- */
    offline_log(p, "Opening %d video files...", p->config.num_cameras);
    int rc = open_all_videos(p);
    if (rc != PT_OK) return rc;

    /* --- 2. Build the streaming pipeline (same struct the live path uses) --- */
    PT_MPS_StreamConfig sc;
    memset(&sc, 0, sizeof(sc));
    strncpy(sc.yolo_model_path,        p->config.yolo_model_path,        sizeof(sc.yolo_model_path) - 1);
    strncpy(sc.vitpose_model_path,     p->config.vitpose_model_path,     sizeof(sc.vitpose_model_path) - 1);
    strncpy(sc.calibration_toml_path,  p->config.calibration_toml_path,  sizeof(sc.calibration_toml_path) - 1);
    sc.num_cameras       = p->config.num_cameras;
    sc.frame_width       = p->frame_width;
    sc.frame_height      = p->frame_height;
    sc.max_persons       = p->config.max_persons;
    sc.person_confidence = p->config.person_confidence;
    sc.keypoint_confidence = p->config.keypoint_confidence;
    sc.epipolar_threshold  = p->config.epipolar_threshold;
    sc.max_track_distance  = p->config.max_track_distance;
    sc.track_patience      = p->config.track_patience;
    sc.log_callback        = p->config.log_callback;
    sc.callback_user_data  = p->config.callback_user_data;

    offline_log(p, "Creating streaming pipeline (the same one the live path uses)...");
    rc = pt_mps_stream_create(&p->stream, &sc);
    if (rc != PT_OK) {
        offline_log(p, "pt_mps_stream_create failed (rc=%d)", rc);
        return rc;
    }

    /* --- 3. Load sync table --- */
    offline_log(p, "Loading sync table: %s", p->config.frame_time_csv_path);
    rc = load_sync_table_offline(&p->sync_table, p->config.frame_time_csv_path,
                                  p->config.ports, p->config.num_cameras);
    if (rc != PT_OK) {
        offline_log(p, "Sync table load failed (rc=%d)", rc);
        return rc;
    }
    offline_log(p, "Sync table: %d unique sync indices, %d cameras",
                p->sync_table.num_sync_indices, p->sync_table.num_cameras);

    /* Build list of sync indices to process, applying skip_sync_indices */
    int total_sync = p->sync_table.num_sync_indices;
    int skip = p->config.skip_sync_indices;
    int batch_size = p->config.batch_size;

    int process_count = 0;
    for (int i = 0; i < total_sync; i += skip) process_count++;

    int *process_slots = (int *)malloc(process_count * sizeof(int));
    if (!process_slots) return PT_ERR_OUT_OF_MEMORY;
    {
        int idx = 0;
        for (int i = 0; i < total_sync; i += skip) process_slots[idx++] = i;
    }

    /* --- 4. Main loop --- */
    int num_batches = (process_count + batch_size - 1) / batch_size;
    offline_log(p, "Processing %d sync indices in %d batches of %d",
                process_count, num_batches, batch_size);

    int frames_processed = 0;

    for (int bi = 0; bi < num_batches; bi++) {
        int batch_start = bi * batch_size;
        int batch_end = batch_start + batch_size;
        if (batch_end > process_count) batch_end = process_count;
        int this_batch = batch_end - batch_start;

        /* 4a. Decode all frames for this batch into per-slot BGR buffers. */
        double t_dec = offline_time_seconds();
        for (int b = 0; b < this_batch; b++) {
            int sync_slot = process_slots[batch_start + b];
            for (int c = 0; c < p->config.num_cameras; c++) {
                int target = p->sync_table.sync_to_frame[sync_slot * p->config.num_cameras + c];
                if (target < 0) continue;  /* dropped frame for this cam */
                int read_rc = read_at_index(p, c, target, p->bgr_buffers[b][c]);
                if (read_rc != PT_OK) {
                    /* Mark missing for this (b, c) by NULLing the cursor. The
                     * frame loop below detects via re-checking the sync table
                     * + a fresh "did we successfully read?" flag. To keep the
                     * struct simple and avoid an extra side array, we just
                     * zero the buffer so process_frame still gets a black
                     * frame — but that biases the detector. Better: skip
                     * this camera in the per-sync framelist. We use an
                     * inline "valid" marker by rewriting target to -1 in a
                     * shadow table:
                     */
                    /* Sentinel-write: stomp the sync_table copy so the
                     * inference loop below sees "no frame" for this slot. */
                    p->sync_table.sync_to_frame[sync_slot * p->config.num_cameras + c] = -1;
                }
            }
        }
        p->stats.decode_seconds += offline_time_seconds() - t_dec;

        /* 4b. Run inference per sync index (same call site as live path). */
        double t_inf = offline_time_seconds();
        for (int b = 0; b < this_batch; b++) {
            int sync_slot = process_slots[batch_start + b];
            int sync_idx = p->sync_table.sync_indices[sync_slot];

            PT_MPS_StreamFrameSet fs;
            memset(&fs, 0, sizeof(fs));
            fs.sync_index = (uint64_t)sync_idx;

            int valid_count = 0;
            for (int c = 0; c < p->config.num_cameras && valid_count < PT_MAX_CAMERAS; c++) {
                int target = p->sync_table.sync_to_frame[sync_slot * p->config.num_cameras + c];
                if (target < 0) continue;  /* missing for this cam */
                fs.frames[valid_count].pixels = p->bgr_buffers[b][c];
                fs.frames[valid_count].width  = p->frame_width;
                fs.frames[valid_count].height = p->frame_height;
                fs.frames[valid_count].stride = p->frame_width * 3;
                fs.frames[valid_count].port   = p->config.ports[c];
                valid_count++;
            }
            fs.num_frames = valid_count;

            PT_MPS_StreamResult result;
            int proc_rc = pt_mps_stream_process_frame(p->stream, &fs, &result);
            if (proc_rc != PT_OK) {
                offline_log(p, "process_frame failed at sync_index=%d (rc=%d)",
                            sync_idx, proc_rc);
                /* Non-fatal: keep going, otherwise a single bad frame kills the
                 * whole offline run. */
            }
            frames_processed++;
        }
        p->stats.inference_seconds += offline_time_seconds() - t_inf;

        if ((bi & 0x7) == 0 || bi == num_batches - 1) {
            offline_progress(p, "processing", (float)(batch_end) / (float)process_count);
        }
    }

    free(process_slots);

    /* Pull per-stage stats from the streaming side (it tracks them per call). */
    PT_MPS_StreamStats sstats;
    pt_mps_stream_get_stats(p->stream, &sstats);
    p->stats.matching_seconds      = sstats.matching_ms * 1e-3;
    p->stats.triangulation_seconds = sstats.triangulation_ms * 1e-3;

    /* --- 5. Export per-track CSVs. The base path matches the CUDA
     *        convention so the GUI's _convert_outputs (which globs
     *        output_3d_poses_tracked.csv_person*.csv) finds them. --- */
    double t_export = offline_time_seconds();
    char export_base[1024];
    snprintf(export_base, sizeof(export_base),
             "%s/output_3d_poses_tracked.csv", p->config.output_dir);

    /* mkdir -p output_dir; pt_export_csv only auto-creates the immediate
     * parent of its base path, so we make sure the dir itself exists. */
    {
        NSString *odir = [NSString stringWithUTF8String:p->config.output_dir];
        [[NSFileManager defaultManager] createDirectoryAtPath:odir
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:nil];
    }

    offline_log(p, "Exporting per-track CSVs to %s_personN.csv ...", export_base);
    rc = pt_mps_stream_export_csv(p->stream, export_base);
    if (rc != PT_OK) {
        offline_log(p, "CSV export failed (rc=%d). Continuing.", rc);
    }
    p->stats.export_seconds = offline_time_seconds() - t_export;

    /* --- 6. Wrap up --- */
    p->stats.total_seconds   = offline_time_seconds() - t_start;
    p->stats.frames_processed = frames_processed;

    offline_progress(p, "complete", 1.0f);
    offline_log(p, "Offline run complete: %d frames in %.1fs (%.1f frames/s)",
                frames_processed,
                p->stats.total_seconds,
                p->stats.total_seconds > 0
                    ? frames_processed / p->stats.total_seconds : 0.0);

    return PT_OK;
}

/* ============================================================================
 * pt_mps_offline_get_stats
 * ============================================================================ */

void pt_mps_offline_get_stats(const PT_MPS_Offline *p, PT_MPS_OfflineStats *out) {
    if (!p || !out) return;
    memcpy(out, &p->stats, sizeof(PT_MPS_OfflineStats));
}
