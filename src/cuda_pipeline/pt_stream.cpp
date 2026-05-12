/*
 * pt_stream.cpp - Real-time streaming pipeline for live camera frames.
 *
 * Accepts BGR camera frames from cm_capture_synced() and returns 3D tracked
 * poses synchronously.  Reuses the same GPU modules as the batch pipeline
 * (arena, kernels, TensorRT, matching, triangulation, tracker) but skips
 * video decode and sync table -- frames arrive already decoded from cameras.
 *
 * Key differences from the batch pipeline (pt_pipeline.cpp):
 *   - No video decoder (NVDEC), no NV12 buffers -- BGR frames uploaded directly
 *   - No sync table -- caller provides a monotonic sync_index
 *   - No batching across time -- one sync frame at a time
 *   - No CSV export -- results returned as PT_StreamResult struct
 *   - d_bgr_ptrs pre-allocated at create time (no per-frame cudaMalloc)
 *
 * Estimated per-frame latency: 12-18ms for 3 cameras at 640x480.
 *
 * Style: Plain C structs + free functions. No classes, no templates, no STL.
 */

#include "pt_stream.h"
#include "pt_pipeline.h"    /* for pt_load_calibration */
#include "pt_arena.h"
#include "pt_tensorrt.h"
#include "pt_kernels.h"
#include "pt_matching.h"
#include "pt_triangulation.h"
#include "pt_tracker.h"
#include "pt_export.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* High-resolution timing */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static double stream_time_seconds(void) {
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double stream_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/* ============================================================================
 * Internal stream struct (opaque to callers)
 * ============================================================================ */

struct PT_Stream {
    PT_StreamConfig config;
    PT_GpuArena arena;
    PT_CameraConstants constants;
    PT_TrackState tracks;
    PT_StreamStats stats;

    PT_TrtEngine yolo_engine;
    PT_TrtEngine vitpose_engine;

    cudaStream_t cuda_stream;

    /* Pre-allocated GPU pointer array for letterbox kernel.
     * Eliminates per-frame cudaMalloc/cudaFree from the batch pipeline. */
    uint8_t **d_bgr_ptrs;

    /* Letterbox parameters (computed once from frame dimensions) */
    float lb_scale;
    int lb_pad_x;
    int lb_pad_y;

    /* Scratch buffers for CPU-side processing per frame */
    PT_Detection2D detections[PT_MAX_CAMERAS][PT_MAX_DETECTIONS];
    PT_Group groups[PT_MAX_GROUPS];
    PT_CandidateGroup candidate_groups[PT_MAX_GROUPS];

    int is_initialized;
};

/* ============================================================================
 * Logging helper
 * ============================================================================ */

static void stream_log(const PT_Stream *s, const char *fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (s->config.log_callback) {
        s->config.log_callback(buf, s->config.callback_user_data);
    }
    fprintf(stderr, "[pt_stream] %s\n", buf);
}

/* ============================================================================
 * pt_stream_create
 *
 * 1. Load calibration TOML -> projection + fundamental matrices
 * 2. Create CUDA stream
 * 3. Init arena (batch_size=1, single set of camera buffers)
 * 4. Build YOLO engine (max_batch = num_cameras)
 * 5. Build VitPose engine (max_batch = num_cameras * MAX_DET)
 * 6. Pre-allocate d_bgr_ptrs on GPU
 * 7. Compute letterbox parameters
 * ============================================================================ */

extern "C" int pt_stream_create(PT_Stream **out, const PT_StreamConfig *config) {
    if (!out || !config) return PT_ERR_INVALID_PARAM;

    PT_Stream *s = (PT_Stream *)calloc(1, sizeof(PT_Stream));
    if (!s) return PT_ERR_OUT_OF_MEMORY;

    /* Copy configuration and apply defaults */
    memcpy(&s->config, config, sizeof(PT_StreamConfig));

    if (s->config.num_cameras <= 0 || s->config.num_cameras > PT_MAX_CAMERAS) {
        fprintf(stderr, "[pt_stream] Invalid num_cameras: %d\n", s->config.num_cameras);
        free(s);
        return PT_ERR_INVALID_PARAM;
    }
    if (s->config.frame_width <= 0 || s->config.frame_height <= 0) {
        fprintf(stderr, "[pt_stream] Invalid frame dimensions: %dx%d\n",
                s->config.frame_width, s->config.frame_height);
        free(s);
        return PT_ERR_INVALID_PARAM;
    }
    if (s->config.max_persons <= 0) s->config.max_persons = 2;
    if (s->config.person_confidence <= 0.0f) s->config.person_confidence = 0.1f;
    if (s->config.keypoint_confidence <= 0.0f) s->config.keypoint_confidence = 0.1f;
    if (s->config.epipolar_threshold <= 0.0f) s->config.epipolar_threshold = 10.0f;
    if (s->config.max_track_distance <= 0.0f) s->config.max_track_distance = 0.15f;
    if (s->config.track_patience <= 0) s->config.track_patience = 30;
    if (s->config.use_fp16_yolo < 0) s->config.use_fp16_yolo = 1;

    int rc;
    int num_cameras = s->config.num_cameras;
    int frame_w = s->config.frame_width;
    int frame_h = s->config.frame_height;

    /* --- Step 1: Load calibration --- */

    stream_log(s, "Loading calibration from %s ...", s->config.calibration_toml_path);

    rc = pt_load_calibration(&s->constants, s->config.calibration_toml_path);
    if (rc != PT_OK) {
        fprintf(stderr, "[pt_stream] Failed to load calibration (error %d)\n", rc);
        free(s);
        return rc;
    }

    if (s->constants.num_cameras != num_cameras) {
        stream_log(s, "Warning: calibration has %d cameras but config says %d. Using calibration count.",
                   s->constants.num_cameras, num_cameras);
        num_cameras = s->constants.num_cameras;
        s->config.num_cameras = num_cameras;
    }

    s->constants.frame_width = frame_w;
    s->constants.frame_height = frame_h;

    stream_log(s, "Calibration loaded: %d cameras", num_cameras);

    /* --- Step 2: Create CUDA stream --- */

    cudaError_t cerr = cudaStreamCreate(&s->cuda_stream);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "[pt_stream] cudaStreamCreate failed: %s\n", cudaGetErrorString(cerr));
        free(s);
        return PT_ERR_CUDA;
    }

    /* --- Step 3: Initialize GPU arena (batch_size=1) --- */

    stream_log(s, "Initializing GPU arena (batch_size=1, %d cameras, %dx%d)...",
               num_cameras, frame_w, frame_h);
    rc = pt_arena_init(&s->arena, num_cameras, frame_w, frame_h, 1);
    if (rc != PT_OK) {
        fprintf(stderr, "[pt_stream] Arena init failed (error %d)\n", rc);
        cudaStreamDestroy(s->cuda_stream);
        free(s);
        return rc;
    }
    pt_arena_print_stats(&s->arena);

    /* --- Step 4: Build/load TensorRT engines --- */

    stream_log(s, "Initializing TensorRT runtime...");
    rc = pt_trt_init();
    if (rc != PT_OK) {
        fprintf(stderr, "[pt_stream] TensorRT init failed (error %d)\n", rc);
        pt_arena_destroy(&s->arena);
        cudaStreamDestroy(s->cuda_stream);
        free(s);
        return rc;
    }

    /* YOLO engine: max_batch = num_cameras (one sync frame = one image per camera) */
    int yolo_max_batch = num_cameras;
    stream_log(s, "Building YOLO engine (max_batch=%d, fp16=%d)...",
               yolo_max_batch, s->config.use_fp16_yolo);
    memset(&s->yolo_engine, 0, sizeof(PT_TrtEngine));
    rc = pt_trt_build_engine(&s->yolo_engine, s->config.yolo_onnx_path,
                              s->config.engine_cache_dir, yolo_max_batch,
                              s->config.use_fp16_yolo);
    if (rc != PT_OK) {
        fprintf(stderr, "[pt_stream] YOLO engine build failed (error %d)\n", rc);
        pt_trt_shutdown();
        pt_arena_destroy(&s->arena);
        cudaStreamDestroy(s->cuda_stream);
        free(s);
        return rc;
    }
    stream_log(s, "YOLO engine ready.");

    /* VitPose engine: max_batch = num_cameras * PT_MAX_DETECTIONS */
    int vitpose_max_batch = num_cameras * PT_MAX_DETECTIONS;
    stream_log(s, "Building VitPose engine (max_batch=%d)...", vitpose_max_batch);
    memset(&s->vitpose_engine, 0, sizeof(PT_TrtEngine));
    rc = pt_trt_build_engine(&s->vitpose_engine, s->config.vitpose_onnx_path,
                              s->config.engine_cache_dir, vitpose_max_batch, 1);
    if (rc != PT_OK) {
        fprintf(stderr, "[pt_stream] VitPose engine build failed (error %d)\n", rc);
        pt_trt_destroy_engine(&s->yolo_engine);
        pt_trt_shutdown();
        pt_arena_destroy(&s->arena);
        cudaStreamDestroy(s->cuda_stream);
        free(s);
        return rc;
    }
    stream_log(s, "VitPose engine ready.");

    /* --- Step 5: Pre-allocate d_bgr_ptrs on GPU --- */

    cerr = cudaMalloc(&s->d_bgr_ptrs, num_cameras * sizeof(uint8_t *));
    if (cerr != cudaSuccess) {
        fprintf(stderr, "[pt_stream] cudaMalloc d_bgr_ptrs failed: %s\n", cudaGetErrorString(cerr));
        pt_trt_destroy_engine(&s->vitpose_engine);
        pt_trt_destroy_engine(&s->yolo_engine);
        pt_trt_shutdown();
        pt_arena_destroy(&s->arena);
        cudaStreamDestroy(s->cuda_stream);
        free(s);
        return PT_ERR_CUDA;
    }

    /* --- Step 6: Compute letterbox parameters --- */

    s->lb_scale = fminf((float)PT_YOLO_INPUT_W / frame_w,
                         (float)PT_YOLO_INPUT_H / frame_h);
    int lb_new_w = (int)(frame_w * s->lb_scale);
    int lb_new_h = (int)(frame_h * s->lb_scale);
    s->lb_pad_x = (PT_YOLO_INPUT_W - lb_new_w) / 2;
    s->lb_pad_y = (PT_YOLO_INPUT_H - lb_new_h) / 2;

    stream_log(s, "Letterbox: scale=%.4f, pad=(%d,%d), new_size=(%d,%d)",
               s->lb_scale, s->lb_pad_x, s->lb_pad_y, lb_new_w, lb_new_h);

    /* --- Step 7: Initialize tracking state --- */

    pt_track_init(&s->tracks);
    memset(&s->stats, 0, sizeof(PT_StreamStats));

    s->is_initialized = 1;
    *out = s;

    stream_log(s, "Streaming pipeline ready. %d cameras, %dx%d, max_persons=%d",
               num_cameras, frame_w, frame_h, s->config.max_persons);

    return PT_OK;
}

/* ============================================================================
 * pt_stream_destroy
 * ============================================================================ */

extern "C" void pt_stream_destroy(PT_Stream *s) {
    if (!s) return;

    /* Free pre-allocated GPU pointer array */
    if (s->d_bgr_ptrs) {
        cudaFree(s->d_bgr_ptrs);
        s->d_bgr_ptrs = NULL;
    }

    /* Destroy TensorRT engines */
    pt_trt_destroy_engine(&s->yolo_engine);
    pt_trt_destroy_engine(&s->vitpose_engine);

    /* Destroy CUDA stream */
    if (s->cuda_stream) {
        cudaStreamDestroy(s->cuda_stream);
        s->cuda_stream = NULL;
    }

    /* Free GPU arena */
    pt_arena_destroy(&s->arena);

    /* Shutdown TensorRT runtime */
    pt_trt_shutdown();

    free(s);
}

/* ============================================================================
 * pt_stream_process_frame
 *
 * Core processing for one synchronized frame set:
 *   1. Upload BGR frames -> arena.decoded_bgr[0][cam] via cudaMemcpyAsync
 *   2. Upload pointer array -> d_bgr_ptrs
 *   3. Letterbox -> YOLO infer -> filter -> sync (get counts)
 *   4. VitPose crop -> infer -> heatmap decode -> sync (get keypoints)
 *   5. Build Detection2D arrays -> cross-view match -> candidates -> track
 *   6. Copy active tracks -> PT_StreamResult
 *
 * Extracted from pt_pipeline.cpp lines 994-1246 with:
 *   - Video decode replaced by cudaMemcpyAsync H2D for BGR frames
 *   - Batch loop removed (single sync frame)
 *   - Per-frame cudaMalloc/cudaFree replaced by pre-allocated d_bgr_ptrs
 *   - Result returned as PT_StreamResult instead of CSV export
 * ============================================================================ */

extern "C" int pt_stream_process_frame(PT_Stream *s,
                                        const PT_StreamFrameSet *input,
                                        PT_StreamResult *out_result) {
    if (!s || !input || !out_result) return PT_ERR_INVALID_PARAM;
    if (!s->is_initialized) return PT_ERR_NOT_INITIALIZED;

    double t_frame_start = stream_time_seconds();
    memset(out_result, 0, sizeof(PT_StreamResult));

    int num_cameras = s->config.num_cameras;
    int frame_w = s->config.frame_width;
    int frame_h = s->config.frame_height;

    /* ================================================================
     * Step 1: Upload BGR frames to GPU
     *
     * Each frame goes into arena.decoded_bgr[0][cam_index] which is
     * the same slot the batch pipeline uses after NV12->BGR conversion.
     * ================================================================ */

    double t_upload = stream_time_seconds();

    uint8_t *bgr_ptrs[PT_MAX_CAMERAS];
    int valid_image_count = 0;
    int image_cam_map[PT_MAX_CAMERAS]; /* image index -> camera port index */

    for (int i = 0; i < input->num_frames && i < num_cameras; i++) {
        const PT_StreamFrame *frame = &input->frames[i];
        if (!frame->pixels) continue;

        /* Map frame port to camera index in the constants array */
        int cam_idx = -1;
        for (int c = 0; c < s->constants.num_cameras; c++) {
            if (s->constants.ports[c] == frame->port) {
                cam_idx = c;
                break;
            }
        }
        if (cam_idx < 0) continue;

        /* Upload BGR frame to the arena's decoded_bgr slot */
        int frame_bytes = frame->height * frame->stride;
        cudaMemcpyAsync(s->arena.decoded_bgr[0][cam_idx],
                         frame->pixels, frame_bytes,
                         cudaMemcpyHostToDevice, s->cuda_stream);

        bgr_ptrs[valid_image_count] = s->arena.decoded_bgr[0][cam_idx];
        image_cam_map[valid_image_count] = cam_idx;
        valid_image_count++;
    }

    s->stats.upload_ms += (stream_time_seconds() - t_upload) * 1000.0;

    if (valid_image_count == 0) {
        out_result->sync_index = input->sync_index;
        out_result->processing_time_ms = (stream_time_seconds() - t_frame_start) * 1000.0;
        return PT_OK;
    }

    /* ================================================================
     * Step 2: YOLO — Letterbox + Infer + Filter
     * ================================================================ */

    double t_yolo = stream_time_seconds();

    /* Upload BGR pointer array to pre-allocated GPU buffer */
    cudaMemcpyAsync(s->d_bgr_ptrs, bgr_ptrs,
                     valid_image_count * sizeof(uint8_t *),
                     cudaMemcpyHostToDevice, s->cuda_stream);

    /* Letterbox + normalize -> YOLO input */
    pt_launch_letterbox_batch(
        s->d_bgr_ptrs, s->arena.yolo_input,
        valid_image_count,
        frame_w, frame_h,
        PT_YOLO_INPUT_W, PT_YOLO_INPUT_H,
        s->cuda_stream
    );

    /* YOLO inference */
    pt_trt_infer(&s->yolo_engine, s->arena.yolo_input, s->arena.yolo_output,
                  valid_image_count, s->cuda_stream);

    /* Filter detections: threshold + undo letterbox */
    pt_launch_filter_detections(
        s->arena.yolo_output,
        s->arena.detection_boxes,
        s->arena.detection_scores,
        s->arena.detection_counts,
        valid_image_count,
        s->config.person_confidence,
        s->lb_scale, s->lb_pad_x, s->lb_pad_y,
        frame_w, frame_h,
        s->cuda_stream
    );

    /* Copy detection counts to CPU (small transfer, needed before VitPose) */
    cudaMemcpyAsync(s->arena.host_detection_counts,
                     s->arena.detection_counts,
                     valid_image_count * sizeof(int),
                     cudaMemcpyDeviceToHost, s->cuda_stream);

    cudaStreamSynchronize(s->cuda_stream);

    /* Debug: print YOLO detection counts per camera */
    {
        static int debug_frame = 0;
        if (debug_frame < 5) {
            fprintf(stderr, "[pt_stream] YOLO detections (frame %d):", debug_frame);
            for (int img = 0; img < valid_image_count; img++) {
                fprintf(stderr, " cam%d=%d", img, s->arena.host_detection_counts[img]);
            }
            fprintf(stderr, " (conf_thresh=%.2f)\n", s->config.person_confidence);
            debug_frame++;
        }
    }

    s->stats.yolo_ms += (stream_time_seconds() - t_yolo) * 1000.0;

    /* ================================================================
     * Step 3: VitPose — Crop + Normalize + Infer + Heatmap Decode
     * ================================================================ */

    double t_vitpose = stream_time_seconds();

    /* Compute total crops for this frame */
    int total_crops = 0;
    int crop_offsets[PT_MAX_CAMERAS];
    for (int img = 0; img < valid_image_count; img++) {
        crop_offsets[img] = total_crops;
        int det_count = s->arena.host_detection_counts[img];
        if (det_count > PT_MAX_DETECTIONS) det_count = PT_MAX_DETECTIONS;
        total_crops += det_count;
    }

    if (total_crops > 0) {
        /* Crop + normalize for each image's detections */
        for (int img = 0; img < valid_image_count; img++) {
            int det_count = s->arena.host_detection_counts[img];
            if (det_count <= 0) continue;
            if (det_count > PT_MAX_DETECTIONS) det_count = PT_MAX_DETECTIONS;

            int cam = image_cam_map[img];
            float *img_boxes = s->arena.detection_boxes + img * PT_MAX_DETECTIONS * 4;
            int crop_start = crop_offsets[img];
            float *vp_input = s->arena.vitpose_input +
                              crop_start * 3 * PT_VITPOSE_INPUT_H * PT_VITPOSE_INPUT_W;
            float *vp_affine = s->arena.vitpose_affine + crop_start * 6;

            pt_launch_crop_normalize_vitpose(
                s->arena.decoded_bgr[0][cam],
                frame_w, frame_h,
                img_boxes, det_count,
                vp_input, vp_affine,
                s->cuda_stream
            );
        }

        /* VitPose inference on all crops */
        pt_trt_infer(&s->vitpose_engine, s->arena.vitpose_input,
                      s->arena.vitpose_heatmaps, total_crops, s->cuda_stream);

        /* Heatmap decode: heatmaps -> 2D keypoints */
        pt_launch_heatmap_decode(
            s->arena.vitpose_heatmaps,
            s->arena.vitpose_affine,
            s->arena.keypoints_2d,
            total_crops,
            s->cuda_stream
        );

        /* Copy results to CPU (pinned memory) */
        cudaMemcpyAsync(s->arena.host_keypoints_2d,
                         s->arena.keypoints_2d,
                         total_crops * PT_NUM_KEYPOINTS * 3 * sizeof(float),
                         cudaMemcpyDeviceToHost, s->cuda_stream);

        cudaMemcpyAsync(s->arena.host_detection_boxes,
                         s->arena.detection_boxes,
                         valid_image_count * PT_MAX_DETECTIONS * 4 * sizeof(float),
                         cudaMemcpyDeviceToHost, s->cuda_stream);

        cudaMemcpyAsync(s->arena.host_detection_scores,
                         s->arena.detection_scores,
                         valid_image_count * PT_MAX_DETECTIONS * sizeof(float),
                         cudaMemcpyDeviceToHost, s->cuda_stream);
    }

    /* Synchronize: all GPU work for this frame is complete */
    cudaStreamSynchronize(s->cuda_stream);

    s->stats.vitpose_ms += (stream_time_seconds() - t_vitpose) * 1000.0;

    /* ================================================================
     * Step 4: CPU — Build detections, match, triangulate, track
     * ================================================================ */

    double t_matching = stream_time_seconds();

    /* Build PT_Detection2D arrays from host keypoints */
    int det_counts_per_cam[PT_MAX_CAMERAS];
    memset(s->detections, 0, sizeof(s->detections));
    memset(det_counts_per_cam, 0, sizeof(det_counts_per_cam));

    for (int img = 0; img < valid_image_count; img++) {
        int cam = image_cam_map[img];
        int det_count = s->arena.host_detection_counts[img];
        if (det_count <= 0) continue;
        if (det_count > PT_MAX_DETECTIONS) det_count = PT_MAX_DETECTIONS;

        det_counts_per_cam[cam] = det_count;

        for (int d = 0; d < det_count; d++) {
            PT_Detection2D *det = &s->detections[cam][d];
            det->valid = 1;

            /* Copy bounding box */
            int box_offset = img * PT_MAX_DETECTIONS * 4 + d * 4;
            det->bbox[0] = s->arena.host_detection_boxes[box_offset + 0];
            det->bbox[1] = s->arena.host_detection_boxes[box_offset + 1];
            det->bbox[2] = s->arena.host_detection_boxes[box_offset + 2];
            det->bbox[3] = s->arena.host_detection_boxes[box_offset + 3];

            /* Copy person confidence */
            int score_offset = img * PT_MAX_DETECTIONS + d;
            det->person_confidence = s->arena.host_detection_scores[score_offset];

            /* Copy 2D keypoints */
            int crop_idx = crop_offsets[img] + d;
            float *kp_base = s->arena.host_keypoints_2d +
                             crop_idx * PT_NUM_KEYPOINTS * 3;
            for (int k = 0; k < PT_NUM_KEYPOINTS; k++) {
                det->keypoints[k][0] = kp_base[k * 3 + 0];
                det->keypoints[k][1] = kp_base[k * 3 + 1];
                det->keypoints[k][2] = kp_base[k * 3 + 2];
            }

            /* Compute center-of-mass from hip keypoints */
            float lhip_x = det->keypoints[PT_HIP_LEFT_INDEX][0];
            float lhip_y = det->keypoints[PT_HIP_LEFT_INDEX][1];
            float lhip_c = det->keypoints[PT_HIP_LEFT_INDEX][2];
            float rhip_x = det->keypoints[PT_HIP_RIGHT_INDEX][0];
            float rhip_y = det->keypoints[PT_HIP_RIGHT_INDEX][1];
            float rhip_c = det->keypoints[PT_HIP_RIGHT_INDEX][2];

            if (lhip_c > s->config.keypoint_confidence &&
                rhip_c > s->config.keypoint_confidence) {
                det->com_2d[0] = (lhip_x + rhip_x) * 0.5f;
                det->com_2d[1] = (lhip_y + rhip_y) * 0.5f;
            } else if (lhip_c > s->config.keypoint_confidence) {
                det->com_2d[0] = lhip_x;
                det->com_2d[1] = lhip_y;
            } else if (rhip_c > s->config.keypoint_confidence) {
                det->com_2d[0] = rhip_x;
                det->com_2d[1] = rhip_y;
            } else {
                det->com_2d[0] = (det->bbox[0] + det->bbox[2]) * 0.5f;
                det->com_2d[1] = (det->bbox[1] + det->bbox[3]) * 0.5f;
            }
        }
    }

    s->stats.matching_ms += (stream_time_seconds() - t_matching) * 1000.0;

    /* Cross-view matching */
    double t_match2 = stream_time_seconds();

    int num_groups = pt_match_cross_view(
        s->detections, det_counts_per_cam,
        &s->constants,
        s->config.epipolar_threshold,
        s->groups, PT_MAX_GROUPS
    );

    s->stats.matching_ms += (stream_time_seconds() - t_match2) * 1000.0;

    /* Triangulation + tracking */
    double t_tri = stream_time_seconds();

    int num_candidate_groups = pt_generate_candidates(
        s->groups, num_groups,
        s->detections,
        &s->constants,
        s->config.keypoint_confidence,
        s->candidate_groups, PT_MAX_GROUPS
    );

    pt_track_frame(
        &s->tracks,
        s->candidate_groups, num_candidate_groups,
        (int)input->sync_index,
        s->config.max_track_distance,
        s->config.max_persons,
        s->config.track_patience
    );

    s->stats.triangulation_ms += (stream_time_seconds() - t_tri) * 1000.0;

    /* ================================================================
     * Step 5: Copy active tracks to output result
     * ================================================================ */

    double t_track = stream_time_seconds();

    out_result->sync_index = input->sync_index;
    out_result->num_persons = 0;

    for (int t = 0; t < s->tracks.num_tracks; t++) {
        const PT_PersonTrack *track = &s->tracks.tracks[t];
        if (!track->is_active) continue;
        if (out_result->num_persons >= PT_MAX_TRACKS) break;

        PT_StreamPerson *person = &out_result->persons[out_result->num_persons];
        person->person_id = track->person_id;
        person->num_views = track->num_last_views;

        /* Get the most recent frame from the ring buffer */
        if (track->history_count > 0) {
            int last_idx;
            if (track->history_count <= PT_TRACK_HISTORY_SIZE) {
                last_idx = track->history_count - 1;
            } else {
                /* Buffer wrapped: most recent is one before write_idx */
                last_idx = (track->history_write_idx + PT_TRACK_HISTORY_SIZE - 1) % PT_TRACK_HISTORY_SIZE;
            }

            for (int k = 0; k < PT_NUM_KEYPOINTS; k++) {
                person->keypoints_3d[k][0] = track->keypoints_3d[last_idx][k][0];
                person->keypoints_3d[k][1] = track->keypoints_3d[last_idx][k][1];
                person->keypoints_3d[k][2] = track->keypoints_3d[last_idx][k][2];
                person->keypoints_valid[k] = track->keypoints_valid[last_idx][k];
            }
        }

        /* Copy last COM */
        person->com_3d[0] = track->last_com_3d[0];
        person->com_3d[1] = track->last_com_3d[1];
        person->com_3d[2] = track->last_com_3d[2];
        person->com_valid = (track->last_com_3d[0] != 0.0 ||
                             track->last_com_3d[1] != 0.0 ||
                             track->last_com_3d[2] != 0.0) ? 1 : 0;

        out_result->num_persons++;
    }

    s->stats.tracking_ms += (stream_time_seconds() - t_track) * 1000.0;

    /* ================================================================
     * Multi-person debug: emit one fixed-width line per emitted person.
     * Format keeps every column at a constant width so the user can scan
     * a streaming log and see at a glance which track has stale data.
     *   sync=NNNNNNNN  p#=N  id=NNNN  nv=N  com=( X.XX, Y.YY, Z.ZZ)  comV=N  ndets=NN
     * COM is in meters with two decimals; "ndets" reports input detection
     * counts per camera so missed detections are visible at a glance.
     * ================================================================ */
    {
        char det_buf[64];
        int n_written = 0;
        det_buf[0] = '\0';
        for (int c = 0; c < s->constants.num_cameras && n_written < (int)sizeof(det_buf) - 4; c++) {
            n_written += snprintf(det_buf + n_written, sizeof(det_buf) - (size_t)n_written,
                                  "%s%d", c == 0 ? "" : ",", det_counts_per_cam[c]);
        }

        if (out_result->num_persons == 0) {
            stream_log(s,
                "sync=%08llu  persons=0                                                 ndets=[%s]",
                (unsigned long long)out_result->sync_index, det_buf);
        } else {
            for (int p = 0; p < out_result->num_persons; p++) {
                const PT_StreamPerson *person = &out_result->persons[p];
                stream_log(s,
                    "sync=%08llu  p#=%d  id=%04d  nv=%d  com=(%6.2f,%6.2f,%6.2f)  comV=%d  ndets=[%s]",
                    (unsigned long long)out_result->sync_index,
                    p,
                    person->person_id,
                    person->num_views,
                    person->com_3d[0], person->com_3d[1], person->com_3d[2],
                    person->com_valid,
                    det_buf);
            }
        }
    }

    /* Total timing */
    double frame_time = (stream_time_seconds() - t_frame_start) * 1000.0;
    out_result->processing_time_ms = frame_time;
    s->stats.total_ms += frame_time;
    s->stats.frames_processed++;

    return PT_OK;
}

/* ============================================================================
 * pt_stream_get_stats
 * ============================================================================ */

extern "C" void pt_stream_get_stats(const PT_Stream *s, PT_StreamStats *out) {
    if (!s || !out) return;
    memcpy(out, &s->stats, sizeof(PT_StreamStats));
}

/* ============================================================================
 * pt_stream_reset_tracks
 * ============================================================================ */

extern "C" void pt_stream_reset_tracks(PT_Stream *s) {
    if (!s) return;
    pt_track_init(&s->tracks);
}

/* ============================================================================
 * pt_stream_stitch_tracks
 * ============================================================================ */

extern "C" int pt_stream_stitch_tracks(PT_Stream *s, int max_gap_frames, float max_distance_m) {
    if (!s) return 0;
    return pt_track_stitch(&s->tracks, max_gap_frames, max_distance_m);
}

/* ============================================================================
 * pt_stream_export_csv
 * ============================================================================ */

extern "C" int pt_stream_export_csv(const PT_Stream *s, const char *output_base_path) {
    if (!s || !output_base_path) return PT_ERR_INVALID_PARAM;
    return pt_export_csv(&s->tracks, output_base_path);
}
