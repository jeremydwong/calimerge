/*
 * pt_stream_mps.m - Streaming pipeline for macOS Apple Silicon.
 *
 * Port of pt_stream.cpp (CUDA pipeline) using CoreML for inference and
 * vImage (Accelerate) for preprocessing.  No GPU memory management needed
 * on Apple Silicon due to unified memory.
 *
 * Style: Plain C structs + free functions.
 */

#import <Foundation/Foundation.h>
#include "pt_stream_mps.h"
#include "pt_coreml.h"
#include "pt_preprocess.h"
#include "pt_heatmap.h"
#include "pt_calibration.h"
#include "../pt_shared/pt_matching.h"
#include "../pt_shared/pt_triangulation.h"
#include "../pt_shared/pt_tracker.h"
#include "../pt_shared/pt_export.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>

/* ============================================================================
 * High-resolution timing (macOS)
 * ============================================================================ */

static double mps_time_seconds(void) {
    static mach_timebase_info_data_t tb = {0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    uint64_t t = mach_absolute_time();
    return (double)(t * tb.numer / tb.denom) * 1e-9;
}

/* ============================================================================
 * Internal stream struct
 * ============================================================================ */

struct PT_MPS_Stream {
    PT_MPS_StreamConfig config;
    PT_CameraConstants  constants;
    PT_TrackState       tracks;
    PT_MPS_StreamStats  stats;

    PT_CoreMLModel yolo_model;
    PT_CoreMLModel vitpose_model;

    /* Pre-allocated CPU buffers (unified memory — same as GPU on Apple Silicon) */
    float *yolo_input;         /* (num_cameras, 3, 640, 640) */
    float *yolo_output;        /* (num_cameras, 300, 6) */
    float *detection_boxes;    /* (num_cameras, MAX_DET, 4) */
    float *detection_scores;   /* (num_cameras, MAX_DET) */
    int   *detection_counts;   /* (num_cameras,) */
    float *vitpose_input;      /* (max_crops, 3, 256, 192) */
    float *vitpose_heatmaps;   /* (max_crops, 52, 64, 48) */
    float *vitpose_affines;    /* (max_crops, 6) */
    float *keypoints_2d;       /* (max_crops, 52, 3) */

    int max_crops;

    /* Per-frame scratch */
    PT_Detection2D detections[PT_MAX_CAMERAS][PT_MAX_DETECTIONS];
    PT_Group groups[PT_MAX_GROUPS];
    PT_CandidateGroup candidate_groups[PT_MAX_GROUPS];

    PT_LetterboxInfo lb_info;

    int is_initialized;
};

/* ============================================================================
 * Logging
 * ============================================================================ */

static void mps_log(const PT_MPS_Stream *s, const char *fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (s->config.log_callback) {
        s->config.log_callback(buf, s->config.callback_user_data);
    }
    fprintf(stderr, "[pt_mps_stream] %s\n", buf);
}

/* ============================================================================
 * Create
 * ============================================================================ */

int pt_mps_stream_create(PT_MPS_Stream **out, const PT_MPS_StreamConfig *config) {
    if (!out || !config) return PT_ERR_INVALID_ARGS;

    PT_MPS_Stream *s = (PT_MPS_Stream *)calloc(1, sizeof(PT_MPS_Stream));
    if (!s) return PT_ERR_INVALID_ARGS;

    memcpy(&s->config, config, sizeof(PT_MPS_StreamConfig));

    /* Apply defaults */
    if (s->config.num_cameras <= 0 || s->config.num_cameras > PT_MAX_CAMERAS) {
        fprintf(stderr, "[pt_mps_stream] Invalid num_cameras: %d\n", s->config.num_cameras);
        free(s);
        return PT_ERR_INVALID_ARGS;
    }
    if (s->config.max_persons <= 0) s->config.max_persons = 2;
    if (s->config.person_confidence <= 0.0f) s->config.person_confidence = 0.1f;
    if (s->config.keypoint_confidence <= 0.0f) s->config.keypoint_confidence = 0.1f;
    if (s->config.epipolar_threshold <= 0.0f) s->config.epipolar_threshold = 10.0f;
    if (s->config.max_track_distance <= 0.0f) s->config.max_track_distance = 0.15f;
    if (s->config.track_patience <= 0) s->config.track_patience = 30;

    int rc;
    int num_cameras = s->config.num_cameras;

    /* --- Load calibration --- */
    mps_log(s, "Loading calibration from %s ...", s->config.calibration_toml_path);
    rc = pt_load_calibration(&s->constants, s->config.calibration_toml_path);
    if (rc != PT_OK) {
        fprintf(stderr, "[pt_mps_stream] Calibration load failed (error %d)\n", rc);
        free(s);
        return rc;
    }

    if (s->constants.num_cameras != num_cameras) {
        mps_log(s, "Warning: calibration has %d cameras, config says %d. Using calibration.",
                s->constants.num_cameras, num_cameras);
        num_cameras = s->constants.num_cameras;
        s->config.num_cameras = num_cameras;
    }

    s->constants.frame_width = s->config.frame_width;
    s->constants.frame_height = s->config.frame_height;
    mps_log(s, "Calibration loaded: %d cameras", num_cameras);

    /* --- Load CoreML models --- */
    mps_log(s, "Loading YOLO model: %s", s->config.yolo_model_path);
    rc = pt_coreml_load(&s->yolo_model, s->config.yolo_model_path);
    if (rc != PT_OK) {
        fprintf(stderr, "[pt_mps_stream] YOLO model load failed (error %d)\n", rc);
        free(s);
        return rc;
    }
    mps_log(s, "YOLO model ready.");

    mps_log(s, "Loading VitPose model: %s", s->config.vitpose_model_path);
    rc = pt_coreml_load(&s->vitpose_model, s->config.vitpose_model_path);
    if (rc != PT_OK) {
        fprintf(stderr, "[pt_mps_stream] VitPose model load failed (error %d)\n", rc);
        pt_coreml_unload(&s->yolo_model);
        free(s);
        return rc;
    }
    mps_log(s, "VitPose model ready.");

    /* --- Allocate CPU buffers --- */
    s->max_crops = num_cameras * PT_MAX_DETECTIONS;

    s->yolo_input = (float *)calloc(num_cameras * 3 * PT_YOLO_INPUT_H * PT_YOLO_INPUT_W, sizeof(float));
    s->yolo_output = (float *)calloc(num_cameras * PT_YOLO_MAX_RAW_DETS * 6, sizeof(float));
    s->detection_boxes = (float *)calloc(num_cameras * PT_MAX_DETECTIONS * 4, sizeof(float));
    s->detection_scores = (float *)calloc(num_cameras * PT_MAX_DETECTIONS, sizeof(float));
    s->detection_counts = (int *)calloc(num_cameras, sizeof(int));
    s->vitpose_input = (float *)calloc(s->max_crops * 3 * PT_VITPOSE_INPUT_H * PT_VITPOSE_INPUT_W, sizeof(float));
    s->vitpose_heatmaps = (float *)calloc(s->max_crops * PT_NUM_KEYPOINTS * PT_VITPOSE_HEATMAP_H * PT_VITPOSE_HEATMAP_W, sizeof(float));
    s->vitpose_affines = (float *)calloc(s->max_crops * 6, sizeof(float));
    s->keypoints_2d = (float *)calloc(s->max_crops * PT_NUM_KEYPOINTS * 3, sizeof(float));

    if (!s->yolo_input || !s->yolo_output || !s->detection_boxes ||
        !s->detection_scores || !s->detection_counts || !s->vitpose_input ||
        !s->vitpose_heatmaps || !s->vitpose_affines || !s->keypoints_2d) {
        fprintf(stderr, "[pt_mps_stream] Buffer allocation failed\n");
        pt_mps_stream_destroy(s);
        return PT_ERR_INVALID_ARGS;
    }

    /* --- Initialize tracking --- */
    pt_track_init(&s->tracks);
    memset(&s->stats, 0, sizeof(PT_MPS_StreamStats));

    s->is_initialized = 1;
    *out = s;

    mps_log(s, "Streaming pipeline ready. %d cameras, %dx%d, max_persons=%d",
            num_cameras, s->config.frame_width, s->config.frame_height,
            s->config.max_persons);

    return PT_OK;
}

/* ============================================================================
 * Destroy
 * ============================================================================ */

void pt_mps_stream_destroy(PT_MPS_Stream *s) {
    if (!s) return;

    pt_coreml_unload(&s->yolo_model);
    pt_coreml_unload(&s->vitpose_model);

    free(s->yolo_input);
    free(s->yolo_output);
    free(s->detection_boxes);
    free(s->detection_scores);
    free(s->detection_counts);
    free(s->vitpose_input);
    free(s->vitpose_heatmaps);
    free(s->vitpose_affines);
    free(s->keypoints_2d);

    free(s);
}

/* ============================================================================
 * Process one frame
 * ============================================================================ */

int pt_mps_stream_process_frame(PT_MPS_Stream *s,
                                 const PT_MPS_StreamFrameSet *input,
                                 PT_MPS_StreamResult *out_result) {
    if (!s || !input || !out_result) return PT_ERR_INVALID_ARGS;
    if (!s->is_initialized) return PT_ERR_INVALID_ARGS;

    double t_frame_start = mps_time_seconds();
    memset(out_result, 0, sizeof(PT_MPS_StreamResult));

    int num_cameras = s->config.num_cameras;
    int frame_w = s->config.frame_width;
    int frame_h = s->config.frame_height;

    /* ================================================================
     * Step 1: Collect valid frames and map ports to camera indices
     * ================================================================ */

    const uint8_t *bgr_ptrs[PT_MAX_CAMERAS];
    int valid_image_count = 0;
    int image_cam_map[PT_MAX_CAMERAS];

    for (int i = 0; i < input->num_frames && i < num_cameras; i++) {
        const PT_MPS_StreamFrame *frame = &input->frames[i];
        if (!frame->pixels) continue;

        int cam_idx = -1;
        for (int c = 0; c < s->constants.num_cameras; c++) {
            if (s->constants.ports[c] == frame->port) {
                cam_idx = c;
                break;
            }
        }
        if (cam_idx < 0) continue;

        bgr_ptrs[valid_image_count] = frame->pixels;
        image_cam_map[valid_image_count] = cam_idx;
        valid_image_count++;
    }

    if (valid_image_count == 0) {
        out_result->sync_index = input->sync_index;
        out_result->processing_time_ms = (mps_time_seconds() - t_frame_start) * 1000.0;
        return PT_OK;
    }

    /* ================================================================
     * Step 2: YOLO — Letterbox + Infer + Filter
     * ================================================================ */

    double t_preprocess = mps_time_seconds();

    pt_preprocess_letterbox_batch(bgr_ptrs, valid_image_count,
                                  frame_w, frame_h,
                                  s->yolo_input,
                                  PT_YOLO_INPUT_W, PT_YOLO_INPUT_H,
                                  &s->lb_info);

    s->stats.preprocess_ms += (mps_time_seconds() - t_preprocess) * 1000.0;

    double t_yolo = mps_time_seconds();

    int yolo_rc = pt_coreml_infer(&s->yolo_model, s->yolo_input,
                                   s->yolo_output, valid_image_count);
    if (yolo_rc != PT_OK) {
        fprintf(stderr, "[pt_mps_stream] YOLO inference failed (error %d)\n", yolo_rc);
        return yolo_rc;
    }

    pt_preprocess_filter_detections(s->yolo_output,
                                    s->detection_boxes,
                                    s->detection_scores,
                                    s->detection_counts,
                                    valid_image_count,
                                    s->config.person_confidence,
                                    &s->lb_info,
                                    frame_w, frame_h);

    s->stats.coreml_yolo_ms += (mps_time_seconds() - t_yolo) * 1000.0;

    /* ================================================================
     * Step 3: VitPose — Crop + Normalize + Infer + Heatmap Decode
     * ================================================================ */

    double t_vitpose = mps_time_seconds();

    int vp_max_batch = s->vitpose_model.input_batch;
    int total_crops = 0;
    int crop_offsets[PT_MAX_CAMERAS];
    for (int img = 0; img < valid_image_count; img++) {
        crop_offsets[img] = total_crops;
        int det_count = s->detection_counts[img];
        if (det_count > PT_MAX_DETECTIONS) det_count = PT_MAX_DETECTIONS;
        int remaining = vp_max_batch - total_crops;
        if (det_count > remaining) det_count = remaining;
        s->detection_counts[img] = det_count;
        total_crops += det_count;
    }

    if (total_crops > 0 && total_crops <= s->max_crops) {
        for (int img = 0; img < valid_image_count; img++) {
            int det_count = s->detection_counts[img];
            if (det_count <= 0) continue;

            float *img_boxes = s->detection_boxes + img * PT_MAX_DETECTIONS * 4;
            int crop_start = crop_offsets[img];
            int crop_size = 3 * PT_VITPOSE_INPUT_H * PT_VITPOSE_INPUT_W;

            pt_preprocess_crop_vitpose_batch(
                bgr_ptrs[img],
                frame_w, frame_h,
                img_boxes, det_count,
                s->vitpose_input + crop_start * crop_size,
                s->vitpose_affines + crop_start * 6
            );
        }

        double t_vp_infer = mps_time_seconds();

        int vp_batch = vp_max_batch;

        int vp_rc = pt_coreml_infer(&s->vitpose_model, s->vitpose_input,
                                     s->vitpose_heatmaps, vp_batch);
        if (vp_rc != PT_OK) {
            fprintf(stderr, "[pt_mps_stream] VitPose inference failed (error %d)\n", vp_rc);
            /* Continue with zero keypoints rather than failing */
        } else {
            /* Heatmap decode: heatmaps -> 2D keypoints */
            pt_heatmap_decode(s->vitpose_heatmaps,
                              s->vitpose_affines,
                              s->keypoints_2d,
                              total_crops);
        }

        double vp_infer_delta_ms = (mps_time_seconds() - t_vp_infer) * 1000.0;
        s->stats.coreml_vitpose_ms += vp_infer_delta_ms;

        /* VitPose section time minus inference = crop+normalize time */
        double vp_section_ms = (mps_time_seconds() - t_vitpose) * 1000.0;
        s->stats.preprocess_ms += vp_section_ms - vp_infer_delta_ms;
    } else {
        s->stats.preprocess_ms += (mps_time_seconds() - t_vitpose) * 1000.0;
    }

    /* ================================================================
     * Step 4: CPU — Build detections, match, triangulate, track
     * ================================================================ */

    double t_matching = mps_time_seconds();

    int det_counts_per_cam[PT_MAX_CAMERAS];
    memset(s->detections, 0, sizeof(s->detections));
    memset(det_counts_per_cam, 0, sizeof(det_counts_per_cam));

    for (int img = 0; img < valid_image_count; img++) {
        int cam = image_cam_map[img];
        int det_count = s->detection_counts[img];
        if (det_count <= 0) continue;
        if (det_count > PT_MAX_DETECTIONS) det_count = PT_MAX_DETECTIONS;

        det_counts_per_cam[cam] = det_count;

        for (int d = 0; d < det_count; d++) {
            PT_Detection2D *det = &s->detections[cam][d];
            det->valid = 1;

            int box_offset = img * PT_MAX_DETECTIONS * 4 + d * 4;
            det->bbox[0] = s->detection_boxes[box_offset + 0];
            det->bbox[1] = s->detection_boxes[box_offset + 1];
            det->bbox[2] = s->detection_boxes[box_offset + 2];
            det->bbox[3] = s->detection_boxes[box_offset + 3];

            int score_offset = img * PT_MAX_DETECTIONS + d;
            det->person_confidence = s->detection_scores[score_offset];

            int crop_idx = crop_offsets[img] + d;
            float *kp_base = s->keypoints_2d + crop_idx * PT_NUM_KEYPOINTS * 3;
            for (int k = 0; k < PT_NUM_KEYPOINTS; k++) {
                det->keypoints[k][0] = kp_base[k * 3 + 0];
                det->keypoints[k][1] = kp_base[k * 3 + 1];
                det->keypoints[k][2] = kp_base[k * 3 + 2];
            }

            /* COM from hip keypoints */
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

    int num_groups = pt_match_cross_view(
        s->detections, det_counts_per_cam,
        &s->constants,
        s->config.epipolar_threshold,
        s->groups, PT_MAX_GROUPS
    );

    s->stats.matching_ms += (mps_time_seconds() - t_matching) * 1000.0;

    double t_tri = mps_time_seconds();

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

    s->stats.triangulation_ms += (mps_time_seconds() - t_tri) * 1000.0;

    /* ================================================================
     * Step 5: Copy active tracks to output
     * ================================================================ */

    double t_track = mps_time_seconds();

    out_result->sync_index = input->sync_index;
    out_result->num_persons = 0;

    for (int t = 0; t < s->tracks.num_tracks; t++) {
        const PT_PersonTrack *track = &s->tracks.tracks[t];
        if (!track->is_active) continue;
        if (out_result->num_persons >= PT_MAX_TRACKS) break;

        PT_MPS_StreamPerson *person = &out_result->persons[out_result->num_persons];
        person->person_id = track->person_id;
        person->num_views = track->num_last_views;

        if (track->history_count > 0) {
            int last_idx;
            if (track->history_count <= PT_TRACK_HISTORY_SIZE) {
                last_idx = track->history_count - 1;
            } else {
                last_idx = (track->history_write_idx + PT_TRACK_HISTORY_SIZE - 1)
                           % PT_TRACK_HISTORY_SIZE;
            }

            for (int k = 0; k < PT_NUM_KEYPOINTS; k++) {
                person->keypoints_3d[k][0] = track->keypoints_3d[last_idx][k][0];
                person->keypoints_3d[k][1] = track->keypoints_3d[last_idx][k][1];
                person->keypoints_3d[k][2] = track->keypoints_3d[last_idx][k][2];
                person->keypoints_valid[k] = track->keypoints_valid[last_idx][k];
            }
        }

        person->com_3d[0] = track->last_com_3d[0];
        person->com_3d[1] = track->last_com_3d[1];
        person->com_3d[2] = track->last_com_3d[2];
        person->com_valid = (track->last_com_3d[0] != 0.0 ||
                             track->last_com_3d[1] != 0.0 ||
                             track->last_com_3d[2] != 0.0) ? 1 : 0;

        out_result->num_persons++;
    }

    s->stats.tracking_ms += (mps_time_seconds() - t_track) * 1000.0;

    double frame_time = (mps_time_seconds() - t_frame_start) * 1000.0;
    out_result->processing_time_ms = frame_time;
    s->stats.total_ms += frame_time;
    s->stats.frames_processed++;

    return PT_OK;
}

/* ============================================================================
 * Stats / Reset / Export
 * ============================================================================ */

void pt_mps_stream_get_stats(const PT_MPS_Stream *s, PT_MPS_StreamStats *out) {
    if (!s || !out) return;
    memcpy(out, &s->stats, sizeof(PT_MPS_StreamStats));
}

void pt_mps_stream_reset_tracks(PT_MPS_Stream *s) {
    if (!s) return;
    pt_track_init(&s->tracks);
}

int pt_mps_stream_export_csv(const PT_MPS_Stream *s, const char *output_base_path) {
    if (!s || !output_base_path) return PT_ERR_INVALID_ARGS;
    return pt_export_csv(&s->tracks, output_base_path);
}
