/*
 * pt_common.h - Shared constants, structs, error codes for the CUDA pose tracking pipeline.
 *
 * Every module includes this header. It defines the data contracts between:
 *   - pt_arena (memory allocation)
 *   - pt_nvdec (video decode)
 *   - pt_tensorrt (model inference)
 *   - pt_kernels (CUDA preprocessing/postprocessing)
 *   - pt_matching (cross-view epipolar matching)
 *   - pt_triangulation (SVD 3D reconstruction)
 *   - pt_tracker (multi-person tracking)
 *   - pt_pipeline (orchestrator)
 *   - pt_export (CSV output)
 *
 * Style: Plain C structs + free functions. No classes, no methods, no templates.
 *        Fixed-size arrays. No STL. No Boost.
 */

#ifndef PT_COMMON_H
#define PT_COMMON_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

#define PT_MAX_CAMERAS          16
#define PT_MAX_PERSONS          8       /* max tracked persons simultaneously */
#define PT_MAX_TRACKS           32      /* max track slots (active + recently lost) */
#define PT_NUM_KEYPOINTS        52      /* SynthPose/VitPose marker count */
#define PT_MAX_DETECTIONS       16      /* max person detections per image */
#define PT_MAX_GROUPS           32      /* max cross-view groups per sync index */
#define PT_PIPELINE_DEPTH       2       /* double-buffer: decode N+1 while processing N */
#define PT_BATCH_SIZE_MAX       32      /* max sync indices per batch */
#define PT_TRACK_HISTORY_SIZE   128     /* ring buffer size for track history */

/* Model input dimensions */
#define PT_YOLO_INPUT_W         640
#define PT_YOLO_INPUT_H         640
#define PT_YOLO_MAX_RAW_DETS    300     /* YOLO v10 max raw detections */
#define PT_VITPOSE_INPUT_W      192
#define PT_VITPOSE_INPUT_H      256
#define PT_VITPOSE_HEATMAP_W    48
#define PT_VITPOSE_HEATMAP_H    64

/* COCO person class ID */
#define PT_COCO_PERSON_CLASS    0

/* VitPose ImageNet normalization constants */
#define PT_IMAGENET_MEAN_R      0.485f
#define PT_IMAGENET_MEAN_G      0.456f
#define PT_IMAGENET_MEAN_B      0.406f
#define PT_IMAGENET_STD_R       0.229f
#define PT_IMAGENET_STD_G       0.224f
#define PT_IMAGENET_STD_B       0.225f

/* VitPose box expansion factor */
#define PT_VITPOSE_BOX_PAD      1.25f

/* Letterbox padding value (YOLO convention: gray=114) */
#define PT_LETTERBOX_PAD_VALUE  114

/* Hip keypoint indices for center-of-mass (COCO format) */
#define PT_HIP_LEFT_INDEX       11
#define PT_HIP_RIGHT_INDEX      12

/* ============================================================================
 * Error codes
 * ============================================================================ */

#define PT_OK                   0
#define PT_ERR_CUDA             -1
#define PT_ERR_TENSORRT         -2
#define PT_ERR_NVDEC            -3
#define PT_ERR_FILE_NOT_FOUND   -4
#define PT_ERR_INVALID_PARAM    -5
#define PT_ERR_OUT_OF_MEMORY    -6
#define PT_ERR_ENGINE_BUILD     -7
#define PT_ERR_INFERENCE        -8
#define PT_ERR_DECODE           -9
#define PT_ERR_NOT_INITIALIZED  -10

/* ============================================================================
 * GPU Arena - all GPU memory preallocated at startup
 * ============================================================================ */

typedef struct {
    /* --- Decode buffers (double-buffered per camera) --- */
    /* NV12 from NVDEC: Y plane (w*h) + UV plane (w*h/2) */
    uint8_t *decoded_nv12[PT_PIPELINE_DEPTH][PT_MAX_CAMERAS];

    /* BGR8 after colorspace conversion */
    uint8_t *decoded_bgr[PT_PIPELINE_DEPTH][PT_MAX_CAMERAS];

    /* --- YOLO buffers --- */
    /* Letterboxed + normalized input: (batch, 3, 640, 640) fp16 */
    void *yolo_input;   /* __half* on GPU */

    /* Raw YOLO output: (batch, 300, 6) fp32 */
    float *yolo_output;

    /* Filtered detections */
    float *detection_boxes;     /* (batch, MAX_DET, 4) x1,y1,x2,y2 in original coords */
    float *detection_scores;    /* (batch, MAX_DET) */
    int   *detection_counts;    /* (batch,) valid detection count per image */

    /* --- VitPose buffers --- */
    /* Cropped + normalized input: (max_crops, 3, 256, 192) fp32 */
    float *vitpose_input;

    /* Affine transform matrices per crop: (max_crops, 2, 3) for inverse mapping */
    float *vitpose_affine;

    /* Heatmap output: (max_crops, 52, 64, 48) fp32 */
    float *vitpose_heatmaps;

    /* Decoded 2D keypoints: (max_crops, 52, 3) -- x, y, confidence */
    float *keypoints_2d;

    /* --- Pinned host memory for async GPU->CPU transfer --- */
    float *host_keypoints_2d;
    float *host_detection_boxes;
    float *host_detection_scores;
    int   *host_detection_counts;

    /* --- Allocation tracking --- */
    void  *gpu_base;        /* base pointer from single cudaMalloc */
    size_t gpu_total_bytes;
    void  *host_base;       /* base pointer from single cudaMallocHost */
    size_t host_total_bytes;

    /* --- Dimensions for this session --- */
    int num_cameras;
    int frame_width;
    int frame_height;
    int batch_size;         /* sync indices per batch */
    int max_images_per_batch;   /* batch_size * num_cameras */
    int max_crops_per_batch;    /* max_images_per_batch * PT_MAX_DETECTIONS */

} PT_GpuArena;

/* ============================================================================
 * Camera constants - precomputed once at startup
 * ============================================================================ */

typedef struct {
    /* Per-camera intrinsics */
    double camera_matrix[PT_MAX_CAMERAS][3][3];     /* K: [[fx,0,cx],[0,fy,cy],[0,0,1]] */
    double distortion[PT_MAX_CAMERAS][5];           /* [k1, k2, p1, p2, k3] */

    /* Per-camera extrinsics */
    double rotation[PT_MAX_CAMERAS][3][3];          /* R: 3x3 rotation matrix */
    double translation[PT_MAX_CAMERAS][3];          /* t: 3-element translation */

    /* Derived: projection matrix P = K * [R | t] */
    double projection[PT_MAX_CAMERAS][3][4];

    /* Precomputed fundamental matrices for all camera pairs.
     * F[i * PT_MAX_CAMERAS + j] = fundamental matrix from camera i to camera j.
     * Only upper triangle is computed; lower is the transpose.
     * These are CONSTANT for an entire recording session. */
    double fundamental[PT_MAX_CAMERAS * PT_MAX_CAMERAS][3][3];
    int    fundamental_valid[PT_MAX_CAMERAS * PT_MAX_CAMERAS];

    /* Camera port numbers (indices into video file array) */
    int    ports[PT_MAX_CAMERAS];
    int    num_cameras;

    /* Common frame dimensions */
    int    frame_width;
    int    frame_height;

} PT_CameraConstants;

/* ============================================================================
 * Sync table - maps sync_index to per-camera frame indices
 * ============================================================================ */

typedef struct {
    /* sync_to_frame[sync_idx * num_cameras + cam_idx] = video frame index.
     * -1 means no frame for this camera at this sync index. */
    int   *sync_to_frame;
    int   *sync_indices;    /* actual sync_index values (e.g. 73361, 73362, ...) */
    int    num_sync_indices;
    int    num_cameras;

} PT_SyncTable;

/* ============================================================================
 * Detection data - output of YOLO + VitPose, input to matching
 * ============================================================================ */

typedef struct {
    float keypoints[PT_NUM_KEYPOINTS][3];   /* (x, y, confidence) per keypoint */
    float bbox[4];                          /* x1, y1, x2, y2 in original image coords */
    float person_confidence;
    float com_2d[2];                        /* center of mass from hip keypoints */
    int   valid;                            /* 1 if this detection slot is populated */
} PT_Detection2D;

/* ============================================================================
 * Cross-view matching data
 * ============================================================================ */

/* A view-detection pair: identifies one person detection in one camera */
typedef struct {
    int port_index;         /* index into PT_CameraConstants.ports[] */
    int detection_index;    /* index into detections array for this port */
} PT_ViewDetection;

/* A group of matched detections across cameras (same person seen in multiple views) */
typedef struct {
    PT_ViewDetection members[PT_MAX_CAMERAS];
    int num_members;
} PT_Group;

/* ============================================================================
 * Triangulation output
 * ============================================================================ */

typedef struct {
    double xyz[PT_NUM_KEYPOINTS][3];    /* 3D position per keypoint */
    int    valid[PT_NUM_KEYPOINTS];     /* 1 if triangulated, 0 if not */
    double com_3d[3];                   /* center of mass from hips */
    int    com_valid;
    int    views_used[PT_MAX_CAMERAS];  /* which cameras contributed */
    int    num_views;
} PT_Candidate3D;

/* ============================================================================
 * Person track - persistent across frames
 * ============================================================================ */

typedef struct {
    int    person_id;           /* globally unique person ID */
    int    is_active;           /* 1 = active, 0 = deactivated */
    int    frames_since_seen;   /* frames since last successful match */
    int    patience;            /* max frames before deactivation */

    /* Last known state */
    double last_com_3d[3];
    int    last_views[PT_MAX_CAMERAS];
    int    num_last_views;
    int    last_sync_index;

    /* Ring buffer history */
    double keypoints_3d[PT_TRACK_HISTORY_SIZE][PT_NUM_KEYPOINTS][3];
    int    keypoints_valid[PT_TRACK_HISTORY_SIZE][PT_NUM_KEYPOINTS];
    int    sync_indices[PT_TRACK_HISTORY_SIZE];
    int    history_count;
    int    history_write_idx;

} PT_PersonTrack;

typedef struct {
    PT_PersonTrack tracks[PT_MAX_TRACKS];
    int num_tracks;
    int next_person_id;
} PT_TrackState;

/* ============================================================================
 * Pipeline configuration
 * ============================================================================ */

typedef struct {
    /* Input paths */
    char video_paths[PT_MAX_CAMERAS][512];
    int  num_cameras;
    char yolo_onnx_path[512];
    char vitpose_onnx_path[512];
    char engine_cache_dir[512];
    char frame_time_csv_path[512];
    char output_dir[512];

    /* Processing parameters */
    int   batch_size;               /* sync indices per batch (default 8) */
    int   skip_sync_indices;        /* process every Nth sync index (1 = all) */
    int   max_persons;              /* max tracked persons */
    float person_confidence;        /* YOLO detection threshold (default 0.1) */
    float keypoint_confidence;      /* min keypoint confidence (default 0.1) */
    float epipolar_threshold;       /* max epipolar distance in pixels (default 10.0) */
    float max_track_distance;       /* max 3D COM distance for track matching (default 0.15) */
    int   track_patience;           /* frames before losing track (default 30) */

    /* Device */
    int   use_fp16_yolo;            /* 1 = FP16 for YOLO (default 1) */

    /* Callbacks */
    void (*progress_callback)(const char *step, float fraction, void *user_data);
    void (*log_callback)(const char *message, void *user_data);
    void *callback_user_data;

} PT_PipelineConfig;

/* ============================================================================
 * Timing statistics
 * ============================================================================ */

typedef struct {
    double total_seconds;
    double decode_seconds;
    double yolo_seconds;
    double vitpose_seconds;
    double matching_seconds;
    double triangulation_seconds;
    double export_seconds;
    int    frames_processed;
    int    persons_tracked;
} PT_Stats;

/* ============================================================================
 * SynthPose marker names (52 keypoints)
 * Order matches the VitPose model output and the Python SYNTHPOSE_MARKERS dict.
 * ============================================================================ */

static const char *PT_MARKER_NAMES[PT_NUM_KEYPOINTS] = {
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
    "sternum", "rshoulder", "lshoulder",
    "r_lelbow", "l_lelbow", "r_melbow", "l_melbow",
    "r_lwrist", "l_lwrist", "r_mwrist", "l_mwrist",
    "r_ASIS", "l_ASIS", "r_PSIS", "l_PSIS",
    "r_knee", "l_knee", "r_mknee", "l_mknee",
    "r_ankle", "l_ankle", "r_mankle", "l_mankle",
    "r_5meta", "l_5meta", "r_toe", "l_toe",
    "r_big_toe", "l_big_toe", "l_calc", "r_calc",
    "C7", "L2", "T11", "T6"
};

#ifdef __cplusplus
}
#endif

#endif /* PT_COMMON_H */
