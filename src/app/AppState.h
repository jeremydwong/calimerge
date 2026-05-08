#pragma once
// AppState.h - Plain C++ structs for all application state.
//
// Port of src/calimerge/types.py frozen dataclasses and
// src/calimerge/gui/state.py AppState/sub-states.
//
// Design rules (Casey Muratori style):
//   - Plain structs, no methods, no inheritance, no virtual.
//   - Arrays use fixed-size C arrays or std::vector — no hidden allocation
//     in the hot-path structs, std::vector only in the container maps.
//   - Strings that have a known max: char buf[N].
//   - This header is the single source of truth — do not duplicate field
//     definitions in other files.
//   - All structs zero-initialise cleanly (memset-safe for the POD ones).

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

// ============================================================================
// Forward declarations / includes for Qt containers (used in AppState only)
// ============================================================================

#include <QMap>
#include <QString>

// ============================================================================
// Camera Configuration
// Mirrors: CameraConfig in types.py
// ============================================================================

struct CameraConfig {
    char serial_number[64];
    int  port;
    bool enabled;
    int  resolution[2];      // [width, height]
    int  rotation_count;     // 0=0deg, 1=90deg, 2=180deg, 3=270deg
    int  exposure;           // platform-specific units (e.g. -4)
};

inline CameraConfig camera_config_default() {
    CameraConfig c{};
    strncpy(c.serial_number, "", sizeof(c.serial_number) - 1);
    c.port           = 0;
    c.enabled        = true;
    c.resolution[0]  = 1280;
    c.resolution[1]  = 720;
    c.rotation_count = 0;
    c.exposure       = -4;
    return c;
}

// ============================================================================
// Intrinsics
// Mirrors: CameraIntrinsics in types.py
// ============================================================================

struct CameraIntrinsics {
    char  serial_number[64];
    int   resolution[2];         // [width, height]
    float matrix[3][3];          // 3x3 camera matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
    float distortion[5];         // [k1, k2, p1, p2, k3]
    float error;                 // RMSE reprojection error
    int   grid_count;            // number of calibration grids used
    bool  is_scaled;             // true if scaled from a different resolution
    int   scaled_from[2];        // original resolution before scaling; [0,0] if not scaled
};

// ============================================================================
// Extrinsics
// Mirrors: CameraExtrinsics in types.py
// ============================================================================

struct CameraExtrinsics {
    float rotation[3][3];    // 3x3 rotation matrix
    float translation[3];    // translation vector
};

// ============================================================================
// Calibrated camera (intrinsics + extrinsics together)
// Mirrors: CalibratedCamera in types.py
// ============================================================================

struct CalibratedCamera {
    char             serial_number[64];
    int              port;
    CameraIntrinsics intrinsics;
    CameraExtrinsics extrinsics;
};

// ============================================================================
// ChArUco board configuration
// Mirrors: CharucoConfig in types.py
// ============================================================================

struct CharucoConfig {
    int  columns;
    int  rows;
    float square_size_cm;
    char dictionary[32];     // e.g. "DICT_4X4_50"
    bool inverted;
    bool legacy_pattern;
};

inline CharucoConfig charuco_config_default() {
    CharucoConfig c{};
    c.columns        = 7;
    c.rows           = 5;
    c.square_size_cm = 3.0f;
    strncpy(c.dictionary, "DICT_4X4_50", sizeof(c.dictionary) - 1);
    c.inverted       = false;
    c.legacy_pattern = false;
    return c;
}

// ============================================================================
// Recording state
// Mirrors: RecordingState in gui/state.py
// ============================================================================

struct RecordingState {
    bool is_recording;
    char session_dir[512];   // absolute path of the active recording session
    int  frame_count;        // frames written so far
    int  fps;
    float duration_s;        // target duration in seconds
};

inline RecordingState recording_state_default() {
    RecordingState s{};
    s.is_recording = false;
    s.session_dir[0] = '\0';
    s.frame_count  = 0;
    s.fps          = 30;
    s.duration_s   = 10.0f;
    return s;
}

// ============================================================================
// Single-camera runtime state (what we know about a live camera slot)
// Mirrors: CameraState in gui/state.py
// ============================================================================

struct CameraState {
    CameraConfig config;
    bool         is_open;
    char         nickname[64];
    // last_frame is NOT stored here — frames are passed through signals
    // to avoid copying large pixel buffers into AppState.
    int          selected_resolution[2]; // resolution chosen in Record tab UI
};

inline CameraState camera_state_default() {
    CameraState s{};
    s.config              = camera_config_default();
    s.is_open             = false;
    s.nickname[0]         = '\0';
    s.selected_resolution[0] = 1280;
    s.selected_resolution[1] = 720;
    return s;
}

// ============================================================================
// Calibration state
// Mirrors: CalibrationState in gui/state.py
// ============================================================================

// Per-camera intrinsic status tracked for UI display
struct IntrinsicStatus {
    bool  has_intrinsics;
    float error;
    int   grid_count;
    float progress;  // 0.0–1.0 detection progress during calibration run
};

struct CalibrationState {
    // Maps port -> status; populated as calibration runs
    // We use a small fixed-size array (max 8 cameras) to avoid heap alloc
    // in the core struct; QMap lives in AppState where Qt is already used.
    bool  extrinsic_valid;
    float extrinsic_rmse;
    // Calibrated cameras: stored in AppState.calibrated_cameras (QMap)
};

// ============================================================================
// Processing state
// Mirrors: ProcessingState in gui/state.py
// ============================================================================

struct ProcessingState {
    bool  is_processing;
    char  current_step[128];
    float progress;   // 0.0–1.0
};

inline ProcessingState processing_state_default() {
    ProcessingState s{};
    s.is_processing   = false;
    s.current_step[0] = '\0';
    s.progress        = 0.0f;
    return s;
}

// ============================================================================
// AppState — the single, complete application state.
// Mirrors: AppState in gui/state.py
//
// StateManager holds one of these and emits signals when it changes.
// The Qt containers (QMap) live here because AppState is not hot-path POD —
// it is copied infrequently (on state updates).
// ============================================================================

struct AppState {
    // Project
    char project_path[512];
    int  fps;

    // Cameras
    QMap<int, CameraState> cameras;    // port -> CameraState
    bool is_previewing;

    // Recording
    RecordingState recording;

    // Calibration
    CalibrationState calibration;
    QMap<int, CalibratedCamera> calibrated_cameras;  // port -> CalibratedCamera
    QMap<int, IntrinsicStatus>  intrinsic_status;     // port -> status

    // Processing
    ProcessingState processing;

    // ChArUco configs (loaded from project settings)
    CharucoConfig charuco_intrinsic;
    CharucoConfig charuco_extrinsic;

    // UI
    int  current_tab;
    char status_message[256];
};

inline AppState app_state_default() {
    AppState s{};
    s.project_path[0]  = '\0';
    s.fps              = 30;
    s.is_previewing    = false;
    s.recording        = recording_state_default();
    s.calibration      = CalibrationState{};
    s.processing       = processing_state_default();
    s.charuco_intrinsic = charuco_config_default();
    s.charuco_extrinsic = charuco_config_default();
    s.current_tab      = 0;
    s.status_message[0] = '\0';
    return s;
}
