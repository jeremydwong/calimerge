/*
 * calimerge_platform.h
 *
 * Platform-independent API for multi-camera capture.
 *
 * Design principles (Handmade Hero style):
 * - Plain C structs, no member functions
 * - No templates
 * - No STL in hot paths
 * - Platform layer implements these functions
 */

#ifndef CALIMERGE_PLATFORM_H
#define CALIMERGE_PLATFORM_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

#define CM_MAX_CAMERAS 16
#define CM_SERIAL_LEN 64
#define CM_NAME_LEN 128

/* Common resolutions to test */
#define CM_RES_640x480   0
#define CM_RES_1280x720  1
#define CM_RES_1920x1080 2
#define CM_RES_COUNT     3

/* ============================================================================
 * Core Data Structures
 * ============================================================================ */

typedef struct {
    int width;
    int height;
} CM_Resolution;

typedef struct {
    char serial_number[CM_SERIAL_LEN];   /* Unique device identifier */
    char display_name[CM_NAME_LEN];      /* Human-readable name */
    int  device_index;                   /* Platform device index */

    /* Current settings */
    int  width;
    int  height;
    int  fps;
    int  rotation;                       /* 0, 90, 180, 270 degrees */
    int  exposure;                       /* Platform-specific units */
    bool enabled;

    /* Supported resolutions (filled by cm_enumerate_cameras) */
    CM_Resolution supported_resolutions[CM_RES_COUNT];
    int supported_resolution_count;

    /* Opaque platform handle - do not touch from Python */
    void *platform_handle;
} CM_Camera;

typedef struct {
    uint8_t *pixels;                     /* BGR format (OpenCV compatible) */
    int      width;
    int      height;
    int      stride;                     /* Bytes per row */
    uint64_t timestamp_ns;               /* Camera's native PTS (nanoseconds) */
    uint64_t arrival_ns;                 /* Common clock arrival time (for sync verification) */
    uint64_t corrected_ns;               /* PTS + clock_offset = common clock domain */
    int      camera_index;
} CM_Frame;

typedef struct {
    CM_Frame frames[CM_MAX_CAMERAS];
    int      frame_count;
    int      dropped_mask;               /* Bit i = 1 if camera i dropped */
    uint64_t sync_index;
} CM_SyncedFrameSet;

/* ============================================================================
 * Error Codes
 * ============================================================================ */

#define CM_OK                    0
#define CM_ERROR_INIT_FAILED    -1
#define CM_ERROR_NO_CAMERA      -2
#define CM_ERROR_OPEN_FAILED    -3
#define CM_ERROR_CAPTURE_FAILED -4
#define CM_ERROR_INVALID_PARAM  -5
#define CM_ERROR_NOT_SUPPORTED  -6

/* ============================================================================
 * Lifecycle Functions
 * ============================================================================ */

/*
 * Initialize the camera subsystem.
 * Must be called before any other cm_* function.
 * Returns CM_OK on success.
 */
int cm_init(void);

/*
 * Shutdown the camera subsystem.
 * Releases all resources. Safe to call multiple times.
 */
void cm_shutdown(void);

/* ============================================================================
 * Camera Enumeration
 * ============================================================================ */

/*
 * Enumerate available cameras.
 * Fills out_cameras array with camera info (serial, name, supported resolutions).
 * Returns number of cameras found, or negative error code.
 */
int cm_enumerate_cameras(CM_Camera *out_cameras, int max_cameras);

/*
 * Get serial number for a device by index.
 * Returns CM_OK on success, copies serial to out_serial.
 */
int cm_get_camera_serial(int device_index, char *out_serial, int max_len);

/* ============================================================================
 * Camera Control
 * ============================================================================ */

/*
 * Open a camera for capture.
 * Camera must have been returned by cm_enumerate_cameras.
 * Returns CM_OK on success.
 */
int cm_open_camera(CM_Camera *camera);

/*
 * Close a camera.
 * Safe to call on already-closed camera.
 */
void cm_close_camera(CM_Camera *camera);

/*
 * Set camera resolution.
 * Takes effect on next frame capture.
 * Returns CM_OK on success.
 */
int cm_set_resolution(CM_Camera *camera, int width, int height);

/*
 * Set target frame rate.
 * Actual FPS may vary based on camera capabilities.
 * Returns CM_OK on success.
 */
int cm_set_fps(CM_Camera *camera, int fps);

/*
 * Set exposure value.
 * Units are platform-specific (typically EV or direct register values).
 * Returns CM_OK on success.
 */
int cm_set_exposure(CM_Camera *camera, int exposure);

/* ============================================================================
 * Frame Capture
 * ============================================================================ */

/*
 * Capture a single frame from a camera.
 * Blocks until a frame is available (or timeout).
 * Caller must call cm_release_frame when done with the frame.
 * Returns CM_OK on success.
 */
int cm_capture_frame(CM_Camera *camera, CM_Frame *out_frame);

/*
 * Release a captured frame.
 * Frees any buffers associated with the frame.
 */
void cm_release_frame(CM_Frame *frame);

/*
 * Get timestamp of most recent frame (without capturing).
 * Useful for synchronization decisions.
 * Returns timestamp in nanoseconds, or 0 if no frame available.
 */
uint64_t cm_get_latest_timestamp(CM_Camera *camera);

/* ============================================================================
 * Multi-Camera Synchronization (optional - can implement in Python first)
 * ============================================================================ */

/*
 * Capture synchronized frames from multiple cameras.
 * Attempts to get frames as close in time as possible.
 * Sets dropped_mask bits for cameras that failed to provide a frame.
 * Returns CM_OK on success (even if some cameras dropped).
 */
int cm_capture_synced(CM_Camera *cameras, int camera_count, CM_SyncedFrameSet *out);

/*
 * Release all frames in a synced frame set.
 */
void cm_release_synced(CM_SyncedFrameSet *frameset);

#ifdef __cplusplus
}
#endif

#endif /* CALIMERGE_PLATFORM_H */
