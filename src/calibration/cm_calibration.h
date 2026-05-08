/*
 * cm_calibration.h
 *
 * Public C API for ChArUco detection and camera intrinsic calibration.
 *
 * Design principles (same as calimerge_platform.h):
 * - Plain C structs, no member functions
 * - No templates, no STL in the API
 * - No exceptions cross the API boundary — functions return int error codes
 * - Fixed-size arrays where possible
 * - OpenCV types are internal to the .cpp files only
 *
 * Callable from Python via ctypes (extern "C" linkage).
 *
 * Wave 2 (Ceres bundle adjustment) will extend this header with:
 *   CM_CameraExtrinsics, CM_CalibratedCamera, cm_calibrate_extrinsics()
 */

#ifndef CM_CALIBRATION_H
#define CM_CALIBRATION_H

#include <stdint.h>

/* Windows DLL export/import decoration.
 * Define CM_CALIBRATION_BUILDING_DLL when compiling the library itself. */
#ifdef _WIN32
#  ifdef CM_CALIBRATION_BUILDING_DLL
#    define CM_API __declspec(dllexport)
#  else
#    define CM_API __declspec(dllimport)
#  endif
#else
#  define CM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Error codes
 * ============================================================================ */

#define CM_CAL_OK                  0
#define CM_CAL_ERR_INVALID_PARAM  -1
#define CM_CAL_ERR_INSUFFICIENT   -2   /* not enough detections / frames */
#define CM_CAL_ERR_OPENCV         -3   /* OpenCV internal error */
#define CM_CAL_ERR_OUT_OF_MEMORY  -4

/* ============================================================================
 * ChArUco Configuration
 * ============================================================================ */

/*
 * Mirror of Python's CharucoConfig.
 * square_size_cm is in centimetres; square_size_m = square_size_cm / 100.0
 * marker_size_m  = square_size_m * 0.75  (75% standard ratio)
 */
typedef struct {
    int   columns;
    int   rows;
    float square_size_cm;
    char  dictionary[32];   /* e.g. "DICT_4X4_50" */
    int   inverted;         /* 0 = normal, 1 = inverted (white on black) */
    int   legacy_pattern;   /* 0 = modern OpenCV layout, 1 = legacy */
} CM_CharucoConfig;

/* ============================================================================
 * Point Packet  (detected corners from a single frame)
 * ============================================================================ */

/*
 * Heap-allocated result from cm_detect_charuco().
 * All pointer members point into a single allocation owned by this struct.
 * Free with cm_free_point_packet().
 *
 * img_loc  : float[count * 2]   — image (x, y) coordinates
 * obj_loc  : float[count * 3]   — object-space (x, y, z) board coordinates
 * confidence: float[count]      — per-corner confidence (always 1.0f for ChArUco)
 * point_id  : int[count]        — ChArUco corner ID for each detected corner
 * count     : number of detected corners (0 if none)
 */
typedef struct {
    float *img_loc;       /* (count, 2) x,y image coordinates   */
    float *obj_loc;       /* (count, 3) x,y,z board coordinates */
    float *confidence;    /* (count,)   per-corner confidence    */
    int   *point_id;      /* (count,)   ChArUco corner IDs       */
    int    count;
} CM_PointPacket;

/* ============================================================================
 * Intrinsic Calibration Result
 * ============================================================================ */

/*
 * Mirror of Python's CameraIntrinsics.
 * matrix[9]      : row-major 3x3 camera matrix  [[fx,0,cx],[0,fy,cy],[0,0,1]]
 * distortion[5]  : OpenCV distortion coefficients [k1, k2, p1, p2, k3]
 * error          : RMSE reprojection error in pixels
 * grid_count     : number of frames used in calibration
 */
typedef struct {
    char   serial_number[64];
    int    width;
    int    height;
    double matrix[9];       /* row-major 3x3 */
    double distortion[5];   /* [k1, k2, p1, p2, k3] */
    double error;           /* RMSE reprojection error */
    int    grid_count;      /* frames actually used */
} CM_CameraIntrinsics;

/* ============================================================================
 * ChArUco Detection
 * ============================================================================ */

/*
 * Detect ChArUco corners in a single BGR frame.
 *
 * bgr    : pointer to BGR pixel data, row-major, stride = width * 3
 * width  : frame width in pixels
 * height : frame height in pixels
 * cfg    : board configuration
 *
 * Returns a heap-allocated CM_PointPacket (count may be 0 if nothing detected).
 * Returns NULL only on out-of-memory.
 * Caller must free with cm_free_point_packet().
 *
 * Mirrors Python detect_charuco_points():
 *   - Converts to grayscale
 *   - Applies inversion if cfg->inverted
 *   - If no corners found, retries on horizontally mirrored frame and
 *     flips x coordinates back to original frame space
 *   - Runs sub-pixel refinement (cornerSubPix)
 *   - Fills obj_loc from board.getChessboardCorners()
 */
CM_API CM_PointPacket *cm_detect_charuco(
    const uint8_t        *bgr,
    int                   width,
    int                   height,
    const CM_CharucoConfig *cfg
);

/*
 * Free a CM_PointPacket returned by cm_detect_charuco().
 * Safe to call with NULL.
 */
CM_API void cm_free_point_packet(CM_PointPacket *p);

/* ============================================================================
 * Board Image Generation
 * ============================================================================ */

/*
 * Generate a BGR image of the ChArUco board.
 *
 * cfg        : board configuration
 * width      : desired image width in pixels
 * height     : desired image height in pixels
 * out_stride : filled with bytes per row (= width * 3 for packed BGR)
 *
 * Returns a heap-allocated buffer of size (height * out_stride) bytes.
 * Returns NULL on failure.
 * Caller must free with cm_free_image().
 *
 * Mirrors Python generate_board_image(): generates grayscale then converts
 * to BGR, applies bitwise inversion if cfg->inverted.
 */
CM_API uint8_t *cm_generate_board_image(
    const CM_CharucoConfig *cfg,
    int                     width,
    int                     height,
    int                    *out_stride
);

/*
 * Free a buffer returned by cm_generate_board_image().
 * Safe to call with NULL.
 */
CM_API void cm_free_image(uint8_t *img);

/* ============================================================================
 * Intrinsic Calibration
 * ============================================================================ */

/*
 * Calibrate camera intrinsics from collected ChArUco detections.
 *
 * packets   : array of CM_PointPacket* pointers (one per frame)
 * n_packets : number of packets in the array
 * width     : frame width in pixels
 * height    : frame height in pixels
 * serial    : camera serial number string (copied into out->serial_number)
 * out       : caller-allocated CM_CameraIntrinsics to fill
 *
 * Returns CM_CAL_OK on success.
 * Returns CM_CAL_ERR_INSUFFICIENT if fewer than 3 valid frames after filtering.
 * Returns CM_CAL_ERR_OPENCV if cv::calibrateCamera fails.
 *
 * Mirrors Python calibrate_intrinsics():
 *   - Skips packets with fewer than 4 corners
 *   - Skips collinear views (all corners on a single row or column)
 *   - Runs cv::calibrateCamera with no fixed flags
 *   - Rounds error to 4 decimal places
 */
CM_API int cm_calibrate_intrinsics(
    CM_PointPacket **packets,
    int              n_packets,
    int              width,
    int              height,
    const char      *serial,
    CM_CameraIntrinsics *out
);

/* ============================================================================
 * Wave 2 placeholder — do NOT implement here yet.
 *
 * The extrinsic calibration / bundle adjustment agent (Wave 2, uses Ceres)
 * will add the following to this header:
 *
 *   typedef struct CM_CameraExtrinsics { double rotation[9]; double translation[3]; } CM_CameraExtrinsics;
 *   typedef struct CM_CalibratedCamera  { char serial[64]; int port; CM_CameraIntrinsics intrinsics; CM_CameraExtrinsics extrinsics; } CM_CalibratedCamera;
 *
 *   int cm_calibrate_extrinsics(
 *       CM_PointPacket **synced_packets,  // [n_sync_frames * n_cameras] row-major
 *       int              n_sync_frames,
 *       int              n_cameras,
 *       CM_CameraIntrinsics *intrinsics,  // [n_cameras]
 *       int             *ports,           // [n_cameras]
 *       CM_CalibratedCamera *out          // [n_cameras], caller-allocated
 *   );
 *
 * See extrinsic.py for the full Python pipeline to port.
 * ============================================================================ */

#ifdef __cplusplus
}
#endif

#endif /* CM_CALIBRATION_H */
