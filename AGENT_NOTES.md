# Agent Notes — cpp/calibration branch

## What was built

All files are under `src/calibration/`:

| File | Purpose |
|---|---|
| `cm_calibration.h` | Public C API header (extern "C", plain C structs, ctypes-callable from Python) |
| `charuco.cpp` | Port of `calibration/charuco.py` — board creation, detection, image generation |
| `intrinsic.cpp` | Port of `calibration/intrinsic.py` — intrinsic calibration via `cv::calibrateCamera` |
| `calibration_unity.cpp` | Unity build entry point — `#include`s charuco.cpp and intrinsic.cpp |
| `test_calibration.cpp` | Smoke tests: board image generation, blank-frame detection, rendered-board detection, error handling |
| `build_win32.bat` | Windows MSVC build (produces `build/calibration/cm_calibration.dll` + `.lib` + test exe) |
| `build_macos.sh` | macOS clang++ build (produces `build/calibration/libcm_calibration.dylib` + test binary) |

### API surface (cm_calibration.h)

```c
// ChArUco detection
CM_PointPacket *cm_detect_charuco(const uint8_t *bgr, int width, int height, const CM_CharucoConfig *cfg);
void            cm_free_point_packet(CM_PointPacket *p);

// Board image generation
uint8_t        *cm_generate_board_image(const CM_CharucoConfig *cfg, int width, int height, int *out_stride);
void            cm_free_image(uint8_t *img);

// Intrinsic calibration
int             cm_calibrate_intrinsics(CM_PointPacket **packets, int n_packets, int width, int height, const char *serial, CM_CameraIntrinsics *out);
```

### Design decisions

- `CM_PointPacket` uses a single `malloc` for the four data arrays (img_loc, obj_loc, confidence, point_id). `cm_free_point_packet` frees `p->img_loc` (the base of the block) then frees the struct.
- `cm_detect_charuco` returns an empty packet (count=0) rather than NULL when no corners are found, matching Python behavior where an empty PointPacket is returned rather than None.
- Dictionary name → `cv::aruco::PredefinedDictionaryType` mapping covers all 21 entries from Python's `ARUCO_DICTIONARIES` dict.
- Collinear view filtering in `cm_calibrate_intrinsics` uses `std::set<float>` on the X and Y columns of obj_loc, matching Python's `np.unique` check.
- Sub-pixel refinement uses `cornerSubPix` with window (11,11) and criteria (EPS+MAX_ITER, 30, 0.0001) — same as Python.
- RMSE is rounded to 4 decimal places with `round(rms * 10000.0) / 10000.0` — matches Python's `round(error, 4)`.

## How to build (Windows)

From Git Bash (or a cmd prompt):

```bash
cd C:/Git/calimerge/.claude/worktrees/cpp-calibration/src/calibration
./build_win32.bat release
```

Or equivalently:
```bash
cmd //c "call C:/Git/calimerge/.claude/worktrees/cpp-calibration/src/calibration/build_win32.bat release"
```

OpenCV is expected at `C:\OpenCV\opencv\build`. Override with the `OPENCV_PATH` environment variable if installed elsewhere.

Output lands in `build/calibration/` relative to the repo root of the worktree:
- `cm_calibration.dll`
- `cm_calibration.lib`
- `test_calibration.exe`

To run the smoke tests (ensure the OpenCV DLL directory is on PATH):

```cmd
set PATH=C:\OpenCV\opencv\build\x64\vc16\bin;%PATH%
build\calibration\test_calibration.exe
```

## What the Wave 2 (extrinsic / Ceres) agent needs to know

### API additions required in cm_calibration.h

The extrinsic pipeline (see `src/calimerge/calibration/extrinsic.py`) requires Ceres Solver for bundle adjustment. The Wave 2 agent should add these types and function to `cm_calibration.h`:

```c
typedef struct {
    double rotation[9];     /* row-major 3x3 rotation matrix */
    double translation[3];  /* 3-element translation vector */
} CM_CameraExtrinsics;

typedef struct {
    char                serial_number[64];
    int                 port;
    CM_CameraIntrinsics intrinsics;
    CM_CameraExtrinsics extrinsics;
} CM_CalibratedCamera;

/*
 * synced_packets: row-major [n_sync_frames][n_cameras] array of CM_PointPacket*
 *   Access: synced_packets[frame_i * n_cameras + cam_i]
 *   NULL entry means that camera had no detection at that sync frame.
 * ports: int[n_cameras] — port number for each camera
 * intrinsics: CM_CameraIntrinsics[n_cameras] — one per camera
 * out: CM_CalibratedCamera[n_cameras] — caller-allocated, filled on success
 *
 * Returns CM_CAL_OK on success.
 */
int cm_calibrate_extrinsics(
    CM_PointPacket     **synced_packets,
    int                  n_sync_frames,
    int                  n_cameras,
    CM_CameraIntrinsics *intrinsics,
    int                 *ports,
    CM_CalibratedCamera *out
);
```

### Python pipeline stages to port (in order)

1. `stereo_calibrate_pair` — `cv::stereoCalibrate` with `CALIB_FIX_INTRINSIC`
2. `compute_initial_extrinsics` — chain stereo pairs, gap-filling via bridging
3. `build_point_estimates` — triangulate initial 3D points via `cv::triangulatePoints`
4. `run_bundle_adjustment` — Ceres least-squares over camera Rodrigues vectors + 3D points
5. `filter_point_estimates` — remove worst-2.5% observations by reprojection error

### Ceres notes

- Ceres is the only heavy dependency Wave 2 adds; OpenCV is already available.
- The bundle adjustment uses 6-DOF camera params per camera: [rx, ry, rz] (Rodrigues) + [tx, ty, tz].
- The anchor camera (port 0 / reference) should be fixed (not optimized) to remove gauge freedom — mirrors `fix_first_camera=True` in Python.
- The Jacobian sparsity pattern is critical for performance; see `_get_sparsity_pattern` in extrinsic.py for the structure.

### Build script additions

Add Ceres include/lib paths to `build_win32.bat` and `build_macos.sh`. Typical locations:
- Windows: `C:\ceres-solver\build\install` (or vcpkg: `C:\vcpkg\installed\x64-windows`)
- macOS: `/opt/homebrew/opt/ceres-solver` or `/usr/local/opt/ceres-solver`

Add a new unity entry `extrinsic.cpp` to `calibration_unity.cpp`.
