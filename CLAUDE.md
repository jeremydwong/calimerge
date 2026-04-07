# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Platforms

This project is developed on **both Windows and macOS**. Platform-specific instructions are noted below.

## Environment Setup for Claude

**IMPORTANT**: uv is installed at `~/.local/bin/uv`. Add to PATH before running commands:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Or use the full path directly:

```bash
~/.local/bin/uv sync
~/.local/bin/uv run calimerge gui
~/.local/bin/uv run python3 -c "from calimerge import types; print('ok')"
```

### Windows Environment Gotcha (Anaconda)

On Windows, if Anaconda is installed, the `VIRTUAL_ENV` environment variable may be set to the conda base path (e.g. `C:\Users\<user>\anaconda3`). This interferes with `uv run` — it prints a warning and uses the wrong Python interpreter, causing missing module errors (cv2, etc.).

**Fix**: Unset `VIRTUAL_ENV` before running `uv` commands:
```bash
unset VIRTUAL_ENV
~/.local/bin/uv run pytest ...
```

Or prefix each command:
```bash
VIRTUAL_ENV= ~/.local/bin/uv run pytest ...
```

### Platform Differences

| | macOS | Windows |
|---|---|---|
| Python binary | `python3` | `python` |
| Native build | `cd src/native && ./build_macos.sh release` | `cd src/native && cmd //c build_win32.bat release` |
| Camera backend | AVFoundation (`calimerge_macos.mm`) | Media Foundation (`calimerge_win32.cpp`) |
| Exposure API | AVCaptureDevice exposureDuration | IAMVideoProcAmp (log2 seconds) |
| Shell | zsh/bash | Git Bash (use Unix syntax, not Windows) |

## Repository Overview

**calimerge** is a unified multi-camera motion capture application. It merges three legacy packages:

- **caliscope** (legacy, in `caliscope/`): GUI calibration and 3D pose estimation
- **multiwebcam** (legacy, in `multiwebcam/`): Synchronized webcam recording
- **posetrack** (legacy, in `posetrack/`): VitPose-based pose estimation

The active unified package is in `src/calimerge/` and uses **uv** (not Poetry).

## Build and Development Commands

```bash
# Setup (first time)
~/.local/bin/uv sync

# Build native camera library (macOS)
cd src/native && ./build_macos.sh release && cd ../..

# Run the GUI
~/.local/bin/uv run calimerge gui

# Run other tools
~/.local/bin/uv run calimerge clock    # Sync verification clock
~/.local/bin/uv run calimerge record   # Legacy recording GUI

# Run tests
~/.local/bin/uv run pytest

# Lint
~/.local/bin/uv run ruff check src/

# Test imports
~/.local/bin/uv run python3 -c "from calimerge.types import CameraConfig; print('ok')"
```

### Windows Build Tools

MSVC Build Tools are installed at:
```
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\
```

To set up the compiler environment from bash (required before building native code):
```bash
# Initialize MSVC environment (sets up cl.exe, link.exe, etc.)
eval "$('/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/Tools/VsDevCmd.bat' > /dev/null 2>&1 && set)" 2>/dev/null
# Or from a Windows cmd prompt:
# "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
```

Build the native Windows DLL:
```bash
cd src/native && cmd //c build_win32.bat release && cd ../..
```

File search utility: **Everything** (voidtools) is installed with the `es.exe` CLI at:
```
C:\Program Files (x86)\Everything\es.exe
```

### Native C++ Tests (after building)

```bash
cd src/native
./test_enumerate    # List cameras with serial numbers
./test_capture 0    # Single camera capture test (camera index)
./test_multi        # Multi-camera synchronized capture
./test_usb_serials  # Check whether cameras have real USB iSerialNumbers
```

### Building a single .cpp file on Windows

```bash
cd src/native
cmd //c "cl test_usb_serials.cpp /EHsc /link mf.lib mfplat.lib mfuuid.lib ole32.lib setupapi.lib"
```

Note: `cmd //c build_win32.bat` sets up the MSVC environment automatically.
For one-off files, either run from a VS Developer Command Prompt, or use the bat:
```bash
cmd //c "build_win32.bat release && cl test_usb_serials.cpp /EHsc /link mf.lib mfplat.lib mfuuid.lib ole32.lib setupapi.lib"
```

---

## Current Architecture

### Directory Structure

```
src/
├── calimerge/                  # Python package
│   ├── cli.py                  # CLI dispatcher (gui, record, clock)
│   ├── camera_binding.py       # ctypes bindings to C++ lib
│   ├── types.py                # Core data structures (frozen dataclasses)
│   ├── config.py               # TOML config + SQLite intrinsics DB
│   ├── triangulation.py        # 3D reconstruction (Anipose, Numba-jitted)
│   ├── video_recorder.py       # Multi-camera video recording
│   ├── clock_widget.py         # On-screen sync verification clock
│   │
│   ├── calibration/            # Calibration pipeline (pure functions)
│   │   ├── charuco.py          # ChArUco board creation + PDF export
│   │   ├── intrinsic.py        # Per-camera lens calibration
│   │   └── extrinsic.py        # Multi-camera bundle adjustment
│   │
│   └── gui/                    # PySide6 interface
│       ├── main.py             # MainWindow (4 tabs)
│       ├── state.py            # AppState + StateManager
│       ├── workers.py          # QThread workers (enumerate, preview, record, calibrate)
│       ├── colors.py           # Global camera color palette
│       ├── frame_utils.py      # BGR→QPixmap conversion
│       ├── tabs/
│       │   ├── cameras_tab.py  # Tab 1: Record (cameras + preview + FPS graph + recording)
│       │   ├── intrinsic_tab.py# Tab 2: Intrinsic calibration
│       │   ├── extrinsic_tab.py# Tab 3: Extrinsic calibration
│       │   └── process_tab.py  # Tab 4: Tracking + triangulation
│       └── widgets/
│           ├── camera_grid.py  # Multi-camera display grid
│           └── video_player.py # Video scrubber/playback
│
└── native/                     # C++ camera module
    ├── calimerge_platform.h    # Platform-independent API header
    ├── calimerge_macos.mm      # macOS: AVFoundation
    ├── calimerge_win32.cpp     # Windows: Media Foundation
    ├── build_macos.sh          # macOS build script
    └── build_win32.bat         # Windows build script
```

### GUI Tabs

| Tab | Name | Purpose |
|-----|------|---------|
| 1 | Record | Camera detection, live preview, settings (resolution/FPS/exposure), FPS graph, synchronized recording |
| 2 | Intrinsic | Per-camera lens calibration from ChArUco board videos. Shows detection overlay + distortion results |
| 3 | Extrinsic | Multi-camera spatial calibration via bundle adjustment |
| 4 | Process | 2D tracking + triangulation → 3D export (placeholder) |

---

## Core Data Structures

All data types are in `src/calimerge/types.py` as frozen dataclasses. No methods — logic lives in pure functions.

### Camera Configuration

```python
@dataclass(frozen=True, slots=True)
class CameraConfig:
    serial_number: str
    port: int
    enabled: bool = True
    resolution: tuple[int, int] = (1280, 720)  # (width, height)
    rotation_count: int = 0                     # 0=0°, 1=90°, 2=180°, 3=270°
    exposure: int = -4                          # Platform-specific units
```

### Calibration Results

```python
@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    serial_number: str
    resolution: tuple[int, int]     # (width, height)
    matrix: np.ndarray              # 3x3 camera matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
    distortion: np.ndarray          # (5,) distortion coefficients [k1, k2, p1, p2, k3]
    error: float                    # RMSE reprojection error
    grid_count: int                 # Number of grids used

@dataclass(frozen=True, slots=True)
class CameraExtrinsics:
    rotation: np.ndarray            # 3x3 rotation matrix
    translation: np.ndarray         # (3,) translation vector

@dataclass(frozen=True, slots=True)
class CalibratedCamera:
    serial_number: str
    port: int
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
```

### ChArUco Board

```python
@dataclass(frozen=True)
class CharucoConfig:
    columns: int
    rows: int
    square_size_cm: float           # Square edge in cm (auto-converts to meters)
    dictionary: str = "DICT_4X4_50"
    inverted: bool = False          # White markers on black background
    legacy_pattern: bool = False

    @property
    def square_size_m(self) -> float    # Computed from cm
    @property
    def marker_size_m(self) -> float    # 75% of square size
```

### Point Data (Detection → Triangulation pipeline)

```python
@dataclass(frozen=True, slots=True)
class PointPacket:
    """2D points detected in a single frame."""
    point_id: np.ndarray | None     # (n,) unique point identifiers
    img_loc: np.ndarray | None      # (n, 2) image coordinates (x, y)
    obj_loc: np.ndarray | None      # (n, 3) object-space coords (calibration only)
    confidence: np.ndarray | None   # (n,) confidence scores

@dataclass(frozen=True, slots=True)
class FramePoints:
    """Points from a single camera frame."""
    port: int
    frame_index: int
    points: PointPacket
    timestamp_ns: int = 0

@dataclass(frozen=True, slots=True)
class SyncedPoints:
    """Points from all cameras at one sync index."""
    sync_index: int
    frame_points: dict[int, FramePoints | None]  # port -> FramePoints

@dataclass(frozen=True, slots=True)
class XYZPoints:
    """Triangulated 3D points."""
    sync_index: int
    point_ids: np.ndarray           # (n,)
    xyz: np.ndarray                 # (n, 3)
```

### Project Configuration

```python
@dataclass(frozen=True, slots=True)
class ProjectConfig:
    fps: int
    cameras: dict[str, CameraConfig]   # serial_number -> config
    charuco_intrinsic: CharucoConfig    # For per-camera calibration
    charuco_extrinsic: CharucoConfig    # For multi-camera calibration
    pose_backend: Literal["charuco", "mediapipe", "vitpose"] = "charuco"
    pose_device: str = "cpu"            # "cpu", "cuda", "mps"
    max_persons: int = 1
```

### Helper Functions (types.py)

```python
compute_transformation_matrix(extrinsics) -> np.ndarray    # 4x4 homogeneous
compute_projection_matrix(camera) -> np.ndarray             # 3x4 projection
extrinsics_to_vector(extrinsics) -> np.ndarray              # 6-element [rodrigues, translation]
extrinsics_from_vector(vector) -> CameraExtrinsics          # Inverse of above
get_projection_matrices(cameras) -> dict[int, np.ndarray]   # All projection matrices
```

---

## Native Camera Module (C++)

### C Structs (calimerge_platform.h)

```c
CM_Format {
    int width, height, fps;         // Unique (resolution, frame-rate) tuple
};

CM_Camera {
    char serial_number[64];
    char display_name[128];
    int  device_index;
    int  width, height, fps, rotation, exposure;
    bool enabled;
    CM_Format supported_formats[32]; // All supported (w, h, fps) tuples
    int supported_format_count;
    void *platform_handle;          // Opaque — do not touch from Python
};

CM_Frame {
    uint8_t *pixels;                // BGR format (OpenCV compatible)
    int width, height, stride;
    uint64_t timestamp_ns;          // Camera's native PTS (nanoseconds)
    uint64_t arrival_ns;            // Common clock arrival time (mach_absolute_time)
    uint64_t corrected_ns;          // PTS + clock_offset = common clock domain
    int camera_index;
};

CM_SyncedFrameSet {
    CM_Frame frames[16];
    int frame_count;
    int dropped_mask;               // Bit i = 1 if camera i dropped
    uint64_t sync_index;
};
```

### C API

| Function | Purpose |
|----------|---------|
| `cm_init()` / `cm_shutdown()` | Lifecycle |
| `cm_enumerate_cameras(out, max)` | Discover cameras with serial numbers |
| `cm_open_camera(cam)` / `cm_close_camera(cam)` | Open/close for capture |
| `cm_set_resolution(cam, w, h)` | Change resolution (uses activeFormat on macOS) |
| `cm_set_fps(cam, fps)` | Set frame rate |
| `cm_set_exposure(cam, exp)` | Set exposure |
| `cm_capture_frame(cam, out)` | Single frame capture |
| `cm_capture_synced(cams, count, out)` | Multi-camera synchronized capture |
| `cm_release_frame(f)` / `cm_release_synced(fs)` | Free pixel buffers |

### Python Bindings (camera_binding.py)

Thin ctypes wrapper. Key Python data classes:

```python
CameraInfo:     serial_number, display_name, device_index, width, height, fps, exposure, _c_camera
Frame:          pixels (np.ndarray BGR), width, height, timestamp_ns, arrival_ns, corrected_ns
SyncedFrameSet: frames (dict[int, Frame|None]), sync_index, dropped_mask
```

### macOS Implementation Details (calimerge_macos.mm)

- Per-camera `AVCaptureSession` with serial-dispatched capture queue
- Ring buffer (8 frames, mutex+condvar protected) for continuous capture
- Frame delegate converts BGRA→BGR, stores both camera PTS and arrival timestamp
- Clock offset calibration at startup (median of first frames)
- Resolution change via `device.activeFormat` (not session presets)
- `ring_buffer_flush()` clears stale frames on resolution change

---

## GUI Architecture

### State Management (state.py)

```python
AppState (frozen dataclass):
    project_path, project_config, charuco_config
    cameras: dict[int, CameraState]     # port -> CameraState
    is_previewing: bool
    recording: RecordingState
    calibration: CalibrationState       # intrinsics + extrinsics results
    processing: ProcessingState
    current_tab, status_message

StateManager (QObject):
    # Thin coordinator — holds AppState, emits signals, spawns workers
    # Does NOT contain business logic
    state_changed, cameras_changed, calibration_changed, ...
    update_state(**kwargs)              # Immutable update via replace()
```

### Workers (workers.py)

All workers are `QThread` subclasses that emit signals. They do NOT modify state directly.

| Worker | Signals | Purpose |
|--------|---------|---------|
| `CameraEnumerateWorker` | `cameras_found`, `error` | List available cameras |
| `CameraPreviewWorker` | `frame_captured(port, pixels)`, `error` | Continuous frame grab (retries on transient errors) |
| `RecordingWorker` | `log_message`, `progress_update`, `frame_captured`, `recording_finished` | Synced video recording to disk |
| `IntrinsicCalibrationWorker` | `log_message`, `progress_update`, `detection_frame(idx, vis, count)`, `calibration_finished` | ChArUco detection + OpenCV calibration |
| `ExtrinsicCalibrationWorker` | `log_message`, `progress_update`, `calibration_finished(cameras, rmse)` | Bundle adjustment |
| `ProcessingWorker` | `log_message`, `progress_update`, `processing_finished` | Tracking + triangulation (placeholder) |

### Camera Colors (colors.py)

Global palette for consistent camera identification across the GUI:

```python
CAMERA_COLORS = [
    QColor(80, 200, 120),    # 0: green
    QColor(100, 160, 255),   # 1: blue
    QColor(255, 180, 80),    # 2: orange
    QColor(220, 100, 220),   # 3: purple
    QColor(255, 100, 100),   # 4: red
    QColor(100, 220, 220),   # 5: cyan
    QColor(255, 220, 80),    # 6: yellow
    QColor(180, 140, 255),   # 7: lavender
]

camera_color(port) -> QColor           # Get color for port index
camera_color_hex(port) -> str          # "#50c878" format
```

Used by: camera grid borders, camera labels, FPS graph lines.

---

## Calibration Pipeline

### 1. Intrinsic Calibration (calibration/intrinsic.py)

Per-camera lens parameter estimation from ChArUco board videos.

```
Video → detect_charuco_points(frame, config) → PointPacket
                                                  ↓
              collect N frames with ≥4 corners detected
                                                  ↓
       calibrate_intrinsics(packets, resolution, serial) → CameraIntrinsics
                                                              ↓
                                              saved to SQLite DB
```

Key functions:
- `detect_charuco_points(frame, config, board)` → `PointPacket` — detects corners, tries mirrored if needed, supports inverted boards
- `calibrate_intrinsics(packets, resolution, serial)` → `CameraIntrinsics` — wraps `cv2.calibrateCamera`
- `filter_frames_for_calibration(packets, target_count)` → well-spaced subset (available but not wired in yet)

### 2. Extrinsic Calibration (calibration/extrinsic.py)

Multi-camera spatial calibration via stereo pairs + bundle adjustment.

```
Per-camera videos → detect_charuco_points per frame
                          ↓
        sync frames across cameras → SyncedPoints list
                          ↓
     stereo_calibrate_pair() for each camera pair → pairwise R, T
                          ↓
     compute_initial_extrinsics() → chain to all cameras
                          ↓
     build_point_estimates() → triangulate initial 3D points
                          ↓
     run_bundle_adjustment() → joint optimization (scipy.least_squares)
                          ↓
     dict[int, CalibratedCamera], rmse
```

Convenience function: `run_extrinsic_from_videos(video_paths, intrinsics, charuco_config, ...)` wraps the full pipeline.

### 3. Triangulation (triangulation.py)

Adapted from Anipose (BSD-2 licensed). Uses Numba JIT for performance.

```
2D points from N cameras + projection matrices → triangulate_frame() → XYZPoints
```

### ChArUco Utilities (calibration/charuco.py)

- `ARUCO_DICTIONARIES` — mapping of string names to OpenCV constants
- `create_charuco_board(config)` → `cv2.aruco.CharucoBoard`
- `generate_board_image(config, width, height)` → BGR numpy array (supports inverted)
- `create_charuco_pdf(config, filename)` — printable PDF

---

## Configuration & Storage (config.py)

### TOML Project Config

```toml
fps = 30

[cameras.ABC123]    # keyed by serial number
enabled = true
resolution = [1280, 720]
rotation = 0
exposure = -4

[charuco_intrinsic]
columns = 7
rows = 5
square_size_cm = 3.0
dictionary = "DICT_4X4_50"

[charuco_extrinsic]
columns = 7
rows = 5
square_size_cm = 5.0
```

Functions: `load_project_config()`, `save_project_config()`, `create_default_project_config()`

### SQLite Intrinsics Database

Location: `~/.calimerge/intrinsics.db` (global, shared across projects)
Key: `(serial_number, resolution_width, resolution_height)`

Functions: `init_intrinsics_db()`, `save_intrinsics()`, `load_intrinsics()`, `list_intrinsics()`, `delete_intrinsics()`

### Extrinsic Calibration Output

`save_calibration_to_toml()` / `load_calibration_from_toml()` — per-project extrinsic results.

---

## Recording Output Format

When recording from the Record tab:

```
recordings/20250203_143000/
├── port_0.mp4              # Video per camera
├── port_1.mp4
├── frame_time_history.csv  # sync_index, port, frame_index, frame_time
└── camera_mapping.csv      # port, serial_number, display_name
```

---

## Design Principles

### Data-Oriented Architecture

- All data in `@dataclass(frozen=True, slots=True)` — no methods beyond computed properties
- All logic in standalone pure functions
- No hidden state; data flow is explicit
- Exception: PySide6 widgets use classes but are kept thin (UI only)

### C++ Camera Module (Handmade Hero style)

- Plain C structs + free functions — no member functions, no templates, no CMake
- Unity build per platform (single build script)
- No STL in hot paths — fixed-size arrays, ring buffers
- Platform layer abstraction: one `.mm`/`.cpp` per OS
- All frames output as BGR (OpenCV compatible)
- Trust OpenCV; bias against other library imports

### GUI Design

- `StateManager` is a thin QObject coordinator — does NOT contain business logic
- Workers run in QThread, emit signals, never modify state directly
- `AppState` is immutable — updates via `dataclasses.replace()`
- Camera colors are global constants for consistent identification
