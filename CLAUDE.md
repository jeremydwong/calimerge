# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup for Claude

**IMPORTANT**: uv is installed at `~/.local/bin/uv`. Add to PATH before running commands:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Or use the full path directly:

```bash
~/.local/bin/uv sync
~/.local/bin/uv run calimerge clock
~/.local/bin/uv run python3 -c "from calimerge import types; print('ok')"
```

## Repository Overview

This repository is **calimerge** - a unified multi-camera motion capture application merging:

- **caliscope** (legacy, in `caliscope/`): GUI calibration and 3D pose estimation
- **multiwebcam** (legacy, in `multiwebcam/`): Synchronized webcam recording
- **posetrack** (legacy, in `posetrack/`): VitPose-based pose estimation

The new unified package is in `src/calimerge/` and uses **uv** (not Poetry).

## Build and Development Commands

### Calimerge (unified package)

```bash
# Setup (first time)
~/.local/bin/uv sync

# Build native camera library (macOS)
cd src/native && ./build_macos.sh release && cd ../..

# Run applications
~/.local/bin/uv run calimerge clock    # Sync verification clock
~/.local/bin/uv run calimerge record   # Recording GUI

# Run tests
~/.local/bin/uv run pytest

# Lint
~/.local/bin/uv run ruff check src/

# Test imports
~/.local/bin/uv run python3 -c "from calimerge.types import CameraConfig; print('ok')"
```

### Native C++ Tests (after building)

```bash
cd src/native
./test_enumerate    # List cameras
./test_capture      # Single camera test
./test_multi        # Multi-camera test
```

### Legacy Packages (for reference only)

```bash
# Caliscope (in caliscope/ directory)
cd caliscope && poetry install && poetry run caliscope

# MultiWebCam (in multiwebcam/ directory)
cd multiwebcam && poetry install && poetry run mwc clock
```

## Architecture

### Caliscope Core Pipeline

1. **Intrinsic Calibration**: `calibration/intrinsic_calibrator.py` estimates per-camera optical properties (focal length, optical center, distortion) from ChArUco board videos

2. **Extrinsic Calibration**: `calibration/stereocalibrator.py` + `calibration/capture_volume/` performs bundle adjustment to estimate relative camera positions. Key classes:
   - `CaptureVolume`: Holds camera array + point estimates, runs bundle adjustment via scipy least_squares
   - `PointEstimates`: Manages 3D point data used in optimization
   - `QualityController`: Filters outliers during calibration

3. **Tracking**: `tracker.py` defines abstract `Tracker` base class. Implementations in `trackers/`:
   - `CharucoTracker`: For calibration board detection
   - `pose_tracker.py`, `hand_tracker.py`, `face_tracker.py`: Mediapipe-based trackers
   - `simple_holistic_tracker.py`: Full body tracking

4. **Triangulation**: `triangulate/triangulation.py` converts 2D tracked points from multiple cameras to 3D positions (adapted from Anipose, BSD-2 licensed)

5. **Post-Processing**: `post_processing/post_processor.py` orchestrates the full pipeline: 2D tracking → gap filling → triangulation → smoothing → export

### Key Data Structures

- `CameraArray` / `CameraData` (`cameras/camera_array.py`): Holds intrinsic/extrinsic parameters per camera
- `Configurator` (`configurator.py`): Manages workspace config via `config.toml` files
- `Controller` (`controller.py`): Qt-based orchestrator connecting GUI to backend processing
- `PointPacket` / `XYZPacket` (`packets.py`): Data containers for 2D and 3D point data

### Workspace Structure

Caliscope projects use a directory structure managed by `WorkspaceGuide`:
- `intrinsic/`: Per-camera calibration videos
- `extrinsic/`: Multi-camera calibration recordings
- `recording/`: Motion capture recordings
- `config.toml`: Project configuration

### MultiWebCam

Simpler architecture focused on synchronized video capture:
- `cameras/`: Camera detection and management
- `recording/`: Video recording with timestamp synchronization
- `gui/`: PySide6 interface for camera control
- `configurator.py`: Project settings via `recording_config.toml`

## Testing

Tests are in `caliscope/tests/` and use pytest. Test sessions with pre-recorded data are in `tests/sessions/`. The test suite runs on Python 3.10-3.11 across Linux, macOS, and Windows via GitHub Actions.

## Configuration

Both projects use `rtoml` for TOML config files. Ruff is configured for linting with 120 character line length and Python 3.10 target.

---

# Calimerge: Unified Application Design

The goal is to merge caliscope, multiwebcam, and posetrack into a single unified application for multi-camera motion capture.

## Design Principles

### 1. Data-Oriented Architecture

**Goal:** Replace complex class hierarchies with simple data structures and pure functions.

- Use `@dataclass` or `NamedTuple` for data containers (no methods beyond `__post_init__`)
- All logic in standalone functions: `process(data: DataStruct) -> Result`
- No hidden state; data flow should be explicit and traceable
- Exception: PySide6 widgets require classes, but keep them thin (UI only, delegate logic to functions)

**Refactoring pattern:**
```python
# Before (current style)
class Camera:
    def __init__(self, port):
        self.port = port
        self.cap = cv2.VideoCapture(port)
    def read_frame(self):
        return self.cap.read()

# After (target style)
@dataclass
class CameraState:
    port: int
    cap: cv2.VideoCapture

def open_camera(port: int) -> CameraState:
    return CameraState(port=port, cap=cv2.VideoCapture(port))

def read_frame(camera: CameraState) -> tuple[bool, np.ndarray]:
    return camera.cap.read()
```

### 2. Unified Application Structure

**Single app with tabbed workflow:**

1. **Cameras** - Detection, identification, preview, settings
2. **Record** - Synchronized multi-camera recording
3. **Intrinsic** - Per-camera lens calibration (ChArUco)
4. **Extrinsic** - Multi-camera spatial calibration (bundle adjustment)
5. **Process** - 2D pose estimation → triangulation → export

### 3. Camera System: C++ Core Module

**Goal:** High-performance camera capture with unique device identification, following Casey Muratori / Handmade Hero principles.

#### C++ Design Rules

- **NO member functions** - Use plain C-style structs + free functions
- **NO templates** - Use macros or explicit code generation where needed
- **NO CMake** - Unity build with a single build script per platform
- **NO STL containers in hot paths** - Fixed-size arrays, arena allocators
- **USE Libraries with discretion** - Prefer to write our own. there is a huge benefit in being able to read our own mathematics code or processing. we Trust OpenCV but we are biased away from trusting other library imports. 
- **Platform layer abstraction** - Handmade Hero style separation:
  ```
  calimerge_platform.h    // Platform-independent interface
  calimerge_macos.cpp     // macOS: AVFoundation
  calimerge_win32.cpp     // Windows: Media Foundation / DirectShow
  calimerge_linux.cpp     // Linux: V4L2
  ```

#### Python design rules
Should follow the supset of C++ design rules that make sense:
- ** NO member functions in the final design
- minimal data structures that accomplish the task

#### Architecture

```c
// Core data structures (plain structs, no methods)
struct CameraDevice {
    char serial_number[64];
    char display_name[128];
    int device_index;
    int width, height;
    float fps;
    bool enabled;
};

struct CameraFrame {
    uint8_t *pixels;        // Always BGR (OpenCV compatible) - platform layer converts
    int width, height;
    int stride;             // Bytes per row (may include padding)
    uint64_t timestamp_ns;  // Monotonic nanoseconds from platform clock
    int camera_index;
};

struct SyncedFrameSet {
    CameraFrame *frames;
    int frame_count;
    uint64_t sync_index;
};

// Platform-independent API (implemented per-platform)
int  cm_enumerate_cameras(CameraDevice *out_devices, int max_devices);
int  cm_open_camera(CameraDevice *device);
void cm_close_camera(CameraDevice *device);
int  cm_capture_frame(CameraDevice *device, CameraFrame *out_frame);
void cm_get_serial_number(int device_index, char *out_serial, int max_len);
```

#### Build System

Single-file unity build per platform:
```bash
# macOS
clang++ -O2 -framework AVFoundation -framework CoreMedia \
    -shared -o libcalimerge.dylib build_macos.cpp

# Linux
g++ -O2 -lv4l2 -shared -o libcalimerge.so build_linux.cpp

# Windows (from Developer Command Prompt)
cl /O2 /LD build_win32.cpp mfplat.lib mfreadwrite.lib
```

#### Python Binding

Thin ctypes wrapper (no pybind11 complexity):
```python
import ctypes

lib = ctypes.CDLL("libcalimerge.dylib")

class CameraDevice(ctypes.Structure):
    _fields_ = [
        ("serial_number", ctypes.c_char * 64),
        ("display_name", ctypes.c_char * 128),
        ("device_index", ctypes.c_int),
        # ...
    ]

def enumerate_cameras() -> list[CameraDevice]:
    devices = (CameraDevice * 16)()
    count = lib.cm_enumerate_cameras(devices, 16)
    return list(devices[:count])
```

#### Features (matching MWC)

- Camera enable/disable via config file
- Per-camera resolution and FPS settings
- Rotation (0°, 90°, 180°, 270°)
- Exposure control where supported
- Frame timestamp synchronization

#### Frame Format Contract

**Platform layer responsibility:** Convert native format to BGR (OpenCV-compatible).
- **macOS AVFoundation**: Receives BGRA/YUV from CMSampleBuffer → convert to BGR
- **Windows Media Foundation**: Receives NV12/YUY2/RGB → convert to BGR
- **Linux V4L2**: Receives YUYV/MJPEG → convert to BGR

This keeps all downstream code (calibration, tracking, encoding) platform-agnostic.

#### Unique Camera Identification

Each platform provides serial number extraction:
- **macOS**: `AVCaptureDevice.uniqueID`
- **Windows**: Device instance path from SetupAPI
- **Linux**: `/sys/class/video4linux/videoN/device/serial` or USB descriptor

Camera intrinsics stored in SQLite database keyed by serial number.

#### Video Encoding Strategy

**Phase 1:** C++ captures frames with precise timestamps → pass to Python → OpenCV `VideoWriter`
- OpenCV can use VideoToolbox on macOS (hardware H.264) when available
- Simple, works immediately

**Future option:** If encoding becomes a bottleneck, add `cm_encode_frame()` to C++ API using platform encoders directly.

### 4. Pluggable Pose Estimation

Support multiple backends with a common interface:

- **Mediapipe** (existing caliscope approach) - 33 keypoints, lightweight
- **VitPose/SynthPose** (posetrack approach) - 52 keypoints, transformer-based

Configuration:
```python
@dataclass
class PoseConfig:
    backend: Literal["mediapipe", "vitpose"]
    max_persons: int
    device: str  # "cpu", "cuda", "mps"
```

### 5. Package Management

**Use `uv` exclusively.** No Poetry, no conda.

```bash
# Setup
brew install uv
uv sync

# Run
uv run calimerge
```

## Migration Path

### Phase 1: Cleanup & Analysis
- [ ] Audit and document data flow in caliscope
- [ ] Audit and document data flow in multiwebcam
- [ ] Identify shared components (Camera, Synchronizer, Logger, etc.)
- [ ] Set up uv-based pyproject.toml for new unified package
- [ ] Move existing packages to `legacy/` for reference

### Phase 2: C++ Camera Module
- [ ] Define `calimerge.h` platform-independent API
- [ ] Implement macOS backend (AVFoundation) with serial number extraction
- [ ] Implement Linux backend (V4L2)
- [ ] Implement Windows backend (Media Foundation)
- [ ] Create Python ctypes bindings
- [ ] Build scripts for each platform (no CMake)
- [ ] Test frame capture and synchronization

### Phase 3: Core Python Layer
- [ ] Data structures for calibration (dataclasses, no OOP)
- [ ] Port intrinsic calibration logic
- [ ] Port extrinsic calibration / bundle adjustment
- [ ] Port triangulation
- [ ] Camera intrinsics database (SQLite, keyed by serial)

### Phase 4: Unified GUI
- [ ] Single PySide6 application with tabbed workflow
- [ ] Cameras tab (detection, preview, enable/disable)
- [ ] Record tab (synchronized capture)
- [ ] Intrinsic/Extrinsic calibration tabs
- [ ] Processing tab
- [ ] Unified workspace format

### Phase 5: Pose Estimation
- [ ] Define common pose estimation interface
- [ ] Mediapipe backend (port from caliscope)
- [ ] VitPose/SynthPose backend (port from posetrack)
- [ ] Keypoint format standardization
- [ ] Configurable max persons

## Directory Structure (Target)

```
calimerge/
├── CLAUDE.md
├── pyproject.toml              # uv-managed, single package
├── build.sh                    # Builds C++ for current platform
│
├── src/
│   ├── calimerge/              # Python package
│   │   ├── __init__.py
│   │   ├── __main__.py         # Entry point
│   │   ├── camera_binding.py   # ctypes wrapper for C++ lib
│   │   ├── calibration/        # Intrinsic + extrinsic calibration
│   │   ├── tracking/           # Pose estimation backends
│   │   ├── triangulation/      # 3D reconstruction
│   │   └── gui/                # PySide6 interface
│   │
│   └── native/                 # C++ camera module
│       ├── calimerge.h         # Platform-independent API
│       ├── calimerge_types.h   # Shared data structures
│       ├── calimerge_macos.cpp
│       ├── calimerge_win32.cpp
│       ├── calimerge_linux.cpp
│       └── build_*.cpp         # Unity build entry points
│
├── legacy/                     # Original packages (reference during migration)
│   ├── caliscope/
│   ├── multiwebcam/
│   └── posetrack/
│
└── tests/
```

### 6. Configuration Format

**Keep TOML, extend MWC format.** Maintain backward compatibility with existing `recording_config.toml` files where possible.

Additions to support:
- Camera serial number → port mapping
- Per-camera intrinsic calibration reference (from database)
- Pose estimation backend selection

```toml
# recording_config.toml (extended)
fps = 24

[cameras.ABC123]  # keyed by serial number
enabled = true
resolution = [1280, 720]
rotation = 0
exposure = -4

[cameras.DEF456]
enabled = true
resolution = [1920, 1080]
rotation = 90

[calibration]
intrinsics_db = "~/.calimerge/intrinsics.db"

[pose]
backend = "mediapipe"  # or "vitpose"
max_persons = 2
device = "mps"
```

## Open Questions

- Workspace directory structure: keep current caliscope format or redesign?
- Intrinsics database location: per-project or global (`~/.calimerge/`)?
