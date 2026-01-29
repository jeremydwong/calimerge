# Calimerge

Unified multi-camera motion capture: synchronized recording, calibration, and 3D pose estimation.

> **Status:** Early development. Core calibration pipeline complete. macOS camera module working. Windows/Linux WIP.

## Quick Start (macOS)

```bash
# Clone and setup
git clone <repo>
cd calimerge
uv sync

# Build the native camera library
cd src/native
./build_macos.sh release
cd ../..

# Launch the unified GUI
uv run calimerge
```

## Commands

| Command | Description |
|---------|-------------|
| `uv run calimerge` | Launch unified GUI (cameras, record, calibrate, process) |
| `uv run calimerge gui` | Same as above |
| `uv run calimerge clock` | Display sync verification clock (10ms updates) |
| `uv run calimerge record` | Launch legacy recording GUI |
| `uv run calimerge --help` | Show all commands |

## Unified GUI

The main application (`uv run calimerge`) provides a tabbed workflow:

### 1. Cameras Tab
- **Detect cameras** connected to your system
- **Preview** live feeds from each camera
- **Enable/disable** specific cameras for recording
- Cameras identified by unique serial numbers

### 2. Record Tab
- **Synchronized multi-camera recording** with frame timing
- Configure **FPS and duration**
- Output: timestamped folders with per-camera videos

### 3. Intrinsic Tab (Per-Camera Calibration)
- Load calibration videos showing **ChArUco board**
- Configure board: **columns × rows**, **square size (cm)**, dictionary
- Run calibration to compute lens parameters (focal length, distortion)
- **Auto-saves to database** (`~/.calimerge/intrinsics.db`)
- View results: fx, fy, cx, cy, reprojection error

### 4. Extrinsic Tab (Multi-Camera Calibration)
- Load synchronized videos of ChArUco board from all cameras
- Configure **extrinsic board** (typically larger for visibility)
- Run bundle adjustment to compute camera positions
- Export camera rig to TOML file

### 5. Process Tab
- Load multi-camera recordings
- Run 2D tracking → triangulation → 3D export
- (Pose estimation backends: charuco, mediapipe, vitpose)

## ChArUco Board Configuration

Calimerge uses separate board configurations for intrinsic and extrinsic calibration:

| Purpose | Default Size | Square Size | Rationale |
|---------|--------------|-------------|-----------|
| **Intrinsic** | 7×5 | 3 cm | Smaller board, close to camera for good coverage |
| **Extrinsic** | 4×3 | 5 cm | Larger squares visible from multiple cameras |

The marker size is automatically computed as **75% of square size** (standard ratio).

Configure in TOML:
```toml
[charuco_intrinsic]
columns = 7
rows = 5
square_size_cm = 3.0
dictionary = "DICT_4X4_50"

[charuco_extrinsic]
columns = 4
rows = 3
square_size_cm = 5.0
dictionary = "DICT_4X4_50"
```

## Workflow: Camera Calibration

### Step 1: Intrinsic Calibration (per camera)

1. Print a ChArUco board (7×5, 3cm squares recommended)
2. Record video of the board from each camera:
   - Move board through entire frame
   - Vary distance and angle
   - Capture 50+ frames with good detections
3. In **Intrinsic Tab**:
   - Load video for each camera
   - Set board parameters to match your print
   - Click "Calibrate"
   - Results auto-save to database

### Step 2: Extrinsic Calibration (multi-camera)

1. Print a larger ChArUco board (4×3, 5cm squares recommended)
2. Record synchronized video from all cameras:
   - Move board so it's visible from at least 2 cameras
   - Cover the capture volume
3. In **Extrinsic Tab**:
   - Ensure intrinsics are ready for all cameras
   - Load synchronized videos
   - Set extrinsic board parameters
   - Click "Run Extrinsic Calibration"
   - Export camera rig

### Step 3: Process Recordings

1. Record synchronized motion capture session
2. In **Process Tab**:
   - Load videos
   - Run tracking + triangulation
   - Export 3D points

## Workflow: Verifying Camera Synchronization

1. **Run the clock display:**
   ```bash
   uv run calimerge clock
   ```
   This shows a real-time clock with millisecond precision.

2. **Point all cameras at the clock display**

3. **In another terminal, start recording:**
   ```bash
   uv run calimerge record
   ```

4. **Record a few seconds, then stop**

5. **Check recordings in `recordings/<timestamp>/`:**
   - `port_X.mp4` - Video files per camera
   - `frame_time_history.csv` - Frame timing data
   - `camera_mapping.csv` - Camera serial → port mapping

## Native Test Executables

After building, these tests are available in `src/native/`:

```bash
./test_enumerate    # List detected cameras with serial numbers
./test_capture      # Capture frames from single camera
./test_multi        # Multi-camera capture test
./test_sync_log     # Log synchronization timing data
```

## Building the Native Library

### macOS (working)

```bash
cd src/native
./build_macos.sh release    # or 'debug' for symbols
```

Produces `libcalimerge.dylib` using AVFoundation.

### Windows (WIP)

Not yet implemented. Will use Media Foundation.

### Linux (WIP)

Not yet implemented. Will use V4L2.

## Project Structure

```
calimerge/
├── src/
│   ├── calimerge/              # Python package
│   │   ├── cli.py              # Entry points
│   │   ├── types.py            # Core dataclasses
│   │   ├── config.py           # TOML + SQLite persistence
│   │   ├── camera_binding.py   # ctypes wrapper for C++ lib
│   │   ├── triangulation.py    # Numba-optimized 3D reconstruction
│   │   │
│   │   ├── calibration/        # Calibration algorithms
│   │   │   ├── charuco.py      # Board creation and detection
│   │   │   ├── intrinsic.py    # Per-camera lens calibration
│   │   │   └── extrinsic.py    # Multi-camera bundle adjustment
│   │   │
│   │   └── gui/                # PySide6 interface
│   │       ├── main.py         # MainWindow with tabs
│   │       ├── state.py        # Immutable AppState + StateManager
│   │       ├── workers.py      # QThread workers
│   │       ├── tabs/           # Cameras, Record, Intrinsic, Extrinsic, Process
│   │       └── widgets/        # CameraGrid, VideoPlayer
│   │
│   └── native/                 # C++ camera module
│       ├── calimerge_platform.h
│       ├── calimerge_macos.mm
│       └── build_macos.sh
│
├── tests/                      # Test suite
├── recordings/                 # Output directory
├── caliscope/                  # Legacy: GUI calibration package
├── multiwebcam/                # Legacy: webcam recording package
└── posetrack/                  # Legacy: pose estimation package
```

## Recording Output Format

Each recording session creates a timestamped directory:

```
recordings/20260126_214609/
├── port_0.mp4              # Camera 0 video
├── port_1.mp4              # Camera 1 video
├── port_2.mp4              # Camera 2 video
├── frame_time_history.csv  # Per-frame timing
└── camera_mapping.csv      # Serial number → port mapping
```

### camera_mapping.csv
```csv
port,serial_number,display_name
0,6C707041-05AC-0010-0006-000000000001,MacBook Pro Camera
1,0x21400000525a4b1,Nuroum V11
2,0x11000000525a4b1,Nuroum V11
```

### frame_time_history.csv
```csv
sync_index,port,frame_index,frame_time
0,0,0,0.002704
0,1,0,0.002704
0,2,0,0.002704
1,0,1,0.036037
...
```

## Development

```bash
# Install dependencies (including dev)
uv sync --all-extras

# Run tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Lint
uv run ruff check .

# Auto-fix lint issues
uv run ruff check --fix .
```

## Test Suite

The test suite covers all core modules with 81 tests across 6 test files:

### test_types.py (20 tests)
Tests for core dataclasses and pure functions:
- `CameraConfig` - camera settings with defaults
- `CameraIntrinsics` - lens parameters (matrix, distortion)
- `CameraExtrinsics` - rotation/translation
- `CalibratedCamera` - combined intrinsics + extrinsics
- `CharucoConfig` - board configuration with cm→m conversion
- `PointPacket` - 2D point storage
- `XYZPoints` - 3D triangulated points
- `ProjectConfig` - separate intrinsic/extrinsic charuco configs
- `compute_projection_matrix()` - 3x4 projection matrix computation

### test_config.py (12 tests)
Tests for configuration persistence:
- TOML save/load roundtrip for `ProjectConfig`
- SQLite intrinsics database (init, save, load, list, delete)
- Calibration TOML export/import with intrinsics reference

### test_charuco.py (15 tests)
Tests for ChArUco board handling:
- ArUco dictionary mapping (DICT_4X4_50, etc.)
- Board creation with different dictionaries
- Board image generation (normal and inverted)
- Object point extraction (3D coordinates)
- Connected corner computation
- Corner distance calculations

### test_intrinsic.py (6 tests)
Tests for intrinsic calibration:
- ChArUco point detection in synthetic images
- Detection returns object points (3D coordinates)
- Empty detection on blank images
- Prebuilt board passthrough
- Minimum frame validation (requires ≥3 frames)
- Full calibration with synthetic data

### test_triangulation.py (9 tests)
Tests for 3D reconstruction:
- Point undistortion with zero distortion
- Undistortion output shape preservation
- Empty input handling
- Principal point invariance
- Single point triangulation from stereo cameras
- Minimum camera requirement (needs ≥2 views)
- Empty frame handling
- Shared point triangulation
- Single-camera point exclusion

### test_state.py (19 tests)
Tests for GUI state management:
- `CameraState`, `RecordingState`, `CalibrationState`, `ProcessingState` defaults
- `AppState` immutability (frozen dataclass)
- `StateManager` operations (update, set_status, report_error)
- Nested state updates (recording, calibration, processing)
- Camera state management (set, update, nonexistent handling)
- State immutability preservation across updates

Run specific test files:
```bash
uv run pytest tests/test_types.py -v
uv run pytest tests/test_calibration.py -v
uv run pytest tests/test_triangulation.py -v
```

## Legacy Packages

The original packages are preserved for reference during migration:

- **caliscope/** - Full calibration + pose estimation GUI (Poetry)
- **multiwebcam/** - Synchronized webcam recording (Poetry)
- **posetrack/** - VitPose-based pose estimation

To run legacy packages:
```bash
cd caliscope && poetry install && poetry run caliscope
cd multiwebcam && poetry install && poetry run mwc clock
```

## Architecture Notes

See [CLAUDE.md](CLAUDE.md) for detailed design documentation including:
- Data-oriented architecture principles
- C++ camera module design (Handmade Hero style)
- Platform abstraction layer
- Migration roadmap

## License

BSD-2-Clause
