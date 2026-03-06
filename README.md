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
| **Intrinsic** | 9×10 | 1 cm | Smaller board, close to camera for good coverage |
| **Extrinsic** | 4×3 | 20 cm | Larger squares visible from multiple cameras |

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

### Windows

Uses Media Foundation for camera capture.

#### Prerequisites

1. **Install uv** (Python package manager):
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Install Visual Studio Build Tools** (if not already installed):
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Select "Desktop development with C++" workload

#### Build Steps

```powershell
# Clone and setup
git clone <repo>
cd calimerge
uv sync

# Build native camera library
cd src\native
build_win32.bat
```

If `build_win32.bat` fails, compile manually from a **Developer Command Prompt for VS**:

```powershell
cl /LD /EHsc /O2 /DNDEBUG calimerge_win32.cpp mfplat.lib mfreadwrite.lib mfuuid.lib ole32.lib /Fe:calimerge.dll
```

#### Test Native Library

```powershell
# After building, test camera enumeration
test_enumerate.exe

# Test single camera capture (camera index 0)
test_capture.exe 0

# Test multi-camera sync
test_multi.exe
```

#### Run the GUI

```powershell
cd ..\..  # Back to project root
uv run calimerge gui
```

#### Troubleshooting

**DLL not found**: Ensure `calimerge.dll` is in `src/native/` - the Python binding looks there.

**No cameras detected**:
- Check Device Manager for camera devices
- Some cameras need manufacturer drivers installed
- Try running as Administrator

**Media Foundation errors**: Windows 10/11 should have MF built-in. If issues persist:
```powershell
Get-WindowsCapability -Online | Where-Object Name -like '*Media*'
```

**Import errors**: Verify the binding can load:
```powershell
uv run python -c "from calimerge.camera_binding import enumerate_cameras; print(enumerate_cameras())"
```

#### Debug Build

For better stack traces, edit `build_win32.bat` to use debug flags:
```batch
cl /LD /EHsc /Zi /DEBUG calimerge_win32.cpp mfplat.lib mfreadwrite.lib mfuuid.lib ole32.lib /Fe:calimerge.dll
```

### Linux (WIP)

Not yet implemented. Will use V4L2.

## CUDA Pose Tracking (Optional)

When CUDA, TensorRT, and OpenCV are available, calimerge can run GPU-accelerated 3D pose tracking that is **~15x faster** than the Python pipeline.

### Prerequisites

- NVIDIA GPU (Compute Capability >= 7.0)
- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-toolkit)
- [TensorRT 10.x](https://developer.nvidia.com/tensorrt)
- [OpenCV 4.x](https://opencv.org/) with FFmpeg support
- MSVC Build Tools 2022

### Building the CUDA Pipeline

```bash
set TENSORRT_PATH=C:\TensorRT
set OPENCV_PATH=C:\OpenCV\opencv\build
src\cuda_pipeline\build_cuda_win32.bat release
```

Produces `pt_main.exe` (CLI) and `calimerge_cuda.dll` (for Python integration).

### Offline Processing (Recorded Videos)

Process pre-recorded multi-camera videos through the full pipeline:

```bash
pt_main.exe <recording_dir> <calibration.toml> [options]

Options:
  --batch-size N       Sync indices per batch (default 8)
  --skip N             Process every Nth sync index (default 1)
  --max-persons N      Max tracked persons (default 2)
  --person-conf F      YOLO detection threshold (default 0.1)
  --yolo PATH          Path to YOLO ONNX model
  --vitpose PATH       Path to VitPose ONNX model
```

**Pipeline stages:**

```
Video Decode (NVDEC or CPU fallback)
  → NV12 → BGR (CUDA kernel, BT.601)
  → Letterbox + Normalize to FP16 640x640 (CUDA kernel, writes __half directly)
  → YOLO v10s Person Detection (TensorRT, FP16 input, FP32 output)
  → Filter Detections (CUDA kernel, class=0 person, undo letterbox)
  → VitPose Crop + Normalize 192x256 (CUDA kernel, 1.25x box expansion, ImageNet stats)
  → VitPose Base COCO (TensorRT, 17 keypoints)
  → Heatmap Decode with DARK Refinement (CUDA kernel, sub-pixel via Taylor expansion)
  → Cross-View Epipolar Matching (CPU, Hungarian algorithm + union-find)
  → SVD Triangulation (CPU, Jacobi eigendecomposition)
  → Multi-Person Tracking (CPU, 3D COM distance matching)
  → CSV Export (52-marker SynthPose format, NaN-padded beyond 17 COCO keypoints)
```

**Performance:** 796 frames x 3 cameras in 12 seconds (199 camera-frames/s). Validated against the Python posetrack pipeline with mean 3D Euclidean error of 4cm across 753 overlapping frames.

### Online Processing (Real-Time Streaming)

A streaming C API accepts live BGR camera frames and returns 3D tracked poses
synchronously. Designed for use with `cm_capture_synced()` from the native
camera module.

```c
PT_Stream *stream;
pt_stream_create(&stream, &config);   // allocate GPU, build engines (~5-30s)

// Per-frame loop (called from capture thread):
PT_StreamResult result;
pt_stream_process_frame(stream, &frameset, &result);  // ~10ms

pt_stream_destroy(stream);
```

Test with recorded videos:
```bash
pt_stream_main.exe <recording_dir> <calibration.toml> [options]
```

**Performance:** 796 frames x 3 cameras in 8.2s (10.0 ms/frame, ~100 sync-frames/s).
Batch vs stream agreement: 0.93 cm mean 3D distance (sub-centimeter).

### Architecture

#### GPU Arena: Single-Allocation Pattern

All GPU memory comes from **one `cudaMalloc`** call. All pinned host memory comes from **one `cudaMallocHost`** call. Every buffer (decode, YOLO input/output, VitPose input/output, keypoints, detection boxes) is carved out of these two blocks via pointer arithmetic with 256-byte alignment. This eliminates GPU memory fragmentation, simplifies the lifecycle (one alloc, one free), and ensures all buffers are naturally aligned for coalesced access. Total allocation: ~100MB GPU + ~18KB pinned host for 3 cameras at 640x480 with batch size 8.

#### Batching Strategy

YOLO inference is batched across multiple sync indices: 8 sync indices x 3 cameras = 24 images processed in a single TensorRT call. VitPose runs **per sync index** (inside the batch loop) because it depends on YOLO detection results that vary per frame. Max VitPose batch = num_cameras x 16 max detections = 48 crops. This split maximizes GPU utilization for YOLO while respecting the data dependency for VitPose.

#### TensorRT FP16 I/O

The letterbox CUDA kernel writes `__half` (FP16) values directly into the arena's YOLO input buffer -- no FP32-to-FP16 conversion step. The TensorRT engine's input tensor type is explicitly set to `kHALF` during engine build so it accepts the FP16 data natively. Output is left as FP32 (the `filter_detections` kernel reads FP32). TensorRT engines are cached to disk with keys encoding `{model_name}_{sm_version}_{max_batch}_{precision}.engine`, so engine rebuilds only happen when the model, GPU, or config changes.

#### Pinned Memory for Async GPU-to-CPU Transfer

Detection counts (a few integers) are copied GPU-to-CPU first via `cudaMemcpyAsync` into pinned host memory, then the stream is synchronized to read the counts on CPU (needed to determine VitPose batch size). Larger transfers (2D keypoints, detection boxes, scores) are also async into pinned buffers. This lets CPU math overlap with pending GPU work when possible.

#### CPU Triangulation (Not GPU)

3D triangulation uses Jacobi eigendecomposition of a 4x4 `A^T*A` matrix (smallest eigenvector = homogeneous 3D point) plus iterative Newton-Raphson undistortion. Each keypoint for each person is a separate tiny matrix problem -- too fine-grained for GPU kernel dispatch overhead. CPU handles it in ~1ms per person per frame. Cross-view matching uses the Hungarian algorithm O(n^3) for n <= 16 (max detections per camera), also better suited to CPU.

#### Precomputed Fundamental Matrices

Fundamental matrices `F[i][j]` for all camera pairs are computed once at startup from the calibration data (via null-space decomposition of the projection matrices, rank-2 enforcement via SVD). Per-frame cross-view matching uses these precomputed matrices for epipolar distance calculations -- no per-frame recomputation as the Python pipeline does.

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
│   ├── native/                 # C++ camera module
│   │   ├── calimerge_platform.h
│   │   ├── calimerge_macos.mm
│   │   └── build_macos.sh
│   │
│   └── cuda_pipeline/          # CUDA pose tracking (optional)
│       ├── pt_pipeline.cpp/h   # Batch pipeline orchestrator
│       ├── pt_arena.cu/h       # Single-allocation GPU arena
│       ├── pt_kernels.cu/h     # CUDA kernels (letterbox, crop, heatmap)
│       ├── pt_tensorrt.cpp/h   # TensorRT engine lifecycle
│       ├── pt_nvdec.cpp/h      # Video decode (NVDEC + CPU fallback)
│       ├── pt_matching.cpp/h   # Cross-view epipolar matching
│       ├── pt_triangulation.cpp/h # SVD 3D reconstruction
│       ├── pt_tracker.cpp/h    # Multi-person tracking
│       ├── pt_export.cpp/h     # CSV export
│       ├── pt_stream.cpp/h     # Real-time streaming API
│       ├── pt_common.h         # Shared constants and structs
│       ├── pt_main.cpp         # Batch pipeline CLI test
│       └── pt_stream_main.cpp  # Streaming pipeline test
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

## Memory Management

### Native Camera Module (C++)

The native library (`libcalimerge.dylib`) uses a clear ownership model:

| Resource | Allocation | Deallocation | Notes |
|----------|------------|--------------|-------|
| `CM_Camera` | Stack/caller | N/A | Plain struct, no dynamic members |
| `MacOSCameraHandle` | `cm_open_camera()` via `calloc()` | `cm_close_camera()` via `free()` | Per-camera state |
| Ring buffer pixels | `ring_buffer_push()` via `malloc()` | `ring_buffer_destroy()` via `free()` | Overwritten each frame |
| Frame pixels (output) | `ring_buffer_get_latest()` via `malloc()` | **Caller** via `cm_release_frame()` | Copied from ring buffer |
| Synced frame pixels | `cm_capture_synced()` via `malloc()` | **Caller** via `cm_release_synced()` | One per camera |

**Key patterns:**

1. **Ring buffer ownership:** Each camera maintains an 8-frame ring buffer. Frames are overwritten as new ones arrive. The buffer owns pixel memory until `ring_buffer_destroy()`.

2. **Output frame copies:** `cm_capture_frame()` and `cm_capture_synced()` return **copies** of pixel data. The caller **must** call `cm_release_frame()` or `cm_release_synced()` to free this memory.

3. **AVFoundation objects:** Managed via `__bridge_retained` (take ownership) and `__bridge_transfer` (release ownership). All ARC objects are released in `cm_close_camera()`.

4. **Thread safety:** Ring buffer protected by `pthread_mutex` + `pthread_cond`. Frame capture blocks until a frame is available or timeout.

### Python Bindings

The Python `camera_binding.py` wraps the C library:

```python
# capture_frame() copies pixels to numpy array and releases C memory
frame = capture_frame(camera)   # Returns Frame with numpy array
# frame.pixels is a numpy copy - safe to use, no manual release needed

# For synced capture:
frameset = capture_synced(cameras)  # Returns SyncedFrameSet
# All frames are numpy copies, C buffers already released
```

**Important:** Python handles memory automatically:
- `capture_frame()` copies pixel data to numpy array, then calls `cm_release_frame()`
- `capture_synced()` copies all frames, then calls `cm_release_synced()`
- Numpy arrays are managed by Python's garbage collector

### Video Recording

Recording uses **direct-write mode**:
- Frames written to disk immediately via `cv2.VideoWriter`
- No in-memory buffering of entire recording
- OpenCV handles internal buffering
- Memory usage stays constant regardless of recording duration

## Architecture Notes

See [CLAUDE.md](CLAUDE.md) for detailed design documentation including:
- Data-oriented architecture principles
- C++ camera module design (Handmade Hero style)
- Platform abstraction layer
- Migration roadmap

## License

BSD-2-Clause

## Todo
<li>
- we have no file menu so far! perhaps we won't need one but we'll probably eventually need one. Implement an 'open project' file menu. show the pathname in a 'status bar' which the applciation does currently have, along the bottom of (all of the ) gui tabs. ASSOCIATED WITH THIS, please save all of the files and configurations for each camera and project there.

1. RECORD TAB
- FPS: 'fps' in settings table for each camera is weird, because you don't usually want to have some cameras running faster than others. moreover, changing one camera changes the framerate of all, suggesting they're locked under the hood. I think this is probably a good way to go, we typically DO want the cameras to all share the same framerate. so, let's remove FPS from the table.
- COLORS: each entry in the textbox window should also have a color field. can be a square of the color in question.
- the name of each camera should be name-serial not just name (often you'll have the same kind of camera)
- the live preview should only show 'enabled' cameras. 
- TABLE: can you guess at the required size of the table columns so that they don't default to kinda generic bad? 
- VIDEO PREVIEW the aspect ratio of the cameras by default is not pleasing because everything is quite wide. assume 4x3 and 2x2 grid, make this subsection an appropriate size. squeeze the text sections as necessary. 
- BUFFERING vs WRITING right now, are we buffering the frames? if so, we could get the required pages of memory ahead of time if using the 'timed' approach. probably best! this is likely best for performance since writing to disk is going to be the limiting performance factor when reading camera frames? 
- how are we encoding the videos? 264 or otherwise? provide this as an option or make it clear in a hover-over. 
- EXPOSURE: changes are not doing anything at the moment! suggest that this command isn't effectual for some reason, worth checking. 


