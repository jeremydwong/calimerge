# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains two related Python packages for multicamera motion capture:

- **caliscope**: GUI-based multicamera calibration and 3D pose estimation
- **multiwebcam**: Concurrent webcam recording for synchronized video capture

Both packages use Poetry for dependency management and PySide6 for the GUI.

## Build and Development Commands

### Caliscope (in `caliscope/` directory)

```bash
# Install dependencies
poetry install

# Run the application
poetry run caliscope

# Run tests
poetry run pytest

# Run a single test
poetry run pytest tests/test_calibration.py::test_name

# Lint code
poetry run ruff check .

# Auto-fix lint issues
poetry run ruff check --fix .
```

### MultiWebCam (in `multiwebcam/` directory)

```bash
# Install dependencies
poetry install

# Run the application
poetry run mwc

# Run with clock overlay for timestamp verification
poetry run mwc clock
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
