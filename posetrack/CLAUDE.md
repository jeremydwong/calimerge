# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PoseTrack is a Python project for multi-camera 3D human pose estimation and tracking. It processes synchronized video feeds from multiple cameras to detect, track, and triangulate 3D human poses using transformer-based models (DETR for detection, VitPose for pose estimation).

## Key Dependencies & Environment

- **Environment**: Use conda environment named `posetrack`
- **Setup**: `conda env create -f environment.yaml`
- **Key libraries**: PyTorch, OpenCV, Transformers (Hugging Face), Supervision, Pandas, NumPy

## Development Commands

### Running Tests
```bash
# Test camera calibration parsing
python test/test_cs_parse.py

# Test pose estimation on single images
python test/test_estimate_poses.py

# Test multi-camera video processing
python test/test_mwc_video.py
```

### Core Processing
```bash
# Process synchronized multi-camera video for 3D pose tracking
python src/process_synced_poses.py

# Example with custom parameters
python src/process_synced_poses.py --csv_path test/caliscope/coord_3x1_3/frame_time_history.csv --calibration_path test/caliscope/coord_3x1_3/config.toml --video_dir test/caliscope/coord_3x1_3 --output_path output/test_output.csv --device mps
```

## Architecture

### Core Modules

1. **`src/pose_detector.py`**: Main pose detection pipeline
   - `load_models()`: Loads DETR (detection) and VitPose (pose estimation) models
   - `detect_persons()`: Detects human bounding boxes in images
   - `estimate_poses()`: Estimates 52-keypoint poses for detected persons
   - Uses local model directories specified by `LOCAL_SP_DIR` and `LOCAL_DET_DIR`

2. **`src/cs_parse.py`**: Camera calibration parsing and 3D triangulation
   - `parse_calibration_mwc()`: Parses MWC-format calibration files
   - `parse_calibration_fmc()`: Parses FMC-format calibration files
   - `calculate_projection_matrices()`: Computes camera projection matrices
   - `triangulate_keypoints()`: Triangulates 3D points from multi-view 2D keypoints

3. **`src/process_synced_poses.py`**: Multi-camera synchronized processing
   - Main processing pipeline for synchronized video frames
   - Implements person tracking across frames using head position matching
   - Outputs both CSV (wide format) and pickle files with 3D pose data

4. **`src/libwalk.py`**: Interactive visualization utilities for pose data analysis

### Data Flow

1. **Input**: Multi-camera synchronized videos + calibration files + frame timing CSV
2. **Detection**: DETR model detects persons in each camera view
3. **Pose Estimation**: VitPose estimates 52 keypoints per detected person
4. **Tracking**: Tracks persons across frames using head keypoint matching
5. **Triangulation**: Computes 3D poses from multi-view 2D keypoints
6. **Output**: CSV with 3D coordinates and pickle files for further processing

### Keypoint Format

The project uses a 52-keypoint model (SynthPose) extending standard COCO keypoints with additional anatomical landmarks. Keypoints are defined in `SynthPoseMarkers` class in `pose_detector.py`.

## Model Setup

Before running, you must:
1. Download models from Hugging Face:
   - DETR: `PekingU/rtdetr_r50vd_coco_o365`
   - SynthPose: `stanfordmimi/synthpose-vitpose-base-hf`
2. Update `LOCAL_SP_DIR` and `LOCAL_DET_DIR` in `src/pose_detector.py` to point to your local model directories

## Calibration Formats

- **MWC**: Multi-camera calibration format (from Caliscope)
- **FMC**: FreeMoCap calibration format
- Both formats contain camera intrinsics, extrinsics, and distortion parameters

## Test Data Structure

- `test/caliscope/`: Contains multi-camera test datasets
- `test/calibration/`: Camera calibration test files
- `test/image/`: Single image test cases
- Each dataset includes: config.toml, frame_time_history.csv, and port_X.mp4 files

## Output Format

The system outputs:
- **CSV**: Wide format with columns for sync_index, person_id, and X/Y/Z coordinates for each keypoint
- **Pickle**: Raw Python objects for further processing
- **Debug images**: Visualization of detected poses (when enabled)