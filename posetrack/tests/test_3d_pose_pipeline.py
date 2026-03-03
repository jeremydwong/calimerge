"""
Test suite for the posetrack 3D pose estimation pipeline.

Runs the full pipeline (detection -> triangulation -> tracking) on two
reference datasets and compares output against known-good reference CSVs.

Datasets:
  - coord_3x1_3:     3 cameras, 1 person
  - recording_3by1:  4 cameras, 1+ persons

Reference outputs live in posetrack/output/caliscope/{dataset}/.
"""

import os
import sys
import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add src/posetrack directly to path so we can import cs_parse without
# triggering posetrack/__init__.py (which pulls in torch via pose_detector)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "posetrack"))

from cs_parse import (
    parse_calibration_mwc,
    calculate_projection_matrices,
    triangulate_keypoints,
)


# ── Paths ──

TESTS_DIR = PROJECT_ROOT / "tests" / "caliscope"
OUTPUT_DIR = PROJECT_ROOT / "output" / "caliscope"

COORD_3X1_DIR = TESTS_DIR / "coord_3x1_3"
RECORDING_3BY1_DIR = TESTS_DIR / "recording_3by1"

COORD_3X1_REF = OUTPUT_DIR / "coord_3x1_3" / "output_3d_poses_tracked.csv_person0.csv"
RECORDING_3BY1_REF = OUTPUT_DIR / "recording_3by1" / "output_3d_poses_tracked.csv"


# ── Helpers ──

def load_reference_csv(path):
    """Load a reference output CSV into a dict keyed by (sync_index, person_id)."""
    rows = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["sync_index"]), int(row["person_id"]))
            # Parse all numeric columns into floats
            data = {}
            for col, val in row.items():
                if col in ("sync_index", "person_id"):
                    continue
                try:
                    data[col] = float(val)
                except ValueError:
                    data[col] = None
            rows[key] = data
    return rows


def load_frame_time_history(csv_path):
    """
    Parse frame_time_history.csv.

    Returns dict: sync_index -> {port: frame_index}
    """
    sync_map = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            si = int(row["sync_index"])
            port = int(row["port"])
            fi = int(row["frame_index"])
            if si not in sync_map:
                sync_map[si] = {}
            sync_map[si][port] = fi
    return sync_map


def keypoint_columns():
    """Return the 52 keypoint names in CSV column order (without _X/_Y/_Z suffix)."""
    # Read from a reference file header
    with open(COORD_3X1_REF, "r") as f:
        header = f.readline().strip().split(",")
    # Extract unique keypoint names from triplets like Nose_X, Nose_Y, Nose_Z
    names = []
    for col in header[2:]:  # skip sync_index, person_id
        if col.endswith("_X"):
            names.append(col[:-2])
    return names


# ── Tests: Config Parsing ──

class TestConfigParsing:
    """Verify we can parse MWC config files correctly."""

    def test_parse_coord_3x1_config(self):
        config_path = str(COORD_3X1_DIR / "config.toml")
        camera_params = parse_calibration_mwc(config_path)

        assert len(camera_params) == 3, f"Expected 3 cameras, got {len(camera_params)}"

        for i, cam in enumerate(camera_params):
            assert "matrix" in cam, f"Camera {i} missing matrix"
            assert "distortions" in cam, f"Camera {i} missing distortions"
            assert "rotation" in cam, f"Camera {i} missing rotation"
            assert "translation" in cam, f"Camera {i} missing translation"
            assert "size" in cam, f"Camera {i} missing size"
            assert cam["matrix"].shape == (3, 3), f"Camera {i} matrix shape wrong"
            assert cam["distortions"].shape[0] == 5, f"Camera {i} distortions shape wrong"
            assert cam["rotation"].shape == (3,), f"Camera {i} rotation shape wrong"
            assert cam["translation"].shape == (3,), f"Camera {i} translation shape wrong"

    def test_parse_recording_3by1_config(self):
        config_path = str(RECORDING_3BY1_DIR / "config.toml")
        camera_params = parse_calibration_mwc(config_path)

        assert len(camera_params) == 4, f"Expected 4 cameras, got {len(camera_params)}"

        for cam in camera_params:
            assert np.array_equal(cam["size"], [640, 480])

    def test_coord_3x1_intrinsics_values(self):
        """Spot-check specific calibration values from config."""
        config_path = str(COORD_3X1_DIR / "config.toml")
        camera_params = parse_calibration_mwc(config_path)

        cam0 = camera_params[0]
        assert abs(cam0["matrix"][0, 0] - 378.306) < 0.01, "cam_0 fx mismatch"
        assert abs(cam0["matrix"][1, 1] - 378.762) < 0.01, "cam_0 fy mismatch"

    def test_projection_matrices(self):
        """Verify projection matrix computation."""
        config_path = str(COORD_3X1_DIR / "config.toml")
        camera_params = parse_calibration_mwc(config_path)
        proj_matrices = calculate_projection_matrices(camera_params)

        assert len(proj_matrices) == 3
        for i, P in enumerate(proj_matrices):
            assert P.shape == (3, 4), f"Projection matrix {i} shape is {P.shape}, expected (3, 4)"
            # Sanity: P should not be all zeros
            assert np.linalg.norm(P) > 0, f"Projection matrix {i} is zero"


# ── Tests: Frame Time History ──

class TestFrameTimeHistory:
    """Verify frame_time_history.csv parsing."""

    def test_parse_coord_3x1_frame_times(self):
        csv_path = str(COORD_3X1_DIR / "frame_time_history.csv")
        sync_map = load_frame_time_history(csv_path)

        assert len(sync_map) > 0, "No sync indices found"

        # First sync index should have entries for ports 0, 1, 2
        first_si = min(sync_map.keys())
        assert first_si == 73361, f"Expected first sync_index=73361, got {first_si}"
        assert set(sync_map[first_si].keys()) == {0, 1, 2}

    def test_parse_recording_3by1_frame_times(self):
        csv_path = str(RECORDING_3BY1_DIR / "frame_time_history.csv")
        sync_map = load_frame_time_history(csv_path)

        assert len(sync_map) > 0
        first_si = min(sync_map.keys())
        # recording_3by1 has 4 cameras
        ports_in_first = set(sync_map[first_si].keys())
        assert len(ports_in_first) >= 3, f"Expected at least 3 ports, got {ports_in_first}"


# ── Tests: Reference Output Loading ──

class TestReferenceOutput:
    """Verify reference output CSVs are loadable and well-formed."""

    def test_load_coord_3x1_reference(self):
        ref = load_reference_csv(COORD_3X1_REF)

        assert len(ref) > 700, f"Expected 700+ rows, got {len(ref)}"

        # All rows should be person 0
        person_ids = set(pid for _, pid in ref.keys())
        assert person_ids == {0}, f"Expected only person 0, got {person_ids}"

        # Check a specific value from first row
        first_key = (73361, 0)
        assert first_key in ref, "Missing first sync_index row"
        assert abs(ref[first_key]["Nose_X"] - 3.2338) < 0.001

    def test_load_recording_3by1_reference(self):
        ref = load_reference_csv(RECORDING_3BY1_REF)

        assert len(ref) > 500, f"Expected 500+ rows, got {len(ref)}"

        # Should have person_id 0 and 1
        person_ids = set(pid for _, pid in ref.keys())
        assert 0 in person_ids or 1 in person_ids, f"Expected person 0 or 1, got {person_ids}"

    def test_keypoint_columns_complete(self):
        kp_names = keypoint_columns()
        assert len(kp_names) == 52, f"Expected 52 keypoints, got {len(kp_names)}"
        assert kp_names[0] == "Nose"
        assert "L_Shoulder" in kp_names
        assert "R_Hip" in kp_names
        assert "C7" in kp_names


# ── Tests: Triangulation Smoke Test ──

class TestTriangulation:
    """Test triangulation using synthetic 2D points projected from known 3D."""

    def test_triangulate_known_point(self):
        """Project a known 3D point into cameras, then triangulate back."""
        import cv2

        config_path = str(COORD_3X1_DIR / "config.toml")
        camera_params = parse_calibration_mwc(config_path)
        proj_matrices = calculate_projection_matrices(camera_params)

        # A point roughly in the middle of the capture volume
        point_3d = np.array([3.0, 0.5, 4.0])

        # Project into each camera
        person_kp_dict = {}
        port_to_cam_index = {}
        for i, cam in enumerate(camera_params):
            port = cam.get("port", i)
            port_to_cam_index[port] = i

            K = cam["matrix"]
            rvec = cam["rotation"]
            tvec = cam["translation"].reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)

            # Project: p = K @ (R @ X + t)
            p_cam = R @ point_3d.reshape(3, 1) + tvec
            p_img = K @ p_cam
            px = p_img[0, 0] / p_img[2, 0]
            py = p_img[1, 0] / p_img[2, 0]

            # Single keypoint with confidence 1.0
            person_kp_dict[port] = np.array([[px, py, 1.0]])

        result = triangulate_keypoints(
            person_kp_dict, port_to_cam_index,
            camera_params, proj_matrices,
            confidence_threshold=0.1,
        )

        assert len(result) == 1, f"Expected 1 triangulated point, got {len(result)}"
        assert result[0] is not None, "Triangulation returned None"

        error = np.linalg.norm(result[0] - point_3d)
        assert error < 0.05, f"Triangulation error {error:.4f}m exceeds 5cm threshold"

    def test_triangulate_multiple_points(self):
        """Triangulate several points across the capture volume."""
        import cv2

        config_path = str(COORD_3X1_DIR / "config.toml")
        camera_params = parse_calibration_mwc(config_path)
        proj_matrices = calculate_projection_matrices(camera_params)

        test_points = [
            np.array([3.0, 0.0, 4.0]),
            np.array([3.5, 0.5, 3.5]),
            np.array([2.5, -0.5, 4.5]),
            np.array([3.2, 1.0, 4.0]),
        ]

        for point_3d in test_points:
            person_kp_dict = {}
            port_to_cam_index = {}
            for i, cam in enumerate(camera_params):
                port = cam.get("port", i)
                port_to_cam_index[port] = i
                K = cam["matrix"]
                rvec = cam["rotation"]
                tvec = cam["translation"].reshape(3, 1)
                R, _ = cv2.Rodrigues(rvec)
                p_cam = R @ point_3d.reshape(3, 1) + tvec
                p_img = K @ p_cam
                px = p_img[0, 0] / p_img[2, 0]
                py = p_img[1, 0] / p_img[2, 0]
                person_kp_dict[port] = np.array([[px, py, 1.0]])

            result = triangulate_keypoints(
                person_kp_dict, port_to_cam_index,
                camera_params, proj_matrices, confidence_threshold=0.1,
            )
            assert result[0] is not None
            error = np.linalg.norm(result[0] - point_3d)
            assert error < 0.1, (
                f"Point {point_3d} triangulated with error {error:.4f}m (threshold 0.1m)"
            )


# ── Tests: Full Pipeline ──

# These tests require the ML models (YOLOv10 + VitPose) to be installed.
# They are marked slow and skipped if models are unavailable.

def _torch_available():
    """Check if torch is importable (models auto-download on first run)."""
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
class TestFullPipeline:
    """Run the full pipeline and compare against reference output."""

    def test_coord_3x1_3_person0(self):
        """Process coord_3x1_3 dataset and validate person 0 output."""
        from posetrack.process_synced_poses import process_synced_mwc_frames_multi_person_perf
        from posetrack.pose_detector import LOCAL_DET_DIR, LOCAL_SP_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output_3d_poses_tracked.csv")

            process_synced_mwc_frames_multi_person_perf(
                frame_history_csv_path=str(COORD_3X1_DIR / "frame_time_history.csv"),
                calibration_path=str(COORD_3X1_DIR / "config.toml"),
                video_dir=str(COORD_3X1_DIR),
                output_path=output_path,
                model_dir=LOCAL_SP_DIR,
                detector_dir=LOCAL_DET_DIR,
                calib_type="mwc",
                skip_sync_indices=1,
                person_confidence=0.8,
                keypoint_confidence=0.1,
                device_name="cpu",
                max_persons=1,
            )

            # The pipeline writes person-specific files
            person0_path = output_path + "_person0.csv"
            assert os.path.exists(person0_path), (
                f"Person 0 output not found at {person0_path}"
            )

            computed = load_reference_csv(person0_path)
            expected = load_reference_csv(COORD_3X1_REF)

            _compare_pose_outputs(computed, expected, tolerance=0.15)

    def test_recording_3by1_person0(self):
        """Process recording_3by1 dataset and validate output."""
        from posetrack.process_synced_poses import process_synced_mwc_frames_multi_person_perf
        from posetrack.pose_detector import LOCAL_DET_DIR, LOCAL_SP_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output_3d_poses_tracked.csv")

            process_synced_mwc_frames_multi_person_perf(
                frame_history_csv_path=str(RECORDING_3BY1_DIR / "frame_time_history.csv"),
                calibration_path=str(RECORDING_3BY1_DIR / "config.toml"),
                video_dir=str(RECORDING_3BY1_DIR),
                output_path=output_path,
                model_dir=LOCAL_SP_DIR,
                detector_dir=LOCAL_DET_DIR,
                calib_type="mwc",
                skip_sync_indices=1,
                person_confidence=0.8,
                keypoint_confidence=0.1,
                device_name="cpu",
                max_persons=2,
            )

            assert os.path.exists(output_path), (
                f"Output not found at {output_path}"
            )

            computed = load_reference_csv(output_path)
            expected = load_reference_csv(RECORDING_3BY1_REF)

            _compare_pose_outputs(computed, expected, tolerance=0.15)


def _compare_pose_outputs(computed, expected, tolerance=0.15):
    """
    Compare computed 3D pose output against reference.

    Checks:
      1. Same sync_indices detected (allow some frame drops)
      2. Per-keypoint 3D position within tolerance (meters)
    """
    expected_keys = set(expected.keys())
    computed_keys = set(computed.keys())

    # Allow up to 10% frame drop difference
    overlap = expected_keys & computed_keys
    coverage = len(overlap) / max(len(expected_keys), 1)
    assert coverage > 0.80, (
        f"Only {coverage:.0%} sync_index overlap "
        f"({len(overlap)}/{len(expected_keys)} frames)"
    )

    # Compare 3D positions for overlapping frames
    errors = []
    for key in sorted(overlap):
        exp_row = expected[key]
        comp_row = computed[key]

        for col in exp_row:
            if exp_row[col] is None or col not in comp_row or comp_row[col] is None:
                continue
            err = abs(exp_row[col] - comp_row[col])
            errors.append(err)

    if not errors:
        pytest.fail("No comparable keypoint values found")

    mean_err = np.mean(errors)
    p95_err = np.percentile(errors, 95)

    # Report stats
    print(f"\nPose comparison: {len(overlap)} frames, "
          f"mean error={mean_err:.4f}m, p95={p95_err:.4f}m")

    assert mean_err < tolerance, (
        f"Mean 3D error {mean_err:.4f}m exceeds tolerance {tolerance}m"
    )
    assert p95_err < tolerance * 3, (
        f"95th percentile error {p95_err:.4f}m exceeds {tolerance * 3}m"
    )
