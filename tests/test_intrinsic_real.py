"""
Tests for intrinsic calibration using real video data.

Test data: testdata/intrinsic/
  - port_A.mp4, port_B.mp4: calibration videos
  - board_config.toml: charuco board parameters
  - reference_intrinsics.pkl: saved reference results (generated on first run)

To regenerate reference data, delete reference_intrinsics.pkl and run:
    uv run pytest tests/test_intrinsic_real.py -v
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import rtoml

from calimerge.calibration.charuco import create_charuco_board
from calimerge.calibration.intrinsic import (
    calibrate_intrinsics,
    detect_charuco_points,
    filter_frames_for_calibration,
)
from calimerge.types import CharucoConfig

TESTDATA_DIR = Path(__file__).parent.parent / "testdata" / "intrinsic"
REFERENCE_PKL = TESTDATA_DIR / "reference_intrinsics.pkl"
VIDEOS = {"A": TESTDATA_DIR / "port_A.mp4", "B": TESTDATA_DIR / "port_B.mp4"}


@pytest.fixture(scope="module")
def board_config() -> CharucoConfig:
    """Load charuco config from testdata."""
    config_path = TESTDATA_DIR / "board_config.toml"
    data = rtoml.load(config_path)
    return CharucoConfig(
        columns=data["columns"],
        rows=data["rows"],
        square_size_cm=data["square_size_cm"],
        dictionary=data.get("dictionary", "DICT_4X4_50"),
        inverted=data.get("inverted", False),
    )


@pytest.fixture(scope="module")
def calibration_results(board_config):
    """Run calibration on both test videos and return results dict."""
    import cv2

    board = create_charuco_board(board_config)
    results = {}

    for label, video_path in VIDEOS.items():
        if not video_path.exists():
            pytest.skip(f"Test video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            pytest.skip(f"Cannot open video: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = (w, h)

        packets = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % 10 == 0:
                pkt = detect_charuco_points(frame, board_config, board)
                if pkt.point_id is not None and len(pkt.point_id) >= 4:
                    packets.append(pkt)
            idx += 1
        cap.release()

        assert len(packets) >= 10, (
            f"port_{label}: only {len(packets)} valid detections, need >= 10"
        )

        filtered = filter_frames_for_calibration(packets, target_count=40)
        intrinsics = calibrate_intrinsics(filtered, resolution, f"test_port_{label}")
        results[label] = intrinsics

    return results


class TestIntrinsicFromVideo:
    """Calibrate from real test videos and validate results."""

    def test_detection_count(self, calibration_results):
        """Should detect enough grids from both videos."""
        for label, intr in calibration_results.items():
            assert intr.grid_count >= 10, f"port_{label}: only {intr.grid_count} grids"

    def test_reprojection_error(self, calibration_results):
        """Reprojection error should be reasonable (< 1.0 pixel)."""
        for label, intr in calibration_results.items():
            assert intr.error < 1.0, (
                f"port_{label}: reprojection error {intr.error:.4f} too high"
            )

    def test_focal_length_reasonable(self, calibration_results):
        """Focal lengths should be positive and within expected range for 640x480."""
        for label, intr in calibration_results.items():
            fx = intr.matrix[0, 0]
            fy = intr.matrix[1, 1]
            # For a 640x480 webcam, focal length typically 400-1200 pixels
            assert 200 < fx < 2000, f"port_{label}: fx={fx:.1f} out of range"
            assert 200 < fy < 2000, f"port_{label}: fy={fy:.1f} out of range"
            # fx and fy should be close to each other (non-anamorphic lens)
            ratio = fx / fy
            assert 0.8 < ratio < 1.2, f"port_{label}: fx/fy={ratio:.3f} too skewed"

    def test_principal_point_near_center(self, calibration_results):
        """Principal point should be near image center."""
        for label, intr in calibration_results.items():
            w, h = intr.resolution
            cx = intr.matrix[0, 2]
            cy = intr.matrix[1, 2]
            assert abs(cx - w / 2) < w * 0.2, f"port_{label}: cx={cx:.1f} far from center"
            assert abs(cy - h / 2) < h * 0.2, f"port_{label}: cy={cy:.1f} far from center"

    def test_distortion_shape(self, calibration_results):
        """Distortion should have 5 coefficients."""
        for label, intr in calibration_results.items():
            assert intr.distortion.shape == (5,), (
                f"port_{label}: distortion shape {intr.distortion.shape}"
            )

    def test_matches_reference(self, calibration_results):
        """Results should closely match saved reference values.

        On first run (no reference file), this test saves the reference and passes.
        On subsequent runs, it compares against the saved reference.
        """
        if not REFERENCE_PKL.exists():
            # First run: save reference
            ref_data = {}
            for label, intr in calibration_results.items():
                ref_data[label] = {
                    "matrix": intr.matrix.copy(),
                    "distortion": intr.distortion.copy(),
                    "error": intr.error,
                    "grid_count": intr.grid_count,
                    "resolution": intr.resolution,
                }
            with open(REFERENCE_PKL, "wb") as f:
                pickle.dump(ref_data, f)
            pytest.skip("Reference data created; re-run to compare")

        # Load reference and compare
        with open(REFERENCE_PKL, "rb") as f:
            ref_data = pickle.load(f)

        for label, intr in calibration_results.items():
            ref = ref_data[label]

            # Matrix elements should match within 5% relative tolerance
            np.testing.assert_allclose(
                intr.matrix, ref["matrix"], rtol=0.05,
                err_msg=f"port_{label}: camera matrix drifted from reference",
            )

            # Distortion should match within absolute tolerance
            np.testing.assert_allclose(
                intr.distortion, ref["distortion"], atol=0.05,
                err_msg=f"port_{label}: distortion drifted from reference",
            )

            # Error should be similar
            assert abs(intr.error - ref["error"]) < 0.1, (
                f"port_{label}: error {intr.error:.4f} vs ref {ref['error']:.4f}"
            )
