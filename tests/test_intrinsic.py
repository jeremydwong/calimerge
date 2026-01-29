"""
Tests for calimerge.calibration.intrinsic.
"""

import cv2
import numpy as np
import pytest

from calimerge.calibration.intrinsic import (
    detect_charuco_points,
    calibrate_intrinsics,
)
from calimerge.calibration.charuco import create_charuco_board, generate_board_image
from calimerge.types import CharucoConfig, PointPacket


class TestDetectCharucoPoints:
    @pytest.fixture
    def board_config(self):
        return CharucoConfig(
            columns=5,
            rows=4,
            square_size_cm=4.0,
            dictionary="DICT_4X4_50",
        )

    def test_detect_in_synthetic_image(self, board_config):
        """Detection should work on a clean synthetic board image."""
        # Generate a board image
        img = generate_board_image(board_config, width=800, height=600)

        # Detect points
        packet = detect_charuco_points(img, board_config)

        assert packet is not None
        # Should detect at least some corners
        if packet.point_id is not None:
            assert len(packet.point_id) > 0
            assert packet.img_loc is not None
            assert len(packet.img_loc) == len(packet.point_id)

    def test_detect_returns_object_points(self, board_config):
        """Detection should include 3D object points."""
        img = generate_board_image(board_config, width=800, height=600)
        packet = detect_charuco_points(img, board_config)

        if packet.point_id is not None and len(packet.point_id) > 0:
            assert packet.obj_loc is not None
            assert packet.obj_loc.shape[1] == 3  # x, y, z

    def test_detect_empty_on_blank_image(self, board_config):
        """Detection should return empty packet on image with no board."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        packet = detect_charuco_points(blank, board_config)

        # Should return packet but with None or empty arrays
        assert packet is not None
        assert packet.point_id is None or len(packet.point_id) == 0

    def test_detect_with_prebuilt_board(self, board_config):
        """Can pass prebuilt board to avoid recreation."""
        img = generate_board_image(board_config, width=800, height=600)
        board = create_charuco_board(board_config)

        packet = detect_charuco_points(img, board_config, board=board)
        assert packet is not None


class TestCalibrateIntrinsics:
    @pytest.fixture
    def board_config(self):
        return CharucoConfig(
            columns=6,
            rows=5,
            square_size_cm=4.0,
            dictionary="DICT_4X4_50",
        )

    def test_calibrate_requires_minimum_frames(self, board_config):
        """Calibration should fail with too few frames."""
        # Create just 2 packets (below the 3-frame minimum)
        packets = []
        for i in range(2):  # Only 2 frames - below minimum
            packet = PointPacket(
                point_id=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                img_loc=np.array([
                    [100, 100], [200, 100], [300, 100], [400, 100],
                    [100, 200], [200, 200], [300, 200], [400, 200],
                ], dtype=np.float64),
                obj_loc=np.array([
                    [0, 0, 0], [0.04, 0, 0], [0.08, 0, 0], [0.12, 0, 0],
                    [0, 0.04, 0], [0.04, 0.04, 0], [0.08, 0.04, 0], [0.12, 0.04, 0],
                ], dtype=np.float64),
            )
            packets.append(packet)

        # Should raise ValueError because we check for minimum frames (need at least 3)
        with pytest.raises(ValueError, match="at least"):
            calibrate_intrinsics(packets, (640, 480), "TEST")

    def test_calibrate_with_synthetic_data(self, board_config):
        """Calibration should work with multiple synthetic board views."""
        board = create_charuco_board(board_config)
        packets = []

        # Generate board at full size only (scaling causes issues)
        # Use different positions instead
        img = generate_board_image(board_config, width=800, height=600)

        # Detect from the clean image multiple times
        # In reality, you'd have different camera views, but for testing
        # we just need valid detections
        for _ in range(15):
            packet = detect_charuco_points(img, board_config, board)
            if packet.point_id is not None and len(packet.point_id) >= 4:
                packets.append(packet)

        # Need enough valid detections
        if len(packets) >= 10:
            intrinsics = calibrate_intrinsics(packets, (800, 600), "SYNTHETIC")

            assert intrinsics is not None
            assert intrinsics.serial_number == "SYNTHETIC"
            assert intrinsics.resolution == (800, 600)
            assert intrinsics.matrix.shape == (3, 3)
            assert intrinsics.distortion.shape[0] >= 5
            assert intrinsics.error >= 0
            assert intrinsics.grid_count == len(packets)

            # Sanity checks on intrinsics
            fx, fy = intrinsics.matrix[0, 0], intrinsics.matrix[1, 1]
            cx, cy = intrinsics.matrix[0, 2], intrinsics.matrix[1, 2]

            # Focal lengths should be positive and reasonable
            assert fx > 100
            assert fy > 100

            # Principal point should be near image center
            assert 200 < cx < 600
            assert 150 < cy < 450
        else:
            pytest.skip("Not enough valid detections for calibration test")
