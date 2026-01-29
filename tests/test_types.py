"""
Tests for calimerge.types dataclasses.
"""

import numpy as np
import pytest

from calimerge.types import (
    CameraConfig,
    CameraIntrinsics,
    CameraExtrinsics,
    CalibratedCamera,
    CharucoConfig,
    PointPacket,
    FramePoints,
    SyncedPoints,
    XYZPoints,
    ProjectConfig,
    compute_projection_matrix,
)


class TestCameraConfig:
    def test_creation_with_defaults(self):
        config = CameraConfig(serial_number="ABC123", port=0)
        assert config.serial_number == "ABC123"
        assert config.port == 0
        assert config.enabled is True
        assert config.resolution == (1280, 720)
        assert config.rotation_count == 0
        assert config.exposure == -4

    def test_creation_with_custom_values(self):
        config = CameraConfig(
            serial_number="XYZ789",
            port=2,
            enabled=False,
            resolution=(1920, 1080),
            rotation_count=1,
            exposure=-6,
        )
        assert config.serial_number == "XYZ789"
        assert config.port == 2
        assert config.enabled is False
        assert config.resolution == (1920, 1080)
        assert config.rotation_count == 1
        assert config.exposure == -6

    def test_frozen(self):
        config = CameraConfig(serial_number="ABC", port=0)
        with pytest.raises(AttributeError):
            config.port = 1


class TestCameraIntrinsics:
    def test_creation(self, sample_intrinsics_matrix, sample_distortion):
        intrinsics = CameraIntrinsics(
            serial_number="CAM001",
            resolution=(1280, 720),
            matrix=sample_intrinsics_matrix,
            distortion=sample_distortion,
            error=0.3,
            grid_count=25,
        )
        assert intrinsics.serial_number == "CAM001"
        assert intrinsics.resolution == (1280, 720)
        assert intrinsics.matrix.shape == (3, 3)
        assert intrinsics.distortion.shape == (5,)
        assert intrinsics.error == 0.3
        assert intrinsics.grid_count == 25

    def test_frozen(self, sample_camera_intrinsics):
        with pytest.raises(AttributeError):
            sample_camera_intrinsics.error = 0.5


class TestCameraExtrinsics:
    def test_identity(self):
        ext = CameraExtrinsics(
            rotation=np.eye(3),
            translation=np.zeros(3),
        )
        np.testing.assert_array_equal(ext.rotation, np.eye(3))
        np.testing.assert_array_equal(ext.translation, np.zeros(3))

    def test_with_transform(self):
        # 90 degree rotation around Z
        R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        t = np.array([1.0, 2.0, 3.0])

        ext = CameraExtrinsics(rotation=R, translation=t)
        np.testing.assert_array_almost_equal(ext.rotation, R)
        np.testing.assert_array_almost_equal(ext.translation, t)


class TestCalibratedCamera:
    def test_creation(self, sample_calibrated_camera):
        cam = sample_calibrated_camera
        assert cam.serial_number == "TEST123"
        assert cam.port == 0
        assert cam.intrinsics is not None
        assert cam.extrinsics is not None


class TestCharucoConfig:
    def test_defaults(self):
        config = CharucoConfig(
            columns=4,
            rows=5,
            square_size_cm=5.0,  # 5cm
        )
        assert config.dictionary == "DICT_4X4_50"
        assert config.inverted is False
        assert config.legacy_pattern is False

    def test_square_size_conversion(self):
        """square_size_cm should convert to meters."""
        config = CharucoConfig(columns=4, rows=5, square_size_cm=4.0)
        assert config.square_size_m == 0.04  # 4cm = 0.04m

    def test_marker_size_auto_computed(self):
        """Marker size should be 75% of square size."""
        config = CharucoConfig(columns=4, rows=5, square_size_cm=4.0)
        assert config.marker_size_m == 0.03  # 75% of 0.04m

    def test_custom(self):
        config = CharucoConfig(
            columns=7,
            rows=9,
            square_size_cm=3.0,
            dictionary="DICT_5X5_100",
            inverted=True,
            legacy_pattern=True,
        )
        assert config.columns == 7
        assert config.rows == 9
        assert config.inverted is True
        assert config.square_size_m == 0.03
        assert config.marker_size_m == 0.0225  # 75% of 0.03


class TestPointPacket:
    def test_empty(self):
        packet = PointPacket()
        assert packet.point_id is None
        assert packet.img_loc is None

    def test_with_data(self):
        ids = np.array([0, 1, 2, 3])
        locs = np.array([[100, 200], [150, 250], [200, 300], [250, 350]], dtype=np.float64)

        packet = PointPacket(point_id=ids, img_loc=locs)
        np.testing.assert_array_equal(packet.point_id, ids)
        np.testing.assert_array_equal(packet.img_loc, locs)

    def test_with_object_points(self):
        ids = np.array([0, 1])
        img = np.array([[100, 200], [150, 250]], dtype=np.float64)
        obj = np.array([[0, 0, 0], [0.04, 0, 0]], dtype=np.float64)

        packet = PointPacket(point_id=ids, img_loc=img, obj_loc=obj)
        assert packet.obj_loc is not None
        assert packet.obj_loc.shape == (2, 3)


class TestXYZPoints:
    def test_creation(self):
        xyz = XYZPoints(
            sync_index=42,
            point_ids=np.array([0, 1, 2]),
            xyz=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        )
        assert xyz.sync_index == 42
        assert len(xyz.point_ids) == 3
        assert xyz.xyz.shape == (3, 3)


class TestComputeProjectionMatrix:
    def test_identity_extrinsics(self, sample_calibrated_camera):
        P = compute_projection_matrix(sample_calibrated_camera)
        assert P.shape == (3, 4)

        # With identity rotation and zero translation,
        # P should be [K | 0] essentially
        K = sample_calibrated_camera.intrinsics.matrix
        expected_left = K
        np.testing.assert_array_almost_equal(P[:, :3], expected_left)

    def test_with_translation(self, sample_camera_intrinsics):
        ext = CameraExtrinsics(
            rotation=np.eye(3),
            translation=np.array([1.0, 0.0, 0.0]),
        )
        cam = CalibratedCamera(
            serial_number="TEST",
            port=0,
            intrinsics=sample_camera_intrinsics,
            extrinsics=ext,
        )

        P = compute_projection_matrix(cam)
        assert P.shape == (3, 4)
        # The last column should reflect the translation
        assert P[0, 3] != 0  # Translation affects projection


class TestProjectConfig:
    def test_defaults(self, sample_charuco_config):
        config = ProjectConfig(
            fps=30,
            cameras={},
            charuco_intrinsic=sample_charuco_config,
            charuco_extrinsic=sample_charuco_config,
        )
        assert config.fps == 30
        assert config.pose_backend == "charuco"
        assert config.pose_device == "cpu"
        assert config.max_persons == 1

    def test_separate_charuco_configs(self):
        """Intrinsic and extrinsic can have different charuco boards."""
        intrinsic_charuco = CharucoConfig(columns=7, rows=5, square_size_cm=3.0)
        extrinsic_charuco = CharucoConfig(columns=4, rows=3, square_size_cm=5.0)

        config = ProjectConfig(
            fps=30,
            cameras={},
            charuco_intrinsic=intrinsic_charuco,
            charuco_extrinsic=extrinsic_charuco,
        )
        assert config.charuco_intrinsic.square_size_cm == 3.0
        assert config.charuco_extrinsic.square_size_cm == 5.0

    def test_with_cameras(self, sample_charuco_config):
        cameras = {
            "ABC": CameraConfig(serial_number="ABC", port=0),
            "DEF": CameraConfig(serial_number="DEF", port=1),
        }
        config = ProjectConfig(
            fps=24,
            cameras=cameras,
            charuco_intrinsic=sample_charuco_config,
            charuco_extrinsic=sample_charuco_config,
        )
        assert len(config.cameras) == 2
        assert "ABC" in config.cameras
