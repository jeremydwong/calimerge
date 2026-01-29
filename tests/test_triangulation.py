"""
Tests for calimerge.triangulation.
"""

import numpy as np
import pytest

from calimerge.triangulation import (
    undistort_points,
    triangulate_points,
    triangulate_frame,
)
from calimerge.types import (
    CameraIntrinsics,
    CameraExtrinsics,
    CalibratedCamera,
    PointPacket,
    FramePoints,
    SyncedPoints,
)


class TestUndistortPoints:
    def test_no_distortion(self, sample_camera_intrinsics):
        """Points should be nearly unchanged with zero distortion."""
        # Create intrinsics with zero distortion
        intrinsics = CameraIntrinsics(
            serial_number="TEST",
            resolution=(1280, 720),
            matrix=sample_camera_intrinsics.matrix,
            distortion=np.zeros(5, dtype=np.float64),
            error=0.1,
            grid_count=10,
        )

        points = np.array([[640, 360], [320, 180], [960, 540]], dtype=np.float64)
        undistorted = undistort_points(points, intrinsics)

        # With zero distortion, should be very close to original
        np.testing.assert_array_almost_equal(undistorted, points, decimal=3)

    def test_output_shape(self, sample_camera_intrinsics):
        """Output should have same shape as input."""
        points = np.array([[100, 200], [300, 400], [500, 600]], dtype=np.float64)
        undistorted = undistort_points(points, sample_camera_intrinsics)

        assert undistorted.shape == points.shape

    def test_empty_input(self, sample_camera_intrinsics):
        """Should handle empty input array."""
        points = np.array([], dtype=np.float64).reshape(0, 2)
        undistorted = undistort_points(points, sample_camera_intrinsics)

        assert undistorted.shape == (0, 2)

    def test_principal_point_unchanged(self, sample_camera_intrinsics):
        """Principal point should be unchanged by undistortion."""
        cx = sample_camera_intrinsics.matrix[0, 2]
        cy = sample_camera_intrinsics.matrix[1, 2]

        points = np.array([[cx, cy]], dtype=np.float64)
        undistorted = undistort_points(points, sample_camera_intrinsics)

        # Principal point maps to itself
        np.testing.assert_array_almost_equal(undistorted[0], [cx, cy], decimal=3)


class TestTriangulatePoints:
    @pytest.fixture
    def stereo_cameras(self, sample_camera_intrinsics):
        """Create a simple stereo camera setup."""
        # Camera 0: at origin, looking along Z
        cam0 = CalibratedCamera(
            serial_number="CAM0",
            port=0,
            intrinsics=sample_camera_intrinsics,
            extrinsics=CameraExtrinsics(
                rotation=np.eye(3),
                translation=np.zeros(3),
            ),
        )

        # Camera 1: translated 0.5m to the right
        cam1 = CalibratedCamera(
            serial_number="CAM1",
            port=1,
            intrinsics=sample_camera_intrinsics,
            extrinsics=CameraExtrinsics(
                rotation=np.eye(3),
                translation=np.array([0.5, 0.0, 0.0]),
            ),
        )

        return {0: cam0, 1: cam1}

    def test_triangulate_single_point(self, stereo_cameras):
        """Should triangulate a point from two camera observations."""
        # Get intrinsics
        K = stereo_cameras[0].intrinsics.matrix
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Create two 2D observations that would correspond to a 3D point
        # Camera 0 at origin sees point at image center
        p0 = np.array([cx, cy])

        # Camera 1 is 0.5m to the right, so it sees the point shifted left
        # If point is at depth Z and centered for cam0, cam1 sees it at:
        # x' = fx * (0 - 0.5) / Z + cx
        # For a point at Z=2: x' = 800 * (-0.5) / 2 + 640 = -200 + 640 = 440
        p1 = np.array([cx - fx * 0.5 / 2.0, cy])

        # Triangulate
        camera_indices = np.array([0, 1], dtype=np.int32)
        img_points = np.array([p0, p1], dtype=np.float64)

        result = triangulate_points(stereo_cameras, camera_indices, img_points)

        assert result is not None
        assert result.shape == (3,)

        # The result should have:
        # - X near 0 (point is on optical axis of camera 0)
        # - Y near 0 (point is at same height)
        # - Z should be non-zero (point is in front of cameras)
        assert abs(result[0]) < 0.5  # X near center
        assert abs(result[1]) < 0.5  # Y near center
        assert abs(result[2]) > 0.5  # Z is significant (positive or negative depending on convention)

    def test_triangulate_requires_two_cameras(self, stereo_cameras):
        """Should return None with fewer than 2 observations."""
        camera_indices = np.array([0], dtype=np.int32)
        img_points = np.array([[640, 360]], dtype=np.float64)

        result = triangulate_points(stereo_cameras, camera_indices, img_points)
        assert result is None


class TestTriangulateFrame:
    @pytest.fixture
    def stereo_cameras(self, sample_camera_intrinsics):
        """Create stereo camera setup."""
        cam0 = CalibratedCamera(
            serial_number="CAM0",
            port=0,
            intrinsics=sample_camera_intrinsics,
            extrinsics=CameraExtrinsics(
                rotation=np.eye(3),
                translation=np.zeros(3),
            ),
        )
        cam1 = CalibratedCamera(
            serial_number="CAM1",
            port=1,
            intrinsics=sample_camera_intrinsics,
            extrinsics=CameraExtrinsics(
                rotation=np.eye(3),
                translation=np.array([0.5, 0.0, 0.0]),
            ),
        )
        return {0: cam0, 1: cam1}

    def test_triangulate_empty_frame(self, stereo_cameras):
        """Should handle frame with no points."""
        synced = SyncedPoints(
            sync_index=0,
            frame_points={
                0: None,
                1: None,
            },
        )

        result = triangulate_frame(stereo_cameras, synced)

        assert result is not None
        assert result.sync_index == 0
        assert len(result.point_ids) == 0
        assert result.xyz.shape == (0, 3)

    def test_triangulate_frame_with_shared_points(self, stereo_cameras):
        """Should triangulate points seen by both cameras."""
        # Create point packets with shared point IDs
        K = stereo_cameras[0].intrinsics.matrix
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 3D points
        P0 = np.array([0, 0, 2])
        P1 = np.array([0.2, 0.1, 2.5])

        # Project to both cameras
        def project(P, cam):
            t = cam.extrinsics.translation
            P_cam = P - t
            return np.array([fx * P_cam[0] / P_cam[2] + cx, fy * P_cam[1] / P_cam[2] + cy])

        cam0_pts = np.array([project(P0, stereo_cameras[0]), project(P1, stereo_cameras[0])])
        cam1_pts = np.array([project(P0, stereo_cameras[1]), project(P1, stereo_cameras[1])])

        synced = SyncedPoints(
            sync_index=42,
            frame_points={
                0: FramePoints(
                    port=0,
                    frame_index=100,
                    points=PointPacket(
                        point_id=np.array([0, 1]),
                        img_loc=cam0_pts,
                    ),
                ),
                1: FramePoints(
                    port=1,
                    frame_index=100,
                    points=PointPacket(
                        point_id=np.array([0, 1]),
                        img_loc=cam1_pts,
                    ),
                ),
            },
        )

        result = triangulate_frame(stereo_cameras, synced)

        assert result.sync_index == 42
        assert len(result.point_ids) == 2
        assert result.xyz.shape == (2, 3)

    def test_single_camera_points_not_triangulated(self, stereo_cameras):
        """Points seen by only one camera should not be triangulated."""
        synced = SyncedPoints(
            sync_index=0,
            frame_points={
                0: FramePoints(
                    port=0,
                    frame_index=0,
                    points=PointPacket(
                        point_id=np.array([0, 1]),  # Points 0, 1 only in cam 0
                        img_loc=np.array([[100, 100], [200, 200]], dtype=np.float64),
                    ),
                ),
                1: FramePoints(
                    port=1,
                    frame_index=0,
                    points=PointPacket(
                        point_id=np.array([2, 3]),  # Points 2, 3 only in cam 1
                        img_loc=np.array([[150, 150], [250, 250]], dtype=np.float64),
                    ),
                ),
            },
        )

        result = triangulate_frame(stereo_cameras, synced)

        # No points are shared, so nothing should be triangulated
        assert len(result.point_ids) == 0
