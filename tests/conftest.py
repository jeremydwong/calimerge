"""
Pytest configuration and shared fixtures.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after test."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_intrinsics_matrix():
    """Typical camera intrinsics matrix."""
    return np.array([
        [800.0, 0.0, 640.0],
        [0.0, 800.0, 360.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


@pytest.fixture
def sample_distortion():
    """Typical distortion coefficients (k1, k2, p1, p2, k3)."""
    return np.array([0.1, -0.25, 0.001, -0.001, 0.1], dtype=np.float64)


@pytest.fixture
def sample_charuco_config():
    """Standard ChArUco board configuration."""
    from calimerge.types import CharucoConfig
    return CharucoConfig(
        columns=7,
        rows=5,
        square_size_cm=4.0,  # 4cm squares, marker is auto 75% = 3cm
        dictionary="DICT_4X4_50",
    )


@pytest.fixture
def sample_camera_intrinsics(sample_intrinsics_matrix, sample_distortion):
    """Sample CameraIntrinsics dataclass."""
    from calimerge.types import CameraIntrinsics
    return CameraIntrinsics(
        serial_number="TEST123",
        resolution=(1280, 720),
        matrix=sample_intrinsics_matrix,
        distortion=sample_distortion,
        error=0.25,
        grid_count=50,
    )


@pytest.fixture
def sample_extrinsics():
    """Sample CameraExtrinsics (identity transform)."""
    from calimerge.types import CameraExtrinsics
    return CameraExtrinsics(
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


@pytest.fixture
def sample_calibrated_camera(sample_camera_intrinsics, sample_extrinsics):
    """Sample CalibratedCamera."""
    from calimerge.types import CalibratedCamera
    return CalibratedCamera(
        serial_number="TEST123",
        port=0,
        intrinsics=sample_camera_intrinsics,
        extrinsics=sample_extrinsics,
    )
