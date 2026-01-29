"""
Core data structures for calimerge.

All types are frozen dataclasses with slots for immutability and performance.
Logic is in separate pure functions - these are data containers only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# ============================================================================
# Camera Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """
    Configuration for a single camera.
    Corresponds to TOML [cameras.SERIAL] section.
    """

    serial_number: str
    port: int
    enabled: bool = True
    resolution: tuple[int, int] = (1280, 720)  # (width, height)
    rotation_count: int = 0  # 0, 1, 2, 3 for 0, 90, 180, 270 degrees
    exposure: int = -4  # Platform-specific units


# ============================================================================
# Calibration Data
# ============================================================================


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """
    Intrinsic parameters for a camera.
    Stored in SQLite database keyed by (serial_number, resolution).
    """

    serial_number: str
    resolution: tuple[int, int]  # (width, height)
    matrix: np.ndarray  # 3x3 camera matrix
    distortion: np.ndarray  # Distortion coefficients (5,)
    error: float  # RMSE of reprojection
    grid_count: int  # Number of grids used in calibration


@dataclass(frozen=True, slots=True)
class CameraExtrinsics:
    """
    Extrinsic parameters for a camera relative to world origin.
    """

    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # (3,) translation vector


@dataclass(frozen=True, slots=True)
class CalibratedCamera:
    """
    Complete calibration for a camera (intrinsics + extrinsics).
    """

    serial_number: str
    port: int
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics


# ============================================================================
# ChArUco Board Configuration
# ============================================================================


@dataclass(frozen=True)  # No slots - need properties
class CharucoConfig:
    """
    Configuration for ChArUco board detection.

    Specify square_size_cm for convenience - meters computed automatically.
    Marker size is always 75% of square size (standard ratio).
    """

    columns: int
    rows: int
    square_size_cm: float  # Square edge length in centimeters
    dictionary: str = "DICT_4X4_50"
    inverted: bool = False
    legacy_pattern: bool = False

    @property
    def square_size_m(self) -> float:
        """Square size in meters (computed from cm)."""
        return self.square_size_cm / 100.0

    @property
    def marker_size_m(self) -> float:
        """Marker size in meters (75% of square size)."""
        return self.square_size_m * 0.75


# ============================================================================
# Point Data
# ============================================================================


@dataclass(frozen=True, slots=True)
class PointPacket:
    """
    2D points detected in a single frame.

    This is the primary return value of trackers.
    obj_loc is only populated for calibration (ChArUco) tracking.
    """

    point_id: np.ndarray | None = None  # (n,) unique point identifiers
    img_loc: np.ndarray | None = None  # (n, 2) image coordinates (x, y)
    obj_loc: np.ndarray | None = None  # (n, 3) object-space coords (for calibration)
    confidence: np.ndarray | None = None  # (n,) confidence scores


@dataclass(frozen=True, slots=True)
class FramePoints:
    """
    Points from a single camera frame with metadata.
    """

    port: int
    frame_index: int
    points: PointPacket
    timestamp_ns: int = 0


@dataclass(frozen=True, slots=True)
class SyncedPoints:
    """
    Points from all cameras at a single sync index.
    """

    sync_index: int
    frame_points: dict[int, FramePoints | None]  # port -> FramePoints


# ============================================================================
# 3D Points
# ============================================================================


@dataclass(frozen=True, slots=True)
class XYZPoints:
    """
    Triangulated 3D points for a single sync index.
    """

    sync_index: int
    point_ids: np.ndarray  # (n,)
    xyz: np.ndarray  # (n, 3)


# ============================================================================
# Project Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """
    Complete project configuration.
    Loaded from TOML file in project directory.

    Separate charuco configs for intrinsic (single camera, can be smaller)
    and extrinsic (multi-camera, typically larger for visibility).
    """

    fps: int
    cameras: dict[str, CameraConfig]  # serial_number -> config
    charuco_intrinsic: CharucoConfig  # For per-camera intrinsic calibration
    charuco_extrinsic: CharucoConfig  # For multi-camera extrinsic calibration
    pose_backend: Literal["charuco", "mediapipe", "vitpose"] = "charuco"
    pose_device: str = "cpu"  # "cpu", "cuda", "mps"
    max_persons: int = 1


# ============================================================================
# Pure functions for computed properties
# ============================================================================


def compute_transformation_matrix(extrinsics: CameraExtrinsics) -> np.ndarray:
    """
    Compute 4x4 homogeneous transformation matrix from extrinsics.
    """
    t = np.eye(4, dtype=np.float64)
    t[0:3, 0:3] = extrinsics.rotation
    t[0:3, 3] = extrinsics.translation
    return t


def compute_projection_matrix(camera: CalibratedCamera) -> np.ndarray:
    """
    Compute 3x4 projection matrix from calibrated camera.
    """
    t = compute_transformation_matrix(camera.extrinsics)
    return camera.intrinsics.matrix @ t[0:3, :]


def extrinsics_to_vector(extrinsics: CameraExtrinsics) -> np.ndarray:
    """
    Convert extrinsics to 6-element vector for bundle adjustment.
    [rodrigues_x, rodrigues_y, rodrigues_z, tx, ty, tz]
    """
    import cv2

    rodrigues = cv2.Rodrigues(extrinsics.rotation)[0][:, 0]
    return np.hstack([rodrigues, extrinsics.translation])


def extrinsics_from_vector(vector: np.ndarray) -> CameraExtrinsics:
    """
    Create extrinsics from 6-element vector.
    """
    import cv2

    rotation = cv2.Rodrigues(vector[0:3])[0]
    translation = vector[3:6].astype(np.float64)
    return CameraExtrinsics(rotation=rotation, translation=translation)


def get_projection_matrices(
    cameras: dict[int, CalibratedCamera],
) -> dict[int, np.ndarray]:
    """
    Build dict of projection matrices for triangulation.
    """
    return {port: compute_projection_matrix(cam) for port, cam in cameras.items()}
