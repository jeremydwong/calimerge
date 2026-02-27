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

    If scaled from a different resolution, scaled_from contains the original resolution.
    """

    serial_number: str
    resolution: tuple[int, int]  # (width, height) - current/target resolution
    matrix: np.ndarray  # 3x3 camera matrix
    distortion: np.ndarray  # Distortion coefficients (5,)
    error: float  # RMSE of reprojection (at original resolution)
    grid_count: int  # Number of grids used in calibration
    scaled_from: tuple[int, int] | None = None  # Original resolution if scaled

    @property
    def is_scaled(self) -> bool:
        """True if these intrinsics were scaled from a different resolution."""
        return self.scaled_from is not None


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


# ============================================================================
# Intrinsics Scaling
# ============================================================================


def get_aspect_ratio(resolution: tuple[int, int]) -> tuple[int, int]:
    """
    Get simplified aspect ratio for a resolution.

    Args:
        resolution: (width, height) tuple

    Returns:
        Simplified (w, h) ratio, e.g. (4, 3) or (16, 9)
    """
    from math import gcd

    w, h = resolution
    divisor = gcd(w, h)
    return (w // divisor, h // divisor)


def same_aspect_ratio(res1: tuple[int, int], res2: tuple[int, int]) -> bool:
    """Check if two resolutions have the same aspect ratio."""
    return get_aspect_ratio(res1) == get_aspect_ratio(res2)


def scale_intrinsics(
    intrinsics: CameraIntrinsics,
    new_resolution: tuple[int, int],
) -> CameraIntrinsics:
    """
    Scale intrinsics to a different resolution.

    Camera matrix parameters (fx, fy, cx, cy) scale linearly with resolution.
    Distortion coefficients are dimensionless and unchanged.

    Args:
        intrinsics: Original intrinsics
        new_resolution: Target (width, height)

    Returns:
        New CameraIntrinsics scaled to the target resolution

    Raises:
        ValueError: If aspect ratios don't match
    """
    old_w, old_h = intrinsics.resolution
    new_w, new_h = new_resolution

    # Verify aspect ratios match
    if not same_aspect_ratio(intrinsics.resolution, new_resolution):
        old_ar = get_aspect_ratio(intrinsics.resolution)
        new_ar = get_aspect_ratio(new_resolution)
        raise ValueError(
            f"Cannot scale intrinsics: aspect ratio mismatch "
            f"({old_ar[0]}:{old_ar[1]} -> {new_ar[0]}:{new_ar[1]})"
        )

    # Compute scale factors
    scale_x = new_w / old_w
    scale_y = new_h / old_h

    # Scale camera matrix
    # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    new_matrix = intrinsics.matrix.copy()
    new_matrix[0, 0] *= scale_x  # fx
    new_matrix[1, 1] *= scale_y  # fy
    new_matrix[0, 2] *= scale_x  # cx
    new_matrix[1, 2] *= scale_y  # cy
    # [2, 2] stays 1.0

    # Note: The original reprojection error was computed at the original
    # resolution. At higher resolutions the effective error in pixels is larger,
    # but we keep the original value since it represents calibration quality.

    # Track the original resolution for UI display
    # If already scaled, preserve the original source
    original_source = intrinsics.scaled_from or intrinsics.resolution

    return CameraIntrinsics(
        serial_number=intrinsics.serial_number,
        resolution=new_resolution,
        matrix=new_matrix,
        distortion=intrinsics.distortion,  # Unchanged - dimensionless
        error=intrinsics.error,  # Keep original quality metric
        grid_count=intrinsics.grid_count,
        scaled_from=original_source,
    )
