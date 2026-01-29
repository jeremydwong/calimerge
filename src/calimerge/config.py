"""
Configuration loading/saving.

Pure functions operating on dataclasses.
- TOML for project configuration
- SQLite for camera intrinsics (keyed by serial_number + resolution)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import rtoml

from .types import (
    CameraConfig,
    CameraExtrinsics,
    CameraIntrinsics,
    CalibratedCamera,
    CharucoConfig,
    ProjectConfig,
)


# ============================================================================
# TOML Project Configuration
# ============================================================================


def load_project_config(path: Path) -> ProjectConfig:
    """
    Load project configuration from TOML file.

    Args:
        path: Path to config.toml file

    Returns:
        ProjectConfig dataclass
    """
    data = rtoml.load(path)

    # Parse cameras
    cameras = {}
    for key, cam_data in data.items():
        if key.startswith("cameras.") or (
            isinstance(cam_data, dict) and "serial_number" in cam_data
        ):
            serial = cam_data.get("serial_number", key.replace("cameras.", ""))
            cameras[serial] = CameraConfig(
                serial_number=serial,
                port=cam_data.get("port", 0),
                enabled=cam_data.get("enabled", True),
                resolution=tuple(cam_data.get("resolution", [1280, 720])),
                rotation_count=cam_data.get("rotation", 0),
                exposure=cam_data.get("exposure", -4),
            )

    # Also check for [cameras] section (nested format)
    if "cameras" in data and isinstance(data["cameras"], dict):
        for serial, cam_data in data["cameras"].items():
            cameras[serial] = CameraConfig(
                serial_number=serial,
                port=cam_data.get("port", 0),
                enabled=cam_data.get("enabled", True),
                resolution=tuple(cam_data.get("resolution", [1280, 720])),
                rotation_count=cam_data.get("rotation", 0),
                exposure=cam_data.get("exposure", -4),
            )

    # Parse charuco configs (intrinsic and extrinsic)
    def parse_charuco(section_data: dict) -> CharucoConfig:
        return CharucoConfig(
            columns=section_data.get("columns", 4),
            rows=section_data.get("rows", 5),
            square_size_cm=section_data.get("square_size_cm", 4.0),
            dictionary=section_data.get("dictionary", "DICT_4X4_50"),
            inverted=section_data.get("inverted", False),
            legacy_pattern=section_data.get("legacy_pattern", False),
        )

    # Intrinsic charuco (smaller board, closer to camera)
    intrinsic_data = data.get("charuco_intrinsic", data.get("charuco", {}))
    charuco_intrinsic = parse_charuco(intrinsic_data)

    # Extrinsic charuco (larger board, visible from multiple cameras)
    extrinsic_data = data.get("charuco_extrinsic", data.get("charuco", {}))
    charuco_extrinsic = parse_charuco(extrinsic_data)

    # Parse pose settings
    pose_data = data.get("pose", {})

    return ProjectConfig(
        fps=data.get("fps", 30),
        cameras=cameras,
        charuco_intrinsic=charuco_intrinsic,
        charuco_extrinsic=charuco_extrinsic,
        pose_backend=pose_data.get("backend", "charuco"),
        pose_device=pose_data.get("device", "cpu"),
        max_persons=pose_data.get("max_persons", 1),
    )


def save_project_config(config: ProjectConfig, path: Path) -> None:
    """
    Save project configuration to TOML file.

    Args:
        config: ProjectConfig dataclass
        path: Path to save config.toml
    """
    def charuco_to_dict(charuco: CharucoConfig) -> dict:
        return {
            "columns": charuco.columns,
            "rows": charuco.rows,
            "square_size_cm": charuco.square_size_cm,
            "dictionary": charuco.dictionary,
            "inverted": charuco.inverted,
            "legacy_pattern": charuco.legacy_pattern,
        }

    data = {
        "fps": config.fps,
        "cameras": {},
        "charuco_intrinsic": charuco_to_dict(config.charuco_intrinsic),
        "charuco_extrinsic": charuco_to_dict(config.charuco_extrinsic),
        "pose": {
            "backend": config.pose_backend,
            "device": config.pose_device,
            "max_persons": config.max_persons,
        },
    }

    for serial, cam in config.cameras.items():
        data["cameras"][serial] = {
            "port": cam.port,
            "enabled": cam.enabled,
            "resolution": list(cam.resolution),
            "rotation": cam.rotation_count,
            "exposure": cam.exposure,
        }

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        rtoml.dump(data, f)


def create_default_project_config(
    cameras: dict[str, CameraConfig] | None = None,
) -> ProjectConfig:
    """
    Create a default project configuration.

    Args:
        cameras: Optional dict of camera configs

    Returns:
        ProjectConfig with sensible defaults
    """
    # Intrinsic: smaller board for close-up single camera calibration
    charuco_intrinsic = CharucoConfig(
        columns=7,
        rows=5,
        square_size_cm=3.0,  # 3cm squares
    )

    # Extrinsic: larger board for visibility from multiple cameras
    charuco_extrinsic = CharucoConfig(
        columns=4,
        rows=3,
        square_size_cm=5.0,  # 5cm squares
    )

    return ProjectConfig(
        fps=30,
        cameras=cameras or {},
        charuco_intrinsic=charuco_intrinsic,
        charuco_extrinsic=charuco_extrinsic,
    )


# ============================================================================
# SQLite Intrinsics Database
# ============================================================================

DEFAULT_INTRINSICS_DB = Path.home() / ".calimerge" / "intrinsics.db"


def get_default_intrinsics_db() -> Path:
    """Get the default intrinsics database path."""
    return DEFAULT_INTRINSICS_DB


def init_intrinsics_db(db_path: Path = DEFAULT_INTRINSICS_DB) -> None:
    """
    Initialize the intrinsics database if it doesn't exist.

    Args:
        db_path: Path to SQLite database file
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS intrinsics (
            serial_number TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            matrix BLOB NOT NULL,
            distortion BLOB NOT NULL,
            error REAL NOT NULL,
            grid_count INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (serial_number, width, height)
        )
    """)
    conn.commit()
    conn.close()


def save_intrinsics(
    intrinsics: CameraIntrinsics,
    db_path: Path = DEFAULT_INTRINSICS_DB,
) -> None:
    """
    Save camera intrinsics to database.

    Uses INSERT OR REPLACE to update existing entries.

    Args:
        intrinsics: CameraIntrinsics dataclass
        db_path: Path to SQLite database file
    """
    init_intrinsics_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR REPLACE INTO intrinsics
        (serial_number, width, height, matrix, distortion, error, grid_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            intrinsics.serial_number,
            intrinsics.resolution[0],
            intrinsics.resolution[1],
            intrinsics.matrix.astype(np.float64).tobytes(),
            intrinsics.distortion.astype(np.float64).tobytes(),
            intrinsics.error,
            intrinsics.grid_count,
        ),
    )
    conn.commit()
    conn.close()


def load_intrinsics(
    serial_number: str,
    resolution: tuple[int, int],
    db_path: Path = DEFAULT_INTRINSICS_DB,
) -> CameraIntrinsics | None:
    """
    Load camera intrinsics from database.

    Args:
        serial_number: Camera serial number
        resolution: (width, height) tuple
        db_path: Path to SQLite database file

    Returns:
        CameraIntrinsics if found, None otherwise
    """
    if not db_path.exists():
        return None

    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """
        SELECT matrix, distortion, error, grid_count
        FROM intrinsics
        WHERE serial_number = ? AND width = ? AND height = ?
    """,
        (serial_number, resolution[0], resolution[1]),
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return CameraIntrinsics(
        serial_number=serial_number,
        resolution=resolution,
        matrix=np.frombuffer(row[0], dtype=np.float64).reshape(3, 3),
        distortion=np.frombuffer(row[1], dtype=np.float64),
        error=row[2],
        grid_count=row[3],
    )


def list_intrinsics(
    db_path: Path = DEFAULT_INTRINSICS_DB,
) -> list[tuple[str, int, int, float]]:
    """
    List all stored intrinsics.

    Args:
        db_path: Path to SQLite database file

    Returns:
        List of (serial_number, width, height, error) tuples
    """
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT serial_number, width, height, error
        FROM intrinsics
        ORDER BY serial_number, width, height
    """)

    rows = cursor.fetchall()
    conn.close()

    return rows


def delete_intrinsics(
    serial_number: str,
    resolution: tuple[int, int] | None = None,
    db_path: Path = DEFAULT_INTRINSICS_DB,
) -> int:
    """
    Delete intrinsics from database.

    Args:
        serial_number: Camera serial number
        resolution: Optional (width, height) - if None, deletes all for serial
        db_path: Path to SQLite database file

    Returns:
        Number of rows deleted
    """
    if not db_path.exists():
        return 0

    conn = sqlite3.connect(db_path)

    if resolution is not None:
        cursor = conn.execute(
            """
            DELETE FROM intrinsics
            WHERE serial_number = ? AND width = ? AND height = ?
        """,
            (serial_number, resolution[0], resolution[1]),
        )
    else:
        cursor = conn.execute(
            """
            DELETE FROM intrinsics
            WHERE serial_number = ?
        """,
            (serial_number,),
        )

    deleted = cursor.rowcount
    conn.commit()
    conn.close()

    return deleted


# ============================================================================
# Extrinsics Storage (per-project TOML)
# ============================================================================


def save_calibration_to_toml(
    cameras: dict[int, CalibratedCamera],
    path: Path,
) -> None:
    """
    Save calibrated cameras to a TOML file.

    This stores extrinsics for a specific project (intrinsics are in SQLite).

    Args:
        cameras: Dict of port -> CalibratedCamera
        path: Path to calibration.toml file
    """
    import cv2

    data = {"cameras": {}}

    for port, cam in cameras.items():
        # Convert rotation to Rodrigues (3 params) for compact storage
        rodrigues = cv2.Rodrigues(cam.extrinsics.rotation)[0][:, 0]

        data["cameras"][str(port)] = {
            "serial_number": cam.serial_number,
            "port": port,
            "rotation": rodrigues.tolist(),
            "translation": cam.extrinsics.translation.tolist(),
            # Include intrinsics reference info
            "intrinsics_resolution": list(cam.intrinsics.resolution),
            "intrinsics_error": cam.intrinsics.error,
        }

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        rtoml.dump(data, f)


def load_calibration_from_toml(
    path: Path,
    db_path: Path = DEFAULT_INTRINSICS_DB,
) -> dict[int, CalibratedCamera] | None:
    """
    Load calibrated cameras from a TOML file.

    Intrinsics are loaded from SQLite database.

    Args:
        path: Path to calibration.toml file
        db_path: Path to intrinsics database

    Returns:
        Dict of port -> CalibratedCamera, or None if file doesn't exist
    """
    import cv2

    if not path.exists():
        return None

    data = rtoml.load(path)
    cameras = {}

    for port_str, cam_data in data.get("cameras", {}).items():
        port = int(port_str)
        serial = cam_data["serial_number"]
        resolution = tuple(cam_data["intrinsics_resolution"])

        # Load intrinsics from database
        intrinsics = load_intrinsics(serial, resolution, db_path)
        if intrinsics is None:
            # Can't load without intrinsics
            continue

        # Convert Rodrigues back to rotation matrix
        rotation = cv2.Rodrigues(np.array(cam_data["rotation"]))[0]
        translation = np.array(cam_data["translation"], dtype=np.float64)

        extrinsics = CameraExtrinsics(rotation=rotation, translation=translation)

        cameras[port] = CalibratedCamera(
            serial_number=serial,
            port=port,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

    return cameras
