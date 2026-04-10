"""
Configuration loading/saving.

Pure functions operating on dataclasses.
- TOML for project configuration
- SQLite for camera intrinsics (keyed by serial_number + resolution)
- JSON for app settings (last project folder) and per-project settings
"""

from __future__ import annotations

import json
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nicknames (
            serial_number TEXT PRIMARY KEY,
            nickname TEXT NOT NULL
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
    allow_scaling: bool = True,
) -> CameraIntrinsics | None:
    """
    Load camera intrinsics from database.

    If an exact resolution match isn't found but intrinsics exist at a different
    resolution with the same aspect ratio, they will be automatically scaled
    (unless allow_scaling=False).

    Args:
        serial_number: Camera serial number
        resolution: (width, height) tuple
        db_path: Path to SQLite database file
        allow_scaling: If True, scale intrinsics from same aspect ratio if exact not found

    Returns:
        CameraIntrinsics if found (possibly scaled), None otherwise
    """
    if not db_path.exists():
        return None

    conn = sqlite3.connect(db_path)

    # First try exact match
    cursor = conn.execute(
        """
        SELECT matrix, distortion, error, grid_count
        FROM intrinsics
        WHERE serial_number = ? AND width = ? AND height = ?
    """,
        (serial_number, resolution[0], resolution[1]),
    )

    row = cursor.fetchone()

    if row is not None:
        conn.close()
        return CameraIntrinsics(
            serial_number=serial_number,
            resolution=resolution,
            matrix=np.frombuffer(row[0], dtype=np.float64).reshape(3, 3),
            distortion=np.frombuffer(row[1], dtype=np.float64),
            error=row[2],
            grid_count=row[3],
        )

    # No exact match - try scaling if allowed
    if not allow_scaling:
        conn.close()
        return None

    # Find all intrinsics for this camera
    cursor = conn.execute(
        """
        SELECT width, height, matrix, distortion, error, grid_count
        FROM intrinsics
        WHERE serial_number = ?
        ORDER BY width * height DESC
    """,
        (serial_number,),
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    # Import here to avoid circular dependency
    from .types import same_aspect_ratio, scale_intrinsics

    # Find best match with same aspect ratio (prefer higher resolution source)
    for db_w, db_h, matrix_bytes, dist_bytes, error, grid_count in rows:
        db_resolution = (db_w, db_h)
        if same_aspect_ratio(db_resolution, resolution):
            # Found a scalable match
            source_intrinsics = CameraIntrinsics(
                serial_number=serial_number,
                resolution=db_resolution,
                matrix=np.frombuffer(matrix_bytes, dtype=np.float64).reshape(3, 3),
                distortion=np.frombuffer(dist_bytes, dtype=np.float64),
                error=error,
                grid_count=grid_count,
            )
            return scale_intrinsics(source_intrinsics, resolution)

    return None


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


def check_intrinsics_availability(
    serial_number: str,
    target_resolution: tuple[int, int],
    db_path: Path = DEFAULT_INTRINSICS_DB,
) -> tuple[str, tuple[int, int] | None]:
    """
    Check what intrinsics are available for a camera.

    Args:
        serial_number: Camera serial number
        target_resolution: Desired (width, height)
        db_path: Path to SQLite database file

    Returns:
        Tuple of (status_string, source_resolution_or_none)
        - ("exact", (w, h)) if exact match exists
        - ("scalable", (w, h)) if same-aspect-ratio match exists (returns best source res)
        - ("mismatch", None) if intrinsics exist but wrong aspect ratio
        - ("none", None) if no intrinsics for this camera
    """
    if not db_path.exists():
        return ("none", None)

    from .types import same_aspect_ratio

    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """
        SELECT width, height, error
        FROM intrinsics
        WHERE serial_number = ?
        ORDER BY width * height DESC
    """,
        (serial_number,),
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return ("none", None)

    # Check for exact match first
    for w, h, _ in rows:
        if (w, h) == target_resolution:
            return ("exact", (w, h))

    # Check for scalable match (same aspect ratio)
    for w, h, _ in rows:
        if same_aspect_ratio((w, h), target_resolution):
            return ("scalable", (w, h))

    # Intrinsics exist but wrong aspect ratio
    return ("mismatch", rows[0][:2])  # Return highest-res available


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
# Camera Nicknames
# ============================================================================


def save_nickname(
    serial_number: str,
    nickname: str,
    db_path: Path = DEFAULT_INTRINSICS_DB,
) -> None:
    """Save a camera nickname to the database."""
    init_intrinsics_db(db_path)

    conn = sqlite3.connect(db_path)
    if nickname.strip():
        conn.execute(
            "INSERT OR REPLACE INTO nicknames (serial_number, nickname) VALUES (?, ?)",
            (serial_number, nickname.strip()),
        )
    else:
        conn.execute(
            "DELETE FROM nicknames WHERE serial_number = ?",
            (serial_number,),
        )
    conn.commit()
    conn.close()


def load_all_nicknames(
    db_path: Path = DEFAULT_INTRINSICS_DB,
) -> dict[str, str]:
    """Load all nicknames from the database. Returns {serial_number: nickname}."""
    if not db_path.exists():
        return {}

    try:
        init_intrinsics_db(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT serial_number, nickname FROM nicknames")
        result = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return result
    except Exception:
        return {}


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


# ============================================================================
# App Settings  (~/.calimerge/app_settings.json)
# ============================================================================

_APP_SETTINGS_PATH = Path.home() / ".calimerge" / "app_settings.json"

_APP_SETTINGS_DEFAULTS: dict = {
    "last_project_folder": None,
}


def load_app_settings() -> dict:
    """Load application-level settings (persists across projects)."""
    if not _APP_SETTINGS_PATH.exists():
        return dict(_APP_SETTINGS_DEFAULTS)
    try:
        with open(_APP_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {**_APP_SETTINGS_DEFAULTS, **data}
    except Exception:
        return dict(_APP_SETTINGS_DEFAULTS)


def save_app_settings(settings: dict) -> None:
    """Save application-level settings."""
    _APP_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_APP_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


# ============================================================================
# Project Settings  (<project_folder>/settings.json)
# ============================================================================

_PROJECT_SETTINGS_FILENAME = "settings.json"

_PROJECT_SETTINGS_DEFAULTS: dict = {
    "fps": 30,
    "codec": "h264",
    "intrinsic_max_frames": 40,
    "cameras": {},
    "charuco_intrinsic": {
        "columns": 7,
        "rows": 5,
        "square_size_cm": 3.0,
        "inverted": False,
    },
    "charuco_extrinsic": {
        "columns": 4,
        "rows": 3,
        "square_size_cm": 5.0,
        "inverted": False,
    },
}


def load_project_settings(project_folder: Path) -> dict:
    """
    Load per-project settings from <project_folder>/settings.json.

    Returns defaults merged with whatever is stored on disk.
    """
    path = project_folder / _PROJECT_SETTINGS_FILENAME
    if not path.exists():
        return _deep_copy_defaults(_PROJECT_SETTINGS_DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result = _deep_copy_defaults(_PROJECT_SETTINGS_DEFAULTS)
        _deep_merge(result, data)
        return result
    except Exception:
        return _deep_copy_defaults(_PROJECT_SETTINGS_DEFAULTS)


def save_project_settings(settings: dict, project_folder: Path) -> None:
    """Save per-project settings to <project_folder>/settings.json."""
    project_folder.mkdir(parents=True, exist_ok=True)
    path = project_folder / _PROJECT_SETTINGS_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def _deep_copy_defaults(d: dict) -> dict:
    import copy
    return copy.deepcopy(d)


def _deep_merge(base: dict, override: dict) -> None:
    """Merge override into base in-place (nested dicts merged, not replaced)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ============================================================================
# Workouts Database  (~/.calimerge/workouts.db)
# ============================================================================

DEFAULT_WORKOUTS_DB = Path.home() / ".calimerge" / "workouts.db"


def init_workouts_db(db_path: Path = DEFAULT_WORKOUTS_DB) -> None:
    """Create workouts database and tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            mass_kg REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Migration: add program tracking columns to users
    user_cols = [row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()]
    if "active_program_id" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN active_program_id INTEGER")
    if "program_started_at" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN program_started_at TIMESTAMP")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            workout_type TEXT NOT NULL,
            duration_seconds REAL,
            recording_path TEXT,
            calibration_path TEXT,
            config_blob BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    session_cols = [row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()]
    if "config_blob" not in session_cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN config_blob BLOB")
    if "program_exercise_id" not in session_cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN program_exercise_id INTEGER")
    if "set_number" not in session_cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN set_number INTEGER")
    if "model_version" not in session_cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN model_version TEXT")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            metric_name TEXT NOT NULL,
            metric_value REAL,
            metadata TEXT
        )
    """)

    # Program templates + their exercises
    conn.execute("""
        CREATE TABLE IF NOT EXISTS programs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            description TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS program_exercises (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            program_id INTEGER NOT NULL REFERENCES programs(id),
            workout_type TEXT NOT NULL,
            display_name TEXT NOT NULL,
            sets_per_day INTEGER NOT NULL,
            target_reps INTEGER,
            target_duration_seconds REAL,
            days_per_week INTEGER NOT NULL,
            suggested_days TEXT,
            break_seconds INTEGER DEFAULT 60,
            order_index INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()

    # Seed default program templates (idempotent)
    _seed_default_programs(db_path)


def _seed_default_programs(db_path: Path = DEFAULT_WORKOUTS_DB) -> None:
    """Insert the default Vivifrail and Calisthenics programs if they don't exist."""
    from .programs import DEFAULT_PROGRAMS

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    for prog in DEFAULT_PROGRAMS:
        existing = conn.execute(
            "SELECT id FROM programs WHERE name = ?", (prog["name"],)
        ).fetchone()
        if existing:
            continue  # already seeded

        cur = conn.execute(
            "INSERT INTO programs (name, display_name, description) VALUES (?, ?, ?)",
            (prog["name"], prog["display_name"], prog.get("description", "")),
        )
        program_id = cur.lastrowid

        for ex in prog.get("exercises", []):
            conn.execute(
                "INSERT INTO program_exercises "
                "(program_id, workout_type, display_name, sets_per_day, "
                " target_reps, target_duration_seconds, days_per_week, "
                " suggested_days, break_seconds, order_index) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    program_id,
                    ex["workout_type"],
                    ex["display_name"],
                    int(ex["sets_per_day"]),
                    ex.get("target_reps"),
                    ex.get("target_duration_seconds"),
                    int(ex["days_per_week"]),
                    ex.get("suggested_days"),
                    int(ex.get("break_seconds", 60)),
                    int(ex.get("order_index", 0)),
                ),
            )

    conn.commit()
    conn.close()


# ─── Config blob serialization ───
# Format (little-endian):
#   header: int32 magic (0x43414C49 'CALI'), int32 version (1), int32 num_cameras
#   per camera:
#     int32 port
#     uint8 serial_len, serial bytes
#     uint8 workout_type_len — (unused, reserved for future)
#     int32 width, int32 height
#     9 × float64 camera matrix (3x3 row-major)
#     5 × float64 distortion coefficients
#     9 × float64 rotation matrix (3x3 row-major)
#     3 × float64 translation vector

_CONFIG_BLOB_MAGIC = 0x43414C49  # 'CALI'
_CONFIG_BLOB_VERSION = 1


def pack_session_config(calibrated_cameras: dict) -> bytes:
    """
    Pack a dict[port, CalibratedCamera] into a compact binary blob.

    Parameters
    ----------
    calibrated_cameras : dict[int, CalibratedCamera]
        Keyed by port, each with intrinsics + extrinsics.

    Returns
    -------
    bytes suitable for storing in a BLOB column.
    """
    import struct

    parts = []
    parts.append(struct.pack("<iii", _CONFIG_BLOB_MAGIC, _CONFIG_BLOB_VERSION,
                             len(calibrated_cameras)))

    for port in sorted(calibrated_cameras.keys()):
        cam = calibrated_cameras[port]
        serial = cam.serial_number.encode("utf-8")
        parts.append(struct.pack("<i", port))
        parts.append(struct.pack("<B", len(serial)))
        parts.append(serial)
        parts.append(struct.pack("<B", 0))  # reserved (workout_type_len)

        w, h = cam.intrinsics.resolution
        parts.append(struct.pack("<ii", int(w), int(h)))

        matrix = np.asarray(cam.intrinsics.matrix, dtype=np.float64).flatten()
        parts.append(matrix.tobytes())

        dist = np.asarray(cam.intrinsics.distortion, dtype=np.float64).flatten()
        if dist.size < 5:
            dist = np.concatenate([dist, np.zeros(5 - dist.size)])
        parts.append(dist[:5].tobytes())

        rotation = np.asarray(cam.extrinsics.rotation, dtype=np.float64).flatten()
        parts.append(rotation.tobytes())

        translation = np.asarray(cam.extrinsics.translation, dtype=np.float64).flatten()
        parts.append(translation.tobytes())

    return b"".join(parts)


def unpack_session_config(blob: bytes) -> dict[int, dict]:
    """
    Unpack a config blob into a dict keyed by port.

    Returns
    -------
    dict[int, dict] where each value has:
        "serial_number": str
        "resolution": (int, int)
        "matrix": (3, 3) np.ndarray
        "distortion": (5,) np.ndarray
        "rotation": (3, 3) np.ndarray
        "translation": (3,) np.ndarray
    """
    import struct

    if not blob or len(blob) < 12:
        return {}

    offset = 0
    magic, version, num_cameras = struct.unpack_from("<iii", blob, offset)
    offset += 12

    if magic != _CONFIG_BLOB_MAGIC or version != _CONFIG_BLOB_VERSION:
        raise ValueError(f"Invalid config blob header: magic={magic:#x} version={version}")

    result: dict[int, dict] = {}
    for _ in range(num_cameras):
        (port,) = struct.unpack_from("<i", blob, offset); offset += 4
        (serial_len,) = struct.unpack_from("<B", blob, offset); offset += 1
        serial = blob[offset:offset + serial_len].decode("utf-8"); offset += serial_len
        (_reserved,) = struct.unpack_from("<B", blob, offset); offset += 1

        width, height = struct.unpack_from("<ii", blob, offset); offset += 8

        matrix = np.frombuffer(blob, dtype=np.float64, count=9, offset=offset).reshape(3, 3).copy()
        offset += 9 * 8

        distortion = np.frombuffer(blob, dtype=np.float64, count=5, offset=offset).copy()
        offset += 5 * 8

        rotation = np.frombuffer(blob, dtype=np.float64, count=9, offset=offset).reshape(3, 3).copy()
        offset += 9 * 8

        translation = np.frombuffer(blob, dtype=np.float64, count=3, offset=offset).copy()
        offset += 3 * 8

        result[port] = {
            "serial_number": serial,
            "resolution": (width, height),
            "matrix": matrix,
            "distortion": distortion,
            "rotation": rotation,
            "translation": translation,
        }

    return result


def _workouts_conn(db_path: Path = DEFAULT_WORKOUTS_DB) -> sqlite3.Connection:
    init_workouts_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def create_user(username: str, mass_kg: float | None = None,
                db_path: Path = DEFAULT_WORKOUTS_DB) -> dict:
    conn = _workouts_conn(db_path)
    conn.execute("INSERT INTO users (username, mass_kg) VALUES (?, ?)",
                 (username, mass_kg))
    conn.commit()
    row = conn.execute("SELECT * FROM users WHERE username = ?",
                       (username,)).fetchone()
    conn.close()
    return dict(row)


def get_user(username: str, db_path: Path = DEFAULT_WORKOUTS_DB) -> dict | None:
    conn = _workouts_conn(db_path)
    row = conn.execute("SELECT * FROM users WHERE username = ?",
                       (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_users(db_path: Path = DEFAULT_WORKOUTS_DB) -> list[dict]:
    conn = _workouts_conn(db_path)
    rows = conn.execute("SELECT * FROM users ORDER BY username").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_user_mass(user_id: int, mass_kg: float,
                     db_path: Path = DEFAULT_WORKOUTS_DB) -> None:
    conn = _workouts_conn(db_path)
    conn.execute("UPDATE users SET mass_kg = ? WHERE id = ?", (mass_kg, user_id))
    conn.commit()
    conn.close()


def create_session(user_id: int, workout_type: str,
                   duration_seconds: float | None = None,
                   recording_path: str | None = None,
                   calibration_path: str | None = None,
                   config_blob: bytes | None = None,
                   program_exercise_id: int | None = None,
                   set_number: int | None = None,
                   db_path: Path = DEFAULT_WORKOUTS_DB) -> int:
    conn = _workouts_conn(db_path)
    cur = conn.execute(
        "INSERT INTO sessions (user_id, workout_type, duration_seconds, "
        "recording_path, calibration_path, config_blob, "
        "program_exercise_id, set_number) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, workout_type, duration_seconds, recording_path, calibration_path,
         config_blob, program_exercise_id, set_number))
    conn.commit()
    session_id = cur.lastrowid
    conn.close()
    return session_id


def get_session_config(session_id: int,
                       db_path: Path = DEFAULT_WORKOUTS_DB) -> dict[int, dict] | None:
    """Load and unpack the config blob for a session."""
    conn = _workouts_conn(db_path)
    row = conn.execute(
        "SELECT config_blob FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if row is None or row["config_blob"] is None:
        return None
    try:
        return unpack_session_config(row["config_blob"])
    except Exception:
        return None


def get_sessions_for_user(user_id: int,
                          db_path: Path = DEFAULT_WORKOUTS_DB) -> list[dict]:
    conn = _workouts_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_session_result(session_id: int, metric_name: str, metric_value: float,
                        metadata: str | None = None,
                        db_path: Path = DEFAULT_WORKOUTS_DB) -> None:
    conn = _workouts_conn(db_path)
    conn.execute(
        "INSERT INTO session_results (session_id, metric_name, metric_value, metadata) "
        "VALUES (?, ?, ?, ?)",
        (session_id, metric_name, metric_value, metadata))
    conn.commit()
    conn.close()


def delete_session_results(session_id: int,
                           db_path: Path = DEFAULT_WORKOUTS_DB) -> None:
    """Remove all stored metrics for a session. Used before re-analysis."""
    conn = _workouts_conn(db_path)
    conn.execute("DELETE FROM session_results WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


# ─── Program CRUD ───

def list_programs(db_path: Path = DEFAULT_WORKOUTS_DB) -> list[dict]:
    """Return all known program templates."""
    conn = _workouts_conn(db_path)
    rows = conn.execute("SELECT * FROM programs ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_program(program_id: int, db_path: Path = DEFAULT_WORKOUTS_DB) -> dict | None:
    conn = _workouts_conn(db_path)
    row = conn.execute("SELECT * FROM programs WHERE id = ?", (program_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_program_exercises(program_id: int,
                          db_path: Path = DEFAULT_WORKOUTS_DB) -> list[dict]:
    """Return all exercises in a program, ordered by order_index."""
    conn = _workouts_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM program_exercises WHERE program_id = ? "
        "ORDER BY order_index, id",
        (program_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_user_program(user_id: int, program_id: int | None,
                     db_path: Path = DEFAULT_WORKOUTS_DB) -> None:
    """Assign a program to a user. Updates program_started_at on change."""
    from datetime import datetime
    conn = _workouts_conn(db_path)
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn.execute(
        "UPDATE users SET active_program_id = ?, program_started_at = ? WHERE id = ?",
        (program_id, now, user_id),
    )
    conn.commit()
    conn.close()


def get_user_by_id(user_id: int, db_path: Path = DEFAULT_WORKOUTS_DB) -> dict | None:
    conn = _workouts_conn(db_path)
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def count_sets_since(user_id: int, program_exercise_id: int, since: str,
                     db_path: Path = DEFAULT_WORKOUTS_DB) -> int:
    """Count sessions for a given (user, program_exercise) since an ISO timestamp.

    `since` is a string accepted by SQLite's comparison (ISO-formatted datetime).
    """
    conn = _workouts_conn(db_path)
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM sessions "
        "WHERE user_id = ? AND program_exercise_id = ? AND created_at >= ?",
        (user_id, program_exercise_id, since),
    ).fetchone()
    conn.close()
    return int(row["n"]) if row else 0
