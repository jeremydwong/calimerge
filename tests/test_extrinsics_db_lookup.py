"""Locks the extrinsic-session lookup behavior used by live capture.

Live capture must pick the extrinsic session whose camera serial set EXACTLY
matches what's currently plugged in — not just the newest session. This test
fakes a small extrinsics.db + intrinsics.db and verifies:

  1. Newest session matching exactly is returned, even if a more recent
     session covers a different camera set.
  2. A session that's a strict superset (extras) is not returned.
  3. A session that's a strict subset (missing) is not returned.
  4. Empty serial set returns None.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _fake_intrinsics(serial: str, w: int = 1280, h: int = 720):
    from calimerge.types import CameraIntrinsics
    matrix = np.array(
        [[1000.0, 0.0, w / 2], [0.0, 1000.0, h / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.zeros(5, dtype=np.float64)
    return CameraIntrinsics(
        serial_number=serial,
        resolution=(w, h),
        matrix=matrix,
        distortion=distortion,
        error=0.5,
        grid_count=20,
    )


def _fake_calibrated_camera(serial: str, port: int):
    from calimerge.types import CalibratedCamera, CameraExtrinsics
    return CalibratedCamera(
        serial_number=serial,
        port=port,
        intrinsics=_fake_intrinsics(serial),
        extrinsics=CameraExtrinsics(
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
        ),
    )


@pytest.fixture
def db_paths(tmp_path: Path):
    """Build an extrinsics.db + intrinsics.db with three sessions."""
    from calimerge import config as cfg

    intr_db = tmp_path / "intrinsics.db"
    extr_db = tmp_path / "extrinsics.db"

    # Seed intrinsics for every serial we'll reference.
    cfg.init_intrinsics_db(intr_db)
    for serial in ("AA", "BB", "CC", "DD"):
        cfg.save_intrinsics(_fake_intrinsics(serial), db_path=intr_db)

    # Session 1 (oldest): cameras AA + BB (the 2-cam rig)
    cfg.save_extrinsic_session(
        cameras={
            0: _fake_calibrated_camera("AA", 0),
            1: _fake_calibrated_camera("BB", 1),
        },
        rmse=1.0,
        notes="2-cam rig",
        nicknames={},
        db_path=extr_db,
    )

    # Session 2: cameras AA + BB + CC (the 3-cam rig)
    cfg.save_extrinsic_session(
        cameras={
            0: _fake_calibrated_camera("AA", 0),
            1: _fake_calibrated_camera("BB", 1),
            2: _fake_calibrated_camera("CC", 2),
        },
        rmse=1.0,
        notes="3-cam rig",
        nicknames={},
        db_path=extr_db,
    )

    # Session 3 (newest): cameras AA + BB again — different rig physical setup.
    cfg.save_extrinsic_session(
        cameras={
            0: _fake_calibrated_camera("AA", 0),
            1: _fake_calibrated_camera("BB", 1),
        },
        rmse=0.5,
        notes="2-cam rig redo",
        nicknames={},
        db_path=extr_db,
    )

    return extr_db, intr_db


def test_exact_match_returns_newest_matching(db_paths):
    """Asking for {AA, BB} must return session 3 (newest), not session 1."""
    from calimerge.config import find_extrinsic_session_by_serials
    extr_db, intr_db = db_paths

    result = find_extrinsic_session_by_serials(
        {"AA", "BB"}, db_path=extr_db, intrinsics_db=intr_db,
    )
    assert result is not None
    session_id, _created_at, cameras = result
    assert {c.serial_number for c in cameras.values()} == {"AA", "BB"}
    # Session 3 is the newest 2-cam match (id=3, since session ids are 1-indexed)
    assert session_id == 3


def test_exact_match_returns_three_cam_rig(db_paths):
    from calimerge.config import find_extrinsic_session_by_serials
    extr_db, intr_db = db_paths

    result = find_extrinsic_session_by_serials(
        {"AA", "BB", "CC"}, db_path=extr_db, intrinsics_db=intr_db,
    )
    assert result is not None
    session_id, _created_at, cameras = result
    assert {c.serial_number for c in cameras.values()} == {"AA", "BB", "CC"}
    assert session_id == 2


def test_strict_superset_rejected(db_paths):
    """Asking for {AA, BB, DD} must return None — the 3-cam session covers AA/BB/CC, not DD."""
    from calimerge.config import find_extrinsic_session_by_serials
    extr_db, intr_db = db_paths

    result = find_extrinsic_session_by_serials(
        {"AA", "BB", "DD"}, db_path=extr_db, intrinsics_db=intr_db,
    )
    assert result is None


def test_strict_subset_rejected(db_paths):
    """Asking for just {AA} must return None — no session has only AA."""
    from calimerge.config import find_extrinsic_session_by_serials
    extr_db, intr_db = db_paths

    result = find_extrinsic_session_by_serials(
        {"AA"}, db_path=extr_db, intrinsics_db=intr_db,
    )
    assert result is None


def test_empty_serial_set_returns_none(db_paths):
    from calimerge.config import find_extrinsic_session_by_serials
    extr_db, intr_db = db_paths

    assert find_extrinsic_session_by_serials(
        set(), db_path=extr_db, intrinsics_db=intr_db,
    ) is None


def test_missing_db_returns_none(tmp_path: Path):
    from calimerge.config import find_extrinsic_session_by_serials
    nonexistent = tmp_path / "nope.db"
    assert find_extrinsic_session_by_serials(
        {"AA"}, db_path=nonexistent, intrinsics_db=tmp_path / "also-nope.db",
    ) is None
