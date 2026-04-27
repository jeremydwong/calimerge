"""
Tests for ``calimerge.keypoint_export`` -- the post-workout CSV writer.

Covers:

* round-trip of a synthesised ``_recording_keypoints`` buffer through
  CSV + meta and back via :func:`load_keypoints_3d_csv`,
* NaN / missing keypoint handling,
* exact-match intrinsics_match_method recording,
* raw-buffer dump/load (used to keep queued jobs replayable across
  process restarts),
* job descriptor + iter_jobs filtering.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from calimerge.keypoint_export import (
    CSV_FILENAME,
    META_FILENAME,
    RAW_FILENAME,
    build_meta,
    export_session_csv,
    iter_jobs,
    load_keypoints_3d_csv,
    make_job_descriptor,
    read_raw_buffer,
    write_keypoints_csv,
    write_raw_buffer,
)


# ---------------------------------------------------------------------------
# Synthesised buffer fixture
# ---------------------------------------------------------------------------


def _make_buffer(n_frames: int = 4, n_persons: int = 2, n_kps: int = 6):
    """Build a deterministic ``_recording_keypoints`` list."""
    buf: list[dict] = []
    for i in range(n_frames):
        persons = []
        for p in range(n_persons):
            person = []
            for k in range(n_kps):
                # Drop a couple of keypoints to exercise the NaN path.
                if (i, p, k) in {(0, 0, 2), (1, 1, 5), (2, 0, 3)}:
                    person.append(None)
                else:
                    person.append(
                        np.array(
                            [
                                float(i) + 0.1 * p + 0.01 * k,
                                10.0 + p,
                                100.0 - k,
                            ],
                            dtype=np.float32,
                        )
                    )
            persons.append(person)
        buf.append(
            {
                "time": i * 0.04,  # 25 fps
                "persons": persons,
                "primary_index": i % n_persons,
            }
        )
    return buf


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_csv_roundtrip(tmp_path: Path):
    buf = _make_buffer(n_frames=4, n_persons=2, n_kps=6)

    csv_path, meta_path, rows = export_session_csv(
        tmp_path,
        buf,
        num_keypoints=6,
        session_id=42,
        model_backend="pytorch",
        model_name="vitpose_synthpose",
    )

    assert csv_path == tmp_path / CSV_FILENAME
    assert meta_path == tmp_path / META_FILENAME
    assert rows == 4 * 2 * 6
    assert csv_path.exists() and meta_path.exists()

    loaded = load_keypoints_3d_csv(tmp_path)
    assert loaded["num_keypoints"] == 6
    assert len(loaded["frames"]) == 4

    for orig_frame, frame in zip(buf, loaded["frames"]):
        assert frame["time"] == pytest.approx(orig_frame["time"], abs=1e-5)
        assert len(frame["persons"]) == 2
        for p_idx, person in enumerate(frame["persons"]):
            assert person["person_index"] == p_idx
            for k_idx, kp in enumerate(person["keypoints"]):
                orig = orig_frame["persons"][p_idx][k_idx]
                if orig is None:
                    assert kp is None, (p_idx, k_idx)
                else:
                    assert kp is not None
                    np.testing.assert_allclose(
                        np.array(kp, dtype=np.float32),
                        np.asarray(orig, dtype=np.float32),
                        atol=1e-3,
                    )


def test_nan_handling_explicit(tmp_path: Path):
    """Keypoints containing NaN should round-trip as None / valid=0."""
    buf = [
        {
            "time": 0.0,
            "persons": [
                [
                    np.array([1.0, 2.0, 3.0], dtype=np.float32),
                    np.array([np.nan, 1.0, 1.0], dtype=np.float32),
                    None,
                ]
            ],
            "primary_index": 0,
        }
    ]

    csv_path, _, _ = export_session_csv(tmp_path, buf, num_keypoints=3)

    # Manually inspect the CSV to assert the validity column.
    text = csv_path.read_text(encoding="utf-8").splitlines()
    header = text[0].split(",")
    valid_idx = header.index("valid")
    rows = [line.split(",") for line in text[1:]]
    assert rows[0][valid_idx] == "1"
    assert rows[1][valid_idx] == "0"  # NaN inside
    assert rows[2][valid_idx] == "0"  # missing entirely

    loaded = load_keypoints_3d_csv(tmp_path)
    kps = loaded["frames"][0]["persons"][0]["keypoints"]
    assert kps[0] == pytest.approx((1.0, 2.0, 3.0))
    assert kps[1] is None
    assert kps[2] is None


# ---------------------------------------------------------------------------
# Intrinsics match method
# ---------------------------------------------------------------------------


def test_intrinsics_match_method_exact(tmp_path: Path, sample_camera_intrinsics):
    """When intrinsics live in the DB at the requested resolution -> 'exact'."""
    from calimerge.config import save_intrinsics
    from calimerge.types import CalibratedCamera, CameraExtrinsics

    db_path = tmp_path / "intrinsics.db"
    save_intrinsics(sample_camera_intrinsics, db_path=db_path)

    cam = CalibratedCamera(
        serial_number=sample_camera_intrinsics.serial_number,
        port=0,
        intrinsics=sample_camera_intrinsics,
        extrinsics=CameraExtrinsics(
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
        ),
    )

    meta = build_meta(
        tmp_path,
        calibrated_cameras={0: cam},
        intrinsics_db_path=db_path,
        model_backend="pytorch",
        model_name="vitpose_synthpose",
    )

    assert meta["camera_serials_in_order"] == [sample_camera_intrinsics.serial_number]
    assert meta["camera_resolutions"] == {"0": [1280, 720]}
    assert meta["intrinsics_match_method"] == {"0": "exact"}
    assert meta["intrinsics_resolutions_used"] == {"0": [1280, 720]}
    assert meta["model_backend"] == "pytorch"
    assert meta["model_name"] == "vitpose_synthpose"
    assert meta["num_keypoints"] == 52


def test_intrinsics_match_method_none_when_db_missing(tmp_path: Path):
    """Graceful fallback when intrinsics DB does not exist."""
    from calimerge.types import (
        CalibratedCamera, CameraExtrinsics, CameraIntrinsics,
    )

    cam = CalibratedCamera(
        serial_number="MISSING",
        port=0,
        intrinsics=CameraIntrinsics(
            serial_number="MISSING",
            resolution=(640, 480),
            matrix=np.eye(3),
            distortion=np.zeros(5),
            error=0.0,
            grid_count=0,
        ),
        extrinsics=CameraExtrinsics(
            rotation=np.eye(3),
            translation=np.zeros(3),
        ),
    )
    meta = build_meta(
        tmp_path,
        calibrated_cameras={0: cam},
        intrinsics_db_path=tmp_path / "nope.db",
    )
    assert meta["intrinsics_match_method"] == {"0": "none"}


# ---------------------------------------------------------------------------
# Raw buffer round trip (used by queued jobs)
# ---------------------------------------------------------------------------


def test_raw_buffer_roundtrip(tmp_path: Path):
    buf = _make_buffer(n_frames=3, n_persons=1, n_kps=4)
    raw_path = tmp_path / RAW_FILENAME
    write_raw_buffer(raw_path, buf)
    assert raw_path.exists()
    rehydrated = read_raw_buffer(raw_path)
    assert len(rehydrated) == len(buf)
    for orig, got in zip(buf, rehydrated):
        assert got["time"] == pytest.approx(orig["time"])
        for p_orig, p_got in zip(orig["persons"], got["persons"]):
            for kp_orig, kp_got in zip(p_orig, p_got):
                if kp_orig is None:
                    assert kp_got is None
                else:
                    np.testing.assert_allclose(
                        np.asarray(kp_got, dtype=np.float32),
                        np.asarray(kp_orig, dtype=np.float32),
                        atol=1e-3,
                    )


# ---------------------------------------------------------------------------
# Job descriptors
# ---------------------------------------------------------------------------


def test_job_descriptor_and_iter_jobs(tmp_path: Path):
    sd = tmp_path / "session_x"
    sd.mkdir()
    raw_path = sd / RAW_FILENAME
    write_raw_buffer(raw_path, _make_buffer(2, 1, 3))

    job = make_job_descriptor(
        sd,
        session_id=1,
        model_backend="pytorch",
        model_name="vitpose_synthpose",
    )
    assert job["session_dir"] == str(sd)
    assert job["session_id"] == 1
    assert job["raw_buffer_filename"] == RAW_FILENAME

    # A second job pointing at a non-existent dir should be filtered out.
    bad_job = make_job_descriptor(
        tmp_path / "ghost",
        session_id=2,
    )
    assert list(iter_jobs([job, bad_job])) == [job]


# ---------------------------------------------------------------------------
# Meta JSON formatting
# ---------------------------------------------------------------------------


def test_meta_written_as_valid_json(tmp_path: Path, sample_camera_intrinsics):
    from calimerge.types import CalibratedCamera, CameraExtrinsics

    cam = CalibratedCamera(
        serial_number=sample_camera_intrinsics.serial_number,
        port=0,
        intrinsics=sample_camera_intrinsics,
        extrinsics=CameraExtrinsics(
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
        ),
    )
    buf = _make_buffer(n_frames=2, n_persons=1, n_kps=3)
    _, meta_path, _ = export_session_csv(
        tmp_path,
        buf,
        num_keypoints=3,
        calibrated_cameras={0: cam},
        session_id=7,
        model_backend="pytorch",
        model_name="vitpose_synthpose",
    )
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["session_id"] == 7
    assert payload["num_keypoints"] == 3
    assert payload["csv_row_count"] == 2 * 1 * 3
    assert payload["camera_serials_in_order"] == [sample_camera_intrinsics.serial_number]
