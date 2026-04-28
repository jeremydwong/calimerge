"""
Round-trip tests for ``frame_time_history.csv`` -> keypoints CSV / npz.

Covers Task 1 of the worktree changes:

* The ``camera_frame_time_s`` column appears in ``keypoints_3d.csv`` when
  a frame_time_history is supplied, with values pulled from the
  canonical (port 0) row at each ``sync_index``.
* The ``frame_times_per_port`` array appears in
  ``keypoints_3d.raw.npz`` when a frame_time_history is supplied,
  shaped ``(n_frames, n_cameras)`` with NaN where a port dropped a
  frame for that sync_index.
* Backwards compat: with ``frame_time_history_path=None`` the legacy
  schema is preserved verbatim.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from calimerge.keypoint_export import (
    CSV_FIELD_CAMERA_FRAME_TIME,
    CSV_FILENAME,
    FRAME_TIME_HISTORY_FILENAME,
    RAW_FILENAME,
    export_session_csv,
    write_keypoints_csv,
    write_raw_buffer,
)


def _make_buffer(n_frames: int = 4, n_persons: int = 1, n_kps: int = 3):
    buf: list[dict] = []
    for i in range(n_frames):
        persons = []
        for p in range(n_persons):
            person = []
            for k in range(n_kps):
                person.append(
                    np.array(
                        [float(i), float(p), float(k)],
                        dtype=np.float32,
                    )
                )
            persons.append(person)
        buf.append({"time": i * 0.04, "persons": persons, "primary_index": 0})
    return buf


def _write_history(
    path: Path,
    rows: list[tuple[int, int, int, float]],
    *,
    with_comment: bool = True,
) -> None:
    """Emit a frame_time_history.csv exactly as RecordingWorker does."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        if with_comment:
            f.write("# cameras: 0=AAA,1=BBB\n")
        writer = csv.writer(f)
        writer.writerow(["sync_index", "port", "frame_index", "frame_time"])
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# CSV column injection
# ---------------------------------------------------------------------------


def test_csv_includes_camera_frame_time(tmp_path: Path):
    """The new column appears and carries port-0 values from the history."""
    buf = _make_buffer(n_frames=3, n_persons=1, n_kps=2)

    history = tmp_path / FRAME_TIME_HISTORY_FILENAME
    _write_history(
        history,
        [
            (0, 0, 0, 0.000),
            (0, 1, 0, 0.001),
            (1, 0, 1, 0.033),
            (1, 1, 1, 0.034),
            # sync_index 2 — port 0 dropped, only port 1 has a row
            (2, 1, 2, 0.067),
        ],
    )

    csv_path = tmp_path / CSV_FILENAME
    rows = write_keypoints_csv(
        csv_path,
        buf,
        num_keypoints=2,
        frame_time_history_path=history,
    )
    assert rows == 3 * 1 * 2

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    assert CSV_FIELD_CAMERA_FRAME_TIME in reader.fieldnames

    # Bucket by sync_index so we can spot-check easily.
    by_sync: dict[int, list[dict]] = {}
    for row in all_rows:
        by_sync.setdefault(int(row["sync_index"]), []).append(row)

    # sync_index 0 -> port 0 frame_time 0.000
    for r in by_sync[0]:
        assert float(r[CSV_FIELD_CAMERA_FRAME_TIME]) == pytest.approx(0.0)
    # sync_index 1 -> port 0 frame_time 0.033
    for r in by_sync[1]:
        assert float(r[CSV_FIELD_CAMERA_FRAME_TIME]) == pytest.approx(0.033)
    # sync_index 2 -> port 0 dropped, value should be empty
    for r in by_sync[2]:
        assert r[CSV_FIELD_CAMERA_FRAME_TIME] == ""


def test_csv_omits_camera_frame_time_when_history_missing(tmp_path: Path):
    """Backwards compat: no history -> no extra column, legacy schema kept."""
    buf = _make_buffer(n_frames=2, n_persons=1, n_kps=2)
    csv_path = tmp_path / CSV_FILENAME
    write_keypoints_csv(csv_path, buf, num_keypoints=2)

    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert CSV_FIELD_CAMERA_FRAME_TIME not in header
    # Legacy 9-column schema preserved verbatim.
    assert header == [
        "time_s", "sync_index", "person_index", "person_id", "kp_index",
        "x", "y", "z", "valid",
    ]


def test_export_session_auto_picks_up_history(tmp_path: Path):
    """``export_session_csv`` autodetects ``frame_time_history.csv`` next to it."""
    buf = _make_buffer(n_frames=2, n_persons=1, n_kps=2)

    history = tmp_path / FRAME_TIME_HISTORY_FILENAME
    _write_history(
        history,
        [
            (0, 0, 0, 0.000),
            (1, 0, 1, 0.040),
        ],
    )

    csv_path, _, _ = export_session_csv(
        tmp_path,
        buf,
        num_keypoints=2,
    )

    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert CSV_FIELD_CAMERA_FRAME_TIME in header


# ---------------------------------------------------------------------------
# npz frame_times_per_port
# ---------------------------------------------------------------------------


def test_npz_frame_times_per_port_shape_and_nan_padding(tmp_path: Path):
    buf = _make_buffer(n_frames=4, n_persons=1, n_kps=2)

    history = tmp_path / FRAME_TIME_HISTORY_FILENAME
    _write_history(
        history,
        [
            (0, 0, 0, 0.000),
            (0, 1, 0, 0.001),
            (1, 0, 1, 0.033),
            (1, 1, 1, 0.034),
            (2, 0, 2, 0.067),
            # sync_index 2 -> port 1 dropped
            (3, 0, 3, 0.100),
            (3, 1, 3, 0.101),
        ],
    )

    raw_path = tmp_path / RAW_FILENAME
    write_raw_buffer(raw_path, buf, frame_time_history_path=history)

    arrs = np.load(raw_path)
    assert "frame_times_per_port" in arrs
    assert "frame_time_ports" in arrs

    ports = arrs["frame_time_ports"]
    np.testing.assert_array_equal(ports, np.asarray([0, 1], dtype=np.int32))

    ftpp = arrs["frame_times_per_port"]
    assert ftpp.shape == (4, 2)
    assert ftpp[0, 0] == pytest.approx(0.000)
    assert ftpp[0, 1] == pytest.approx(0.001)
    assert ftpp[1, 0] == pytest.approx(0.033)
    assert ftpp[1, 1] == pytest.approx(0.034)
    assert ftpp[2, 0] == pytest.approx(0.067)
    # Dropped frame on port 1 -> NaN
    assert math.isnan(float(ftpp[2, 1]))
    assert ftpp[3, 0] == pytest.approx(0.100)
    assert ftpp[3, 1] == pytest.approx(0.101)


def test_npz_omits_frame_times_when_history_missing(tmp_path: Path):
    """Backwards compat: no path -> no extra arrays."""
    buf = _make_buffer(n_frames=2, n_persons=1, n_kps=2)
    raw_path = tmp_path / RAW_FILENAME
    write_raw_buffer(raw_path, buf)
    arrs = np.load(raw_path)
    assert "frame_times_per_port" not in arrs.files
    assert "frame_time_ports" not in arrs.files


def test_history_with_only_port_1_uses_first_seen_port(tmp_path: Path):
    """Canonical port falls back to first seen if port 0 is absent."""
    buf = _make_buffer(n_frames=2, n_persons=1, n_kps=1)

    history = tmp_path / FRAME_TIME_HISTORY_FILENAME
    _write_history(
        history,
        [
            (0, 1, 0, 0.005),
            (1, 1, 1, 0.045),
        ],
    )

    csv_path = tmp_path / CSV_FILENAME
    write_keypoints_csv(
        csv_path,
        buf,
        num_keypoints=1,
        frame_time_history_path=history,
    )

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert CSV_FIELD_CAMERA_FRAME_TIME in reader.fieldnames
    assert float(rows[0][CSV_FIELD_CAMERA_FRAME_TIME]) == pytest.approx(0.005)
    assert float(rows[1][CSV_FIELD_CAMERA_FRAME_TIME]) == pytest.approx(0.045)
