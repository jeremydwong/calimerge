"""
Post-workout 3D keypoint CSV export.

Pure functions (no Qt) that turn an in-memory `_recording_keypoints` buffer
into:

* `<session_dir>/keypoints_3d.csv`         -- tabular, regenerable
* `<session_dir>/keypoints_3d.meta.json`   -- provenance for re-running
* `<session_dir>/keypoints_3d.raw.npz`     -- raw buffer, written before
                                              queueing so deferred jobs
                                              survive process restarts.

The CSV columns are:

    time_s, sync_index, person_index, person_id,
    kp_index, x, y, z, valid

`valid` = 1 if the (x, y, z) triple is non-null and finite, else 0
(NaNs are written verbatim).

Companion :func:`load_keypoints_3d_csv` reads everything back into a
plain dict for analysis or playback.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CSV_FIELDS = (
    "time_s",
    "sync_index",
    "person_index",
    "person_id",
    "kp_index",
    "x",
    "y",
    "z",
    "valid",
)

CSV_FILENAME = "keypoints_3d.csv"
META_FILENAME = "keypoints_3d.meta.json"
RAW_FILENAME = "keypoints_3d.raw.npz"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _is_finite_xyz(kp: Any) -> bool:
    if kp is None:
        return False
    try:
        arr = np.asarray(kp, dtype=np.float64)
    except Exception:
        return False
    if arr.shape != (3,):
        return False
    return bool(np.all(np.isfinite(arr)))


def _safe_xyz(kp: Any) -> tuple[float, float, float]:
    """Return (x, y, z) tuple of floats. NaNs preserved if input is bad."""
    if kp is None:
        return (math.nan, math.nan, math.nan)
    try:
        arr = np.asarray(kp, dtype=np.float64).reshape(-1)
    except Exception:
        return (math.nan, math.nan, math.nan)
    if arr.size < 3:
        return (math.nan, math.nan, math.nan)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _person_id_for(persons_packet: Any, person_index: int) -> int:
    """Best-effort track-id extraction. Falls back to person_index."""
    if isinstance(persons_packet, dict):
        ids = persons_packet.get("person_ids") or persons_packet.get("ids")
        if ids is not None and person_index < len(ids):
            try:
                return int(ids[person_index])
            except Exception:
                pass
    return int(person_index)


# ----------------------------------------------------------------------------
# Raw buffer persistence (so queued jobs survive a restart)
# ----------------------------------------------------------------------------


def write_raw_buffer(path: Path, recording_keypoints: list[dict]) -> None:
    """
    Dump the in-memory recording buffer to an .npz file.

    Format mirrors what ``write_keypoints_csv`` consumes so a queued job
    can rehydrate after a restart without rerunning detection.
    """
    if not recording_keypoints:
        return

    n_frames = len(recording_keypoints)
    times = np.zeros(n_frames, dtype=np.float64)
    primary = np.zeros(n_frames, dtype=np.int32)

    # Find the maximum (P, K) shape across the buffer so we can pad densely.
    max_persons = 0
    max_kps = 0
    for fr in recording_keypoints:
        persons = fr.get("persons") or []
        max_persons = max(max_persons, len(persons))
        for p in persons:
            if p is None:
                continue
            try:
                max_kps = max(max_kps, len(p))
            except TypeError:
                pass

    if max_persons == 0 or max_kps == 0:
        return

    keypoints = np.full(
        (n_frames, max_persons, max_kps, 3), np.nan, dtype=np.float32
    )
    person_counts = np.zeros(n_frames, dtype=np.int32)

    for i, fr in enumerate(recording_keypoints):
        times[i] = float(fr.get("time", 0.0))
        primary[i] = int(fr.get("primary_index", 0))
        persons = fr.get("persons") or []
        person_counts[i] = min(len(persons), max_persons)
        for p_idx in range(min(len(persons), max_persons)):
            person = persons[p_idx]
            if person is None:
                continue
            for k_idx in range(min(len(person), max_kps)):
                kp = person[k_idx]
                if kp is None:
                    continue
                try:
                    arr = np.asarray(kp, dtype=np.float32).reshape(-1)
                except Exception:
                    continue
                if arr.size < 3:
                    continue
                keypoints[i, p_idx, k_idx, :] = arr[:3]

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        timestamps=times,
        keypoints_3d=keypoints,
        person_count=person_counts,
        primary_person_index=primary,
    )


def read_raw_buffer(path: Path) -> list[dict]:
    """Reverse of :func:`write_raw_buffer`."""
    if not path.exists():
        return []
    data = np.load(path)
    times = data["timestamps"]
    kps = data["keypoints_3d"]
    counts = data["person_count"]
    primary = (
        data["primary_person_index"]
        if "primary_person_index" in data
        else np.zeros(len(times), dtype=np.int32)
    )

    n_frames = kps.shape[0]
    out: list[dict] = []
    for i in range(n_frames):
        n_persons = int(counts[i])
        persons = []
        for p_idx in range(n_persons):
            person = []
            for k_idx in range(kps.shape[2]):
                xyz = kps[i, p_idx, k_idx]
                if np.all(np.isnan(xyz)):
                    person.append(None)
                else:
                    person.append(np.array(xyz, dtype=np.float32))
            persons.append(person)
        out.append(
            {
                "time": float(times[i]),
                "persons": persons,
                "primary_index": int(primary[i]) if i < len(primary) else 0,
            }
        )
    return out


# ----------------------------------------------------------------------------
# CSV writer
# ----------------------------------------------------------------------------


def write_keypoints_csv(
    csv_path: Path,
    recording_keypoints: list[dict],
    num_keypoints: int = 52,
) -> int:
    """
    Write the buffered keypoints to ``csv_path``.

    Returns the number of rows written (excluding header).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)

        for sync_index, frame in enumerate(recording_keypoints):
            t = float(frame.get("time", 0.0))
            persons = frame.get("persons") or []
            for person_index, person in enumerate(persons):
                if person is None:
                    continue
                try:
                    n_kps = min(len(person), num_keypoints)
                except TypeError:
                    continue
                person_id = _person_id_for(frame, person_index)
                for k_idx in range(n_kps):
                    kp = person[k_idx]
                    valid = 1 if _is_finite_xyz(kp) else 0
                    x, y, z = _safe_xyz(kp)
                    writer.writerow(
                        (
                            f"{t:.6f}",
                            sync_index,
                            person_index,
                            person_id,
                            k_idx,
                            "" if math.isnan(x) else f"{x:.6f}",
                            "" if math.isnan(y) else f"{y:.6f}",
                            "" if math.isnan(z) else f"{z:.6f}",
                            valid,
                        )
                    )
                    rows_written += 1

    return rows_written


# ----------------------------------------------------------------------------
# Provenance / meta.json
# ----------------------------------------------------------------------------


def _read_fps_from_history(session_dir: Path) -> float | None:
    history = session_dir / "frame_time_history.csv"
    if not history.exists():
        return None
    try:
        # Columns vary; we just want recorded frame interval. Pick the first
        # camera with at least 2 timestamps.
        import csv as _csv

        per_port: dict[str, list[float]] = {}
        with open(history, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                port = str(row.get("port", "0"))
                ts = row.get("frame_time") or row.get("timestamp")
                if ts is None:
                    continue
                try:
                    per_port.setdefault(port, []).append(float(ts))
                except ValueError:
                    continue

        for stamps in per_port.values():
            if len(stamps) >= 2:
                stamps.sort()
                deltas = np.diff(np.asarray(stamps, dtype=np.float64))
                deltas = deltas[deltas > 0]
                if deltas.size and float(np.median(deltas)) > 0:
                    return float(1.0 / np.median(deltas))
    except Exception:
        return None
    return None


def _intrinsics_match_method(
    serial_number: str,
    target_resolution: tuple[int, int],
    db_path: Path | None,
) -> tuple[str, tuple[int, int] | None]:
    """
    Return ("exact" | "same_ar_scale" | "cross_ar_scale" | "none",
            source_resolution_or_None).

    Falls back gracefully when the intrinsics DB is missing.
    """
    if db_path is None:
        return ("none", None)
    try:
        from .config import check_intrinsics_availability  # local import

        status, source_res = check_intrinsics_availability(
            serial_number, target_resolution, db_path
        )
        if status == "exact":
            return ("exact", source_res)
        if status == "scalable":
            return ("same_ar_scale", source_res)
        if status == "mismatch":
            return ("cross_ar_scale", source_res)
        return ("none", None)
    except Exception:
        return ("none", None)


def build_meta(
    session_dir: Path,
    *,
    calibrated_cameras: dict[int, Any] | None = None,
    extrinsic_session_id: int | None = None,
    extrinsic_calibrated_at: str | None = None,
    model_backend: str | None = None,
    model_name: str | None = None,
    num_keypoints: int = 52,
    session_id: int | None = None,
    intrinsics_db_path: Path | None = None,
    extra: dict | None = None,
) -> dict:
    """
    Build the meta.json payload describing how this CSV was generated.

    Looks up each camera's intrinsics in the SQLite DB to record which
    match strategy was used (exact / same-AR scale / cross-AR scale).
    """
    serials_in_order: list[str] = []
    resolutions: dict[str, list[int]] = {}
    intrinsics_used: dict[str, list[int]] = {}
    match_methods: dict[str, str] = {}

    if calibrated_cameras:
        for port in sorted(calibrated_cameras.keys()):
            cam = calibrated_cameras[port]
            serial = getattr(cam, "serial_number", None) or (
                cam["serial_number"] if isinstance(cam, dict) else None
            )
            if serial is None:
                continue
            res = getattr(getattr(cam, "intrinsics", None), "resolution", None)
            if res is None and isinstance(cam, dict):
                res = cam.get("resolution")
            if res is None:
                continue

            serials_in_order.append(str(serial))
            resolutions[str(port)] = [int(res[0]), int(res[1])]

            method, source_res = _intrinsics_match_method(
                str(serial), (int(res[0]), int(res[1])), intrinsics_db_path
            )
            match_methods[str(port)] = method
            if source_res is not None:
                intrinsics_used[str(port)] = [int(source_res[0]), int(source_res[1])]
            else:
                intrinsics_used[str(port)] = [int(res[0]), int(res[1])]

    try:
        from . import __version__ as calimerge_version
    except Exception:
        calimerge_version = "unknown"

    fps = _read_fps_from_history(session_dir)

    meta: dict = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "extrinsic_session_id": extrinsic_session_id,
        "extrinsic_calibrated_at": extrinsic_calibrated_at,
        "camera_serials_in_order": serials_in_order,
        "camera_resolutions": resolutions,
        "intrinsics_resolutions_used": intrinsics_used,
        "intrinsics_match_method": match_methods,
        "model_backend": model_backend,
        "model_name": model_name,
        "num_keypoints": int(num_keypoints),
        "fps_recorded": fps,
        "session_id": session_id,
        "calimerge_version": calimerge_version,
    }
    if extra:
        meta.update(extra)
    return meta


def write_meta(meta_path: Path, meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=False, default=str)


# ----------------------------------------------------------------------------
# Top-level "do it all" function for the worker
# ----------------------------------------------------------------------------


def export_session_csv(
    session_dir: Path,
    recording_keypoints: list[dict],
    *,
    calibrated_cameras: dict[int, Any] | None = None,
    extrinsic_session_id: int | None = None,
    extrinsic_calibrated_at: str | None = None,
    model_backend: str | None = None,
    model_name: str | None = None,
    num_keypoints: int = 52,
    session_id: int | None = None,
    intrinsics_db_path: Path | None = None,
    extra_meta: dict | None = None,
) -> tuple[Path, Path, int]:
    """
    Generate ``keypoints_3d.csv`` + ``keypoints_3d.meta.json`` in
    ``session_dir`` and return ``(csv_path, meta_path, num_rows)``.
    """
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    csv_path = session_dir / CSV_FILENAME
    meta_path = session_dir / META_FILENAME

    rows = write_keypoints_csv(csv_path, recording_keypoints, num_keypoints)
    meta = build_meta(
        session_dir,
        calibrated_cameras=calibrated_cameras,
        extrinsic_session_id=extrinsic_session_id,
        extrinsic_calibrated_at=extrinsic_calibrated_at,
        model_backend=model_backend,
        model_name=model_name,
        num_keypoints=num_keypoints,
        session_id=session_id,
        intrinsics_db_path=intrinsics_db_path,
        extra=extra_meta,
    )
    meta["csv_row_count"] = rows
    write_meta(meta_path, meta)
    return csv_path, meta_path, rows


# ----------------------------------------------------------------------------
# Reload helper (the inverse of export_session_csv)
# ----------------------------------------------------------------------------


def load_keypoints_3d_csv(session_dir: Path) -> dict:
    """
    Read ``keypoints_3d.csv`` and ``keypoints_3d.meta.json`` from
    ``session_dir`` and return them as a plain dict suitable for
    analysis or playback.

    Returns
    -------
    dict with keys:
        "meta"             -- contents of keypoints_3d.meta.json (or {})
        "frames"           -- list of {time, sync_index, persons}
                              where ``persons`` is a list of
                              ``{person_index, person_id,
                                 keypoints: list[(x,y,z) | None]}``.
        "num_keypoints"    -- max kp_index+1 observed (defaults to meta value)
    """
    session_dir = Path(session_dir)
    csv_path = session_dir / CSV_FILENAME
    meta_path = session_dir / META_FILENAME

    meta: dict = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    if not csv_path.exists():
        return {"meta": meta, "frames": [], "num_keypoints": meta.get("num_keypoints", 0)}

    num_keypoints = int(meta.get("num_keypoints") or 0)

    # First pass: find max kp index, max person index per sync_index
    # Collect rows grouped by sync_index then person_index.
    frames_acc: dict[int, dict[int, dict]] = {}
    sync_to_time: dict[int, float] = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sync_index = int(row["sync_index"])
                person_index = int(row["person_index"])
                person_id = int(row["person_id"])
                kp_index = int(row["kp_index"])
                t = float(row["time_s"])
            except (KeyError, ValueError):
                continue

            x_s, y_s, z_s = row.get("x", ""), row.get("y", ""), row.get("z", "")
            valid_s = row.get("valid", "0")

            def _f(s: str) -> float:
                if s == "" or s is None:
                    return math.nan
                try:
                    return float(s)
                except ValueError:
                    return math.nan

            x, y, z = _f(x_s), _f(y_s), _f(z_s)
            valid = bool(int(valid_s)) if valid_s.isdigit() else False
            if not valid or not all(math.isfinite(v) for v in (x, y, z)):
                kp = None
            else:
                kp = (x, y, z)

            if kp_index + 1 > num_keypoints:
                num_keypoints = kp_index + 1

            sync_to_time[sync_index] = t
            persons = frames_acc.setdefault(sync_index, {})
            person = persons.setdefault(
                person_index,
                {"person_index": person_index, "person_id": person_id, "keypoints": {}},
            )
            person["keypoints"][kp_index] = kp

    # Materialize per-frame dense lists
    frames: list[dict] = []
    for sync_index in sorted(frames_acc.keys()):
        persons_dict = frames_acc[sync_index]
        persons_list = []
        for p_idx in sorted(persons_dict.keys()):
            person = persons_dict[p_idx]
            kps = [None] * num_keypoints
            for k_idx, val in person["keypoints"].items():
                if 0 <= k_idx < num_keypoints:
                    kps[k_idx] = val
            persons_list.append(
                {
                    "person_index": person["person_index"],
                    "person_id": person["person_id"],
                    "keypoints": kps,
                }
            )
        frames.append(
            {
                "sync_index": sync_index,
                "time": sync_to_time.get(sync_index, 0.0),
                "persons": persons_list,
            }
        )

    return {"meta": meta, "frames": frames, "num_keypoints": num_keypoints}


# ----------------------------------------------------------------------------
# Job descriptors (queued mode)
# ----------------------------------------------------------------------------


def make_job_descriptor(
    session_dir: Path,
    *,
    session_id: int | None,
    extrinsic_session_id: int | None = None,
    extrinsic_calibrated_at: str | None = None,
    model_backend: str | None = None,
    model_name: str | None = None,
    num_keypoints: int = 52,
    calibration_path: str | None = None,
) -> dict:
    """
    Build a JSON-serializable dict describing a queued CSV-export job.
    Stored into ``app_settings.json`` so jobs survive restarts.
    """
    return {
        "session_dir": str(session_dir),
        "session_id": session_id,
        "extrinsic_session_id": extrinsic_session_id,
        "extrinsic_calibrated_at": extrinsic_calibrated_at,
        "model_backend": model_backend,
        "model_name": model_name,
        "num_keypoints": int(num_keypoints),
        "calibration_path": calibration_path,
        "raw_buffer_filename": RAW_FILENAME,
        "queued_at": datetime.now().isoformat(timespec="seconds"),
    }


def iter_jobs(jobs: Iterable[dict]) -> Iterable[dict]:
    """Filter to job dicts whose session_dir + raw buffer are still present."""
    for job in jobs:
        sd = Path(job.get("session_dir", ""))
        raw = sd / job.get("raw_buffer_filename", RAW_FILENAME)
        if sd.is_dir() and raw.exists():
            yield job
