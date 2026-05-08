"""
Headless end-to-end test of the offline post-tracking pipeline.

Goal: reproduce what OfflineProcessingWorker does in src/calimerge/gui/workers.py
against a real recording, on the command line, so we can debug the C-side
tracker fragmentation problem and verify that the Python-side stitcher in
_convert_outputs collapses the fragments back into a small number of distinct
people.

Inputs:
  * Recording folder (copied from <last_project_folder>/workouts/<name>/)
    is duplicated into tests/data/<name>/ on first run.
  * Latest extrinsic calibration is read from extrinsics.db via
    config.load_latest_extrinsic_session and dumped to a temp TOML in the
    CUDA parser format via config.write_cuda_calibration_toml.
  * ONNX models are taken from <data_dir>/models/onnx/, with a fallback to
    <repo>/models/onnx/.

Outputs (written into tests/data/<name>/):
  * output_3d_poses_tracked_person*.csv  — written by the C tracker
  * keypoints_3d.raw.npz                 — long-form raw buffer dump
  * keypoints_3d.npz                     — dense (frames, persons, kps, 3) dump

Final report on stdout includes:
  * # of pre-stitch C-side tracks
  * # of distinct persons after Python-side _stitch_tracks
  * Per-survivor frame coverage and hip-COM trajectory range in metres
  * Total wall time

Run:
  VIRTUAL_ENV= ~/.local/bin/uv run python tests/manual/run_offline_pipeline_on_test_data.py

If anything blocks (missing CUDA DLL, missing ONNX, etc.), prints a single
line beginning with "BLOCKED:" and exits non-zero.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = REPO_ROOT / "tests" / "data"
DEFAULT_RECORDING = "zelda_20260428_151934_fga_horizontal_head_turns"


def _parse_args():
    """Argparse for: positional recording name + optional CLI overrides
    we want to tune at the command line (person confidence is the one
    the user has flagged as suspicious — false-positive snaps at low
    thresholds).
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Headless offline post-tracking pipeline reproducer."
    )
    parser.add_argument(
        "recording", nargs="?", default=DEFAULT_RECORDING,
        help="Recording subfolder name under <last_project_folder>/workouts/. "
             f"Defaults to {DEFAULT_RECORDING!r}.",
    )
    parser.add_argument(
        "--person-confidence", type=float, default=0.1,
        help="YOLO person-detection confidence floor (0.0-1.0). "
             "Default 0.1 matches run_cuda_pipeline; raise to ~0.4 to "
             "kill false-positive snaps onto static objects.",
    )
    parser.add_argument(
        "--max-track-distance", type=float, default=0.5,
        help="C tracker max-match distance (m). Default 0.5 matches "
             "the live tracker.",
    )
    parser.add_argument(
        "--track-patience", type=int, default=60,
        help="C tracker frames-of-grace. Default 60 (~2s @ 30fps).",
    )
    parser.add_argument(
        "--worker",
        choices=("deprecated", "unified"),
        default="unified",
        help=(
            "Which offline worker to drive. 'unified' runs "
            "UnifiedOfflineWorker, which shares the live pipeline's "
            "per-sync primitive (PyTorch / CUDA stream / MPS stream). "
            "'deprecated' runs the legacy OfflineProcessingWorker which "
            "drives pt_main.cpp's batched offline pipeline. Default "
            "'unified'."
        ),
    )
    parser.add_argument(
        "--unified-backend",
        choices=("pytorch", "cuda", "mps"),
        default=None,
        help=(
            "Backend for the unified worker. Defaults to 'cuda' on "
            "Windows, 'mps' on macOS (when available), else 'pytorch'. "
            "Only consulted when --worker=unified."
        ),
    )
    parser.add_argument(
        "--max-syncs",
        type=int,
        default=0,
        help=(
            "Cap the number of sync indices the unified worker "
            "processes. 0 = no limit. Useful for debugging — small "
            "values (e.g. 50) iterate in seconds rather than minutes."
        ),
    )
    parser.add_argument(
        "--extrinsic-session-id",
        type=int,
        default=None,
        help=(
            "Force a specific extrinsic_session id from extrinsics.db, "
            "bypassing the workouts.db lookup + timestamp fallback. "
            "Use when the chronological selection picks the wrong rig "
            "(e.g. the user re-calibrated minutes before this recording "
            "but the BA finished after — its created_at would post-date "
            "the recording even though it's the right calibration)."
        ),
    )
    return parser.parse_args()


_ARGS = _parse_args()
RECORDING_NAME = _ARGS.recording


def _setup_cuda_dll_search_path() -> None:
    """Make sure CUDA / TensorRT / OpenCV DLLs are findable.

    Mirrors the logic in calimerge.tracking.cuda_stream_binding._load_lib —
    needed because cuda_binding.py (used by run_cuda_pipeline) doesn't do
    this itself, and outside the GUI nothing else has set up the path.
    """
    if sys.platform != "win32":
        return
    import os
    dep_dirs = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
        r"C:\TensorRT\lib",
        os.environ.get("OPENCV_PATH", r"C:\OpenCV\opencv\build") + r"\x64\vc16\bin",
    ]
    for dep_dir in dep_dirs:
        if os.path.isdir(dep_dir):
            try:
                os.add_dll_directory(dep_dir)
            except Exception:
                pass
            if dep_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = dep_dir + ";" + os.environ.get("PATH", "")


def _blocked(reason: str) -> int:
    print(f"BLOCKED: {reason}", flush=True)
    return 2


def _find_source_recording() -> Path | None:
    """Locate <last_project_folder>/workouts/<RECORDING_NAME>/ via app settings."""
    from calimerge.config import load_app_settings

    settings = load_app_settings()
    folder = settings.get("last_project_folder")
    if not folder:
        return None
    candidate = Path(folder) / "workouts" / RECORDING_NAME
    if candidate.is_dir():
        return candidate
    # Fallback: walk a few well-known places.
    for root in (
        Path(folder),
        Path("~/Documents/Calimerge").expanduser(),
        Path("~/OneDrive/Documents/calimerge/recordings").expanduser(),
    ):
        for hit in root.rglob(RECORDING_NAME):
            if hit.is_dir():
                return hit
    return None


def _copy_recording_to_test_data(src: Path, dst: Path) -> None:
    """Copy the videos + frame_time_history.csv + camera_mapping.csv.

    Preserves filenames exactly so the offline pipeline's port_{N}_{serial}.mp4
    pattern still matches.
    """
    dst.mkdir(parents=True, exist_ok=True)
    wanted = []
    wanted.extend(sorted(src.glob("port_*.mp4")))
    for name in ("frame_time_history.csv", "camera_mapping.csv"):
        p = src / name
        if p.exists():
            wanted.append(p)
    if not wanted:
        raise FileNotFoundError(
            f"no port_*.mp4 / frame_time_history.csv / camera_mapping.csv "
            f"under {src}"
        )

    for p in wanted:
        out = dst / p.name
        if out.exists() and out.stat().st_size == p.stat().st_size:
            print(f"[setup] already copied: {out.name}", flush=True)
            continue
        print(f"[setup] copying {p.name} -> {out}", flush=True)
        shutil.copy(p, out)


def _resolve_onnx_paths() -> tuple[Path | None, Path | None, str]:
    """Find yolo + vitpose ONNX. Returns (yolo, vitpose, source_label)."""
    from calimerge.config import models_dir

    primary = models_dir() / "onnx"
    legacy = REPO_ROOT / "models" / "onnx"

    for root, label in ((primary, "app_data"), (legacy, "repo")):
        yolo = root / "yolo_v10s.onnx"
        vitpose = root / "vitpose_synthpose.onnx"
        if yolo.exists() and vitpose.exists():
            return yolo, vitpose, f"{label}: {root}"
    return None, None, "missing"


def _count_persons_in_npz(npz_path: Path) -> tuple[int, list[dict]]:
    """Read keypoints_3d.npz and return (n_distinct_people, per_person_stats).

    A person slot is "distinct" if it has at least one frame with at least
    one finite (x, y, z) keypoint.
    """
    if not npz_path.exists():
        return 0, []

    data = np.load(str(npz_path))
    kps = data["keypoints_3d"]   # (n_frames, max_persons, n_kps, 3)
    timestamps = data.get("timestamps")
    if timestamps is None:
        timestamps = np.arange(kps.shape[0], dtype=np.float64)

    n_frames, max_persons, n_kps, _ = kps.shape

    stats = []
    for p in range(max_persons):
        # frame_valid[i] = True if person p has any finite kp in frame i
        finite_per_kp = np.isfinite(kps[:, p, :, :]).all(axis=-1)  # (frames, kps)
        frame_valid = finite_per_kp.any(axis=-1)                    # (frames,)
        if not frame_valid.any():
            continue
        idx_valid = np.where(frame_valid)[0]
        first_i, last_i = int(idx_valid[0]), int(idx_valid[-1])

        # Hip COM = mean of kps 11 (L_Hip) and 12 (R_Hip), fall back to
        # shoulders 5/6 if the hips are NaN that frame.
        def _hip_com_at(i: int) -> np.ndarray | None:
            pts = []
            for k in (11, 12):
                if k < n_kps:
                    p3 = kps[i, p, k, :]
                    if np.all(np.isfinite(p3)):
                        pts.append(p3)
            if not pts:
                for k in (5, 6):
                    if k < n_kps:
                        p3 = kps[i, p, k, :]
                        if np.all(np.isfinite(p3)):
                            pts.append(p3)
            if not pts:
                return None
            return np.mean(np.stack(pts, axis=0), axis=0)

        first_com = _hip_com_at(first_i)
        last_com = _hip_com_at(last_i)

        # Bounding-box of all valid hip COMs across this person's track —
        # this is what tells you whether the person stayed put or moved
        # halfway across the room.
        coms = []
        for i in idx_valid:
            c = _hip_com_at(int(i))
            if c is not None:
                coms.append(c)
        if coms:
            arr = np.stack(coms, axis=0)
            com_min = arr.min(axis=0)
            com_max = arr.max(axis=0)
            com_range = com_max - com_min
        else:
            com_min = com_max = com_range = None

        stats.append({
            "person_index": p,
            "n_valid_frames": int(frame_valid.sum()),
            "first_frame": first_i,
            "last_frame": last_i,
            "first_time_s": float(timestamps[first_i]),
            "last_time_s": float(timestamps[last_i]),
            "first_hip_com": first_com,
            "last_hip_com": last_com,
            "com_min": com_min,
            "com_max": com_max,
            "com_range": com_range,
        })

    return len(stats), stats


def _format_vec3(v: np.ndarray | None) -> str:
    if v is None:
        return "n/a"
    return f"[{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]"


def main() -> int:
    _setup_cuda_dll_search_path()
    t0 = time.time()

    # ── 1. Locate source recording + copy into tests/data ──────────────
    print("=" * 70)
    print("offline pipeline test on real recording")
    print("=" * 70)

    test_data_dir = TEST_DATA_DIR / RECORDING_NAME
    already_present = (
        test_data_dir.is_dir()
        and any(test_data_dir.glob("port_*.mp4"))
        and (test_data_dir / "frame_time_history.csv").exists()
    )

    if already_present:
        print(f"[setup] using existing test data at: {test_data_dir}")
    else:
        try:
            src = _find_source_recording()
        except Exception as e:
            return _blocked(f"finding source recording: {e}\n{traceback.format_exc()}")

        if src is None:
            return _blocked(
                f"could not find recording {RECORDING_NAME!r} under "
                f"<last_project_folder>/workouts/ or at {test_data_dir}. "
                "Either drop the recording into tests/data/<name>/ or set "
                "last_project_folder via the GUI's project picker."
            )

        print(f"[setup] source recording: {src}")
        print(f"[setup] test data dir:    {test_data_dir}")
        try:
            _copy_recording_to_test_data(src, test_data_dir)
        except Exception as e:
            return _blocked(f"copying recording: {e}")

    # ── 2. Discover videos in the test data dir ────────────────────────
    port_to_video: dict[int, Path] = {}
    for p in sorted(test_data_dir.glob("port_*.mp4")):
        # Filename forms: port_0.mp4 OR port_0_<serial>-...mp4
        stem = p.stem  # 'port_0_6-3023cdee-0-0000'
        try:
            after = stem[len("port_"):]
            port_str = after.split("_", 1)[0]
            port = int(port_str)
        except Exception:
            print(f"[warn] cannot parse port from {p.name}, skipping")
            continue
        port_to_video[port] = p

    if not port_to_video:
        return _blocked(f"no port_*.mp4 found under {test_data_dir}")
    print(f"[setup] videos: {dict((k, v.name) for k, v in port_to_video.items())}")

    frame_time_csv = test_data_dir / "frame_time_history.csv"
    if not frame_time_csv.exists():
        return _blocked(f"missing frame_time_history.csv in {test_data_dir}")

    # ── 3. Load the extrinsic that was ACTIVE when this recording was
    #       made — NOT the newest extrinsic, since the user may have
    #       recalibrated for a different camera placement after this
    #       trial was recorded. Sources, in priority order:
    #         (a) The session row in workouts.db (matched by
    #             recording_path) carries `extrinsic_session_id` —
    #             that's the authoritative pointer.
    #         (b) Otherwise, parse the YYYYMMDD_HHMMSS timestamp out of
    #             the recording folder name and pick the newest
    #             extrinsic_session whose created_at predates it.
    #         (c) Last resort: load_latest_extrinsic_session — clearly
    #             wrong if (a) or (b) would have produced a different
    #             answer, so we print a loud warning.
    try:
        from calimerge.config import (
            load_extrinsic_session,
            load_latest_extrinsic_session,
            load_view_transform,
            write_cuda_calibration_toml,
            list_extrinsic_sessions,
            extrinsics_db_path,
            workouts_db_path,
        )
    except Exception as e:
        return _blocked(f"importing config: {e}")

    db_p = extrinsics_db_path()
    print(f"[calib] extrinsics db: {db_p} (exists={db_p.exists()})")

    sess_id = None
    created_at = None
    calibrated_cams = None
    chosen_via = None

    # (-) explicit override via --extrinsic-session-id
    if _ARGS.extrinsic_session_id is not None:
        forced = int(_ARGS.extrinsic_session_id)
        loaded = load_extrinsic_session(forced)
        if loaded is None:
            return _blocked(
                f"--extrinsic-session-id {forced} not found in extrinsics.db"
            )
        created_at, calibrated_cams = loaded
        sess_id = forced
        chosen_via = f"--extrinsic-session-id {forced} (forced)"

    # (a) workouts.db session row
    try:
        import sqlite3
        wdb = workouts_db_path()
        if wdb.exists():
            conn = sqlite3.connect(str(wdb))
            try:
                row = conn.execute(
                    "SELECT extrinsic_session_id "
                    "FROM sessions WHERE recording_path LIKE ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (f"%{RECORDING_NAME}",),
                ).fetchone()
            finally:
                conn.close()
            if row is not None and row[0] is not None:
                ext_sid = int(row[0])
                loaded = load_extrinsic_session(ext_sid)
                if loaded is not None:
                    created_at, calibrated_cams = loaded
                    sess_id = ext_sid
                    chosen_via = "workouts.db sessions.extrinsic_session_id"
    except Exception as e:
        print(f"[calib] note: workouts.db lookup raised: {e}")

    # (b) timestamp-before-recording fallback
    if calibrated_cams is None:
        try:
            stamp = RECORDING_NAME.split("_", 2)
            # ['username','YYYYMMDD','HHMMSS_<rest>'] OR
            # legacy ['YYYYMMDD','HHMMSS_<rest>'].
            date_str = None
            time_str = None
            if len(stamp) >= 2 and len(stamp[0]) >= 8 and stamp[0].isdigit():
                date_str = stamp[0]
                time_str = stamp[1].split("_", 1)[0]
            elif len(stamp) >= 3 and stamp[1].isdigit() and len(stamp[1]) == 8:
                date_str = stamp[1]
                time_str = stamp[2].split("_", 1)[0]
            if date_str and time_str and len(time_str) >= 6:
                rec_iso = (
                    f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} "
                    f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                )
                print(f"[calib] recording timestamp parsed as {rec_iso!r}")
                # list_extrinsic_sessions returns newest-first.
                for sess in list_extrinsic_sessions():
                    if str(sess["created_at"]) <= rec_iso:
                        loaded = load_extrinsic_session(int(sess["id"]))
                        if loaded is not None:
                            created_at, calibrated_cams = loaded
                            sess_id = int(sess["id"])
                            chosen_via = (
                                f"newest extrinsic_session before "
                                f"{rec_iso}"
                            )
                            break
        except Exception as e:
            print(f"[calib] note: timestamp fallback raised: {e}")

    # (c) latest, with a warning
    if calibrated_cams is None:
        latest = load_latest_extrinsic_session()
        if latest is None:
            return _blocked("no extrinsic calibration in extrinsics.db")
        sess_id, created_at, calibrated_cams = latest
        chosen_via = "load_latest_extrinsic_session (LAST RESORT)"
        print(
            f"[calib] WARNING: falling back to LATEST extrinsic. "
            f"If the user recalibrated after recording for a different "
            f"rig setup, this calibration WILL produce wrong 3D points."
        )

    print(f"[calib] session id={sess_id} created={created_at} "
          f"cameras={sorted(calibrated_cams.keys())}")
    print(f"[calib] chosen via: {chosen_via}")

    # Reduce calibrated cams to those whose ports we have videos for, OR,
    # if there's no overlap on port number, remap by serial-number from
    # camera_mapping.csv.
    cal_ports = set(calibrated_cams.keys())
    vid_ports = set(port_to_video.keys())
    missing = vid_ports - cal_ports
    if missing:
        print(f"[calib] WARNING: video ports not in calibration: {missing}")
        # Try to remap: look up serial in camera_mapping, find calibrated
        # camera with same serial, re-key to the video port.
        mapping_csv = test_data_dir / "camera_mapping.csv"
        port_to_serial = {}
        if mapping_csv.exists():
            import csv as _csv
            with open(mapping_csv, "r", newline="") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    port_to_serial[int(row["port"])] = row["serial_number"]
        serial_to_cal = {c.serial_number: c for c in calibrated_cams.values()}
        remapped: dict[int, object] = {}
        for vp in vid_ports:
            ser = port_to_serial.get(vp)
            if ser and ser in serial_to_cal:
                base = serial_to_cal[ser]
                # Rebuild CalibratedCamera with the video-side port.
                from calimerge.types import CalibratedCamera
                remapped[vp] = CalibratedCamera(
                    serial_number=base.serial_number,
                    port=vp,
                    intrinsics=base.intrinsics,
                    extrinsics=base.extrinsics,
                )
            elif vp in calibrated_cams:
                remapped[vp] = calibrated_cams[vp]
        if remapped:
            calibrated_cams = remapped
            print(f"[calib] remapped by serial -> ports {sorted(calibrated_cams.keys())}")

    # ── 3a-bis. Normalise intrinsics to match the recorded video size. ──
    # The GUI's `_start_offline_processing` does this before constructing
    # the worker — the test runner has to do it too or the C-side
    # triangulation projects against the wrong scale and every 3D point
    # comes out wrong by a scale factor (we saw 2 m median ankle delta vs
    # the online output before adding this step).
    try:
        import cv2 as _cv2
        any_video = next(iter(port_to_video.values()))
        cap = _cv2.VideoCapture(str(any_video))
        target_w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
        target_h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if target_w > 0 and target_h > 0:
            from calimerge.types import scale_intrinsics
            import dataclasses as _dc
            target_res = (target_w, target_h)
            normalised: dict = {}
            for port, cc in calibrated_cams.items():
                if tuple(cc.intrinsics.resolution) == target_res:
                    normalised[port] = cc
                else:
                    scaled = scale_intrinsics(cc.intrinsics, target_res)
                    normalised[port] = _dc.replace(cc, intrinsics=scaled)
                    print(
                        f"[calib] port {port} intrinsics rescaled "
                        f"{tuple(cc.intrinsics.resolution)} -> {target_res}"
                    )
            calibrated_cams = normalised
    except Exception as e:
        print(f"[calib] WARNING: intrinsic normalisation failed: {e}")

    cuda_cal_path = Path(tempfile.gettempdir()) / "calimerge_offline_test_cal.toml"
    write_cuda_calibration_toml(calibrated_cams, cuda_cal_path)
    print(f"[calib] CUDA calibration TOML: {cuda_cal_path}")

    # ── 3b. Load view transform (rotate-to-human + zero-at-ankle) ──
    #
    # Source priority:
    #   (1) The ORIGINAL recording's keypoints_3d.npz (if present in
    #       <last_project_folder>/workouts/<name>/) — this is the
    #       transform that was active at record-time and is baked into
    #       the online output. Using it makes online vs offline an
    #       apples-to-apples comparison.
    #   (2) The current DB preset for vitpose. Fallback when the user
    #       only has the videos (e.g. paused-tracking trial).
    #   (3) None → camera frame (clearly broken, but caller can opt in
    #       deliberately by deleting both).
    view_R = None
    view_t = None
    view_source = "none"
    try:
        from calimerge.config import load_app_settings
        settings = load_app_settings()
        folder = settings.get("last_project_folder")
        candidates = []
        if folder:
            candidates.append(
                Path(folder) / "workouts" / RECORDING_NAME / "keypoints_3d.npz"
            )
        for online_npz in candidates:
            if online_npz.exists():
                with np.load(str(online_npz)) as on:
                    if "view_transform_R" in on.files and "view_transform_t" in on.files:
                        candidate_R = np.array(on["view_transform_R"])
                        candidate_t = np.array(on["view_transform_t"])
                        # Identity transform = recording was made
                        # without zero pressed, so it's effectively
                        # camera frame — fall through to DB preset.
                        if not (np.allclose(candidate_R, np.eye(3))
                                and np.allclose(candidate_t, 0.0)):
                            view_R = candidate_R
                            view_t = candidate_t
                            view_source = f"online npz: {online_npz}"
                            break
    except Exception as e:
        print(f"[calib] WARNING: probing online npz failed: {e}")

    if view_R is None:
        try:
            # Prefer a preset tagged with the calibration we just chose;
            # fall back to untagged / most-recent presets per
            # load_view_transform's resolution priority.
            preset = load_view_transform(
                "synthpose", extrinsic_session_id=sess_id,
            )
        except Exception as e:
            preset = None
            print(f"[calib] WARNING: load_view_transform raised: {e}")
        if preset is not None:
            view_R, view_t, _ = preset
            view_source = (
                f"view_transforms.db (model=synthpose, "
                f"session_id={sess_id})"
            )

    if view_R is not None:
        print(f"[calib] view transform source: {view_source}")
        print(f"[calib]   R row 0: {view_R[0].tolist()}")
        print(f"[calib]   R row 1: {view_R[1].tolist()}")
        print(f"[calib]   R row 2: {view_R[2].tolist()}")
        print(f"[calib]   t:       {view_t.tolist()}")
    else:
        print(
            "[calib] WARNING: no view transform found. Output will be in "
            "camera frame (z huge, x/y rotated)."
        )

    # ── 4. Resolve ONNX models ─────────────────────────────────────────
    yolo_onnx, vitpose_onnx, onnx_src = _resolve_onnx_paths()
    if yolo_onnx is None or vitpose_onnx is None:
        return _blocked(
            f"could not find yolo_v10s.onnx + vitpose_synthpose.onnx under "
            f"<data_dir>/models/onnx/ or <repo>/models/onnx/. "
            f"Download them first."
        )
    print(f"[onnx] yolo:    {yolo_onnx}")
    print(f"[onnx] vitpose: {vitpose_onnx}")
    print(f"[onnx] source:  {onnx_src}")

    # ── 5. Pick worker + (deprecated path) check CUDA dll availability ──
    last_step = ""

    def _on_prog(step: str, frac: float) -> None:
        nonlocal last_step
        if step != last_step:
            last_step = step
            print(f"[prog] {step}  {frac:.2f}", flush=True)

    def _on_log(msg: str) -> None:
        print(f"[c   ] {msg}", flush=True)

    pipe_secs = 0.0
    per_person_csvs: list[Path] = []

    if _ARGS.worker == "deprecated":
        try:
            from calimerge.tracking.cuda_binding import (
                is_available, run_cuda_pipeline,
            )
        except Exception as e:
            return _blocked(f"importing cuda_binding: {e}")
        if not is_available():
            return _blocked(
                "calimerge_cuda.dll not available. "
                "Build it via src/cuda_pipeline/build_cuda_win32.bat first."
            )

        # ── 6a. Run the deprecated C-side batched pipeline ────────────
        print("=" * 70)
        print(
            "[run ] starting CUDA batched pipeline "
            "(this may take 30-60s on first run)"
        )
        print("=" * 70)
        print(
            f"[run ] params: person_confidence={_ARGS.person_confidence:.2f}  "
            f"max_track_distance={_ARGS.max_track_distance:.2f}  "
            f"track_patience={_ARGS.track_patience}"
        )
        t_pipe = time.time()
        try:
            run_cuda_pipeline(
                video_paths=port_to_video,
                calibration_toml=cuda_cal_path,
                frame_time_csv=frame_time_csv,
                output_path=test_data_dir,
                yolo_onnx=yolo_onnx,
                vitpose_onnx=vitpose_onnx,
                batch_size=8,
                person_confidence=_ARGS.person_confidence,
                max_track_distance=_ARGS.max_track_distance,
                track_patience=_ARGS.track_patience,
                progress_callback=_on_prog,
                log_callback=_on_log,
            )
        except Exception as e:
            print(traceback.format_exc())
            return _blocked(f"run_cuda_pipeline raised: {e}")
        pipe_secs = time.time() - t_pipe
        print(f"[run ] CUDA pipeline finished in {pipe_secs:.1f}s")

        # ── 7a. Count C-side per-track CSVs (pre-stitch fragmentation) ──
        per_person_csvs = sorted(
            list(test_data_dir.glob("output_3d_poses_tracked.csv_person*.csv"))
            + list(test_data_dir.glob("output_3d_poses_tracked_person*.csv"))
        )
        print(f"[run ] C tracker emitted {len(per_person_csvs)} per-person CSVs:")
        for p in per_person_csvs:
            print(f"       {p.name}")

        # ── 8a. Drive the deprecated worker's _convert_outputs ────────
        print("=" * 70)
        print("[run ] running OfflineProcessingWorker._convert_outputs "
              "(stitcher + npz)")
        print("=" * 70)
        try:
            from calimerge.gui.workers import OfflineProcessingWorker
        except Exception as e:
            return _blocked(f"importing OfflineProcessingWorker: {e}")

        worker = OfflineProcessingWorker(
            session_dir=test_data_dir,
            cameras=calibrated_cams,
            port_to_video=port_to_video,
            frame_time_csv=frame_time_csv,
            batch_size=8,
            view_rotation=view_R,
            view_translation=view_t,
            max_track_distance=_ARGS.max_track_distance,
            track_patience=_ARGS.track_patience,
        )
        try:
            worker.log_message.connect(
                lambda m: print(f"[work] {m}", flush=True)
            )
        except Exception:
            pass
        try:
            worker._convert_outputs()
        except Exception as e:
            print(traceback.format_exc())
            return _blocked(f"_convert_outputs raised: {e}")

    else:
        # ── 6b/7b/8b. Drive the unified worker end-to-end ─────────────
        # Pick the backend. Default: CUDA on Windows when available, MPS
        # on macOS when available, else PyTorch.
        backend = _ARGS.unified_backend
        if backend is None:
            if sys.platform == "darwin":
                try:
                    from calimerge.tracking.mps_stream_binding import (
                        is_available as _mps_avail,
                    )
                    backend = "mps" if _mps_avail() else "pytorch"
                except Exception:
                    backend = "pytorch"
            else:
                try:
                    from calimerge.tracking.cuda_stream_binding import (
                        is_available as _cuda_avail,
                    )
                    backend = "cuda" if _cuda_avail() else "pytorch"
                except Exception:
                    backend = "pytorch"

        print("=" * 70)
        print(f"[run ] starting unified offline pipeline (backend={backend})")
        print("=" * 70)
        print(
            f"[run ] params: person_confidence={_ARGS.person_confidence:.2f}  "
            f"max_track_distance={_ARGS.max_track_distance:.2f}  "
            f"track_patience={_ARGS.track_patience}"
        )

        try:
            from calimerge.gui.unified_offline_worker import UnifiedOfflineWorker
        except Exception as e:
            return _blocked(f"importing UnifiedOfflineWorker: {e}")

        worker = UnifiedOfflineWorker(
            session_dir=test_data_dir,
            cameras=calibrated_cams,
            port_to_video=port_to_video,
            frame_time_csv=frame_time_csv,
            backend=backend,
            view_rotation=view_R,
            view_translation=view_t,
            max_track_distance=_ARGS.max_track_distance,
            track_patience=_ARGS.track_patience,
            person_confidence=_ARGS.person_confidence,
        )
        # Debugging knob: stop after N sync indices. Implemented by
        # monkey-patching the worker's frame_time_csv reader so it sees
        # only the first N rows. Cleanest non-invasive way to cap.
        if _ARGS.max_syncs > 0:
            _orig = worker._read_sync_to_ports
            cap = int(_ARGS.max_syncs)
            def _capped_read(_orig=_orig, cap=cap):
                full = _orig()
                keys = sorted(full.keys())[:cap]
                return {k: full[k] for k in keys}
            worker._read_sync_to_ports = _capped_read
            print(f"[run ] DEBUG: capping at {cap} sync indices", flush=True)
        try:
            worker.log_message.connect(
                lambda m: print(f"[work] {m}", flush=True)
            )
            worker.progress.connect(_on_prog)
        except Exception:
            pass

        # Drive run() synchronously rather than start()ing the QThread —
        # we don't have an event loop here and run() does not require one
        # because the signal connections above invoke their slots
        # directly when emitter and slot share a thread.
        t_pipe = time.time()
        try:
            worker.run()
        except Exception as e:
            print(traceback.format_exc())
            return _blocked(f"UnifiedOfflineWorker.run raised: {e}")
        pipe_secs = time.time() - t_pipe
        print(f"[run ] unified pipeline finished in {pipe_secs:.1f}s")

        # The unified path doesn't write per-track CSVs; it goes straight
        # to keypoints_3d.npz. Pre-stitch track count is therefore the
        # number of distinct tracker ids the streaming primitive emitted
        # before stitching — read from the worker's log lines for
        # reporting. Leave the CSV list empty.

    # ── 9. Inspect outputs ─────────────────────────────────────────────
    raw_npz = test_data_dir / "keypoints_3d.raw.npz"
    npz = test_data_dir / "keypoints_3d.npz"
    print(f"[done] raw npz: {raw_npz} (exists={raw_npz.exists()})")
    print(f"[done] dense npz: {npz} (exists={npz.exists()})")

    n_distinct, per_person = _count_persons_in_npz(npz)
    n_pre_stitch = len(per_person_csvs)

    total_secs = time.time() - t0

    # ── 10. Final report ───────────────────────────────────────────────
    print()
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"recording dir:        {test_data_dir}")
    print(f"calibration session:  id={sess_id}  created_at={created_at}")
    print(f"yolo onnx:            {yolo_onnx}")
    print(f"vitpose onnx:         {vitpose_onnx}")
    print(f"pre-stitch tracks:    {n_pre_stitch}    (output_3d_poses_tracked_person*.csv)")
    print(f"post-stitch persons:  {n_distinct}    <-- key fragmentation metric")
    print()
    if per_person:
        for s in per_person:
            print(
                f"  person[{s['person_index']}]: "
                f"{s['n_valid_frames']} valid frames "
                f"(frame {s['first_frame']}..{s['last_frame']}, "
                f"t {s['first_time_s']:.2f}..{s['last_time_s']:.2f}s)"
            )
            print(f"    first hip-COM: {_format_vec3(s['first_hip_com'])}")
            print(f"    last  hip-COM: {_format_vec3(s['last_hip_com'])}")
            if s["com_range"] is not None:
                print(
                    f"    COM bbox:      min={_format_vec3(s['com_min'])} "
                    f"max={_format_vec3(s['com_max'])} "
                    f"range={_format_vec3(s['com_range'])} (m)"
                )
    else:
        print("  (no surviving tracks)")
    print()
    print(f"total wall time:      {total_secs:.1f}s "
          f"(of which CUDA pipeline: {pipe_secs:.1f}s)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
