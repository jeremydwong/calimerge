"""
Headless end-to-end test of the unified offline pipeline.

Drives UnifiedOfflineWorker — the same code path that the GUI's
"Generate 3D-npz post-trial" button uses — against a real recording,
on the command line.

Inputs:
  * Recording folder (copied from <last_project_folder>/workouts/<name>/)
    is duplicated into tests/data/<name>/ on first run.
  * Extrinsic calibration from extrinsics.db (matched by recording
    timestamp or workouts.db session row).
  * Models are resolved by the worker itself (CoreML on macOS, ONNX +
    TensorRT on Windows, PyTorch everywhere).

Outputs (written into tests/data/<name>/):
  * keypoints_3d.raw.npz  — variable-shape lossless archive
  * keypoints_3d.npz      — dense (frames, max_persons, kps, 3)

Final report on stdout includes:
  * # of distinct persons in the dense npz
  * Per-person frame coverage and hip-COM trajectory range in metres
  * Total wall time

Run:
  VIRTUAL_ENV= ~/.local/bin/uv run python tests/manual/run_offline_pipeline_on_test_data.py

If anything blocks (missing native lib, missing calibration, etc.),
prints a single line beginning with "BLOCKED:" and exits non-zero.
"""

from __future__ import annotations

import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = REPO_ROOT / "tests" / "data"
DEFAULT_RECORDING = "zelda_20260428_151934_fga_horizontal_head_turns"


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Headless offline pipeline test (same path as GUI)."
    )
    parser.add_argument(
        "recording", nargs="?", default=DEFAULT_RECORDING,
        help="Recording subfolder name under <last_project_folder>/workouts/. "
             f"Defaults to {DEFAULT_RECORDING!r}.",
    )
    parser.add_argument(
        "--person-confidence", type=float, default=0.5,
        help="YOLO person-detection confidence floor (0.0-1.0). "
             "Default 0.5 matches the GUI live slider default.",
    )
    parser.add_argument(
        "--max-track-distance", type=float, default=0.5,
        help="Tracker max-match distance (m). Default 0.5.",
    )
    parser.add_argument(
        "--track-patience", type=int, default=60,
        help="Tracker frames-of-grace. Default 60 (~2s @ 30fps).",
    )
    parser.add_argument(
        "--backend",
        choices=("pytorch", "cuda", "mps"),
        default=None,
        help=(
            "Detection backend. Defaults to 'mps' on macOS (when the "
            "native lib is available), 'cuda' on Windows, else 'pytorch'."
        ),
    )
    parser.add_argument(
        "--max-syncs",
        type=int,
        default=0,
        help=(
            "Cap the number of sync indices processed. "
            "0 = no limit. Small values (e.g. 50) iterate in seconds."
        ),
    )
    parser.add_argument(
        "--extrinsic-session-id",
        type=int,
        default=None,
        help=(
            "Force a specific extrinsic_session id from extrinsics.db, "
            "bypassing the workouts.db lookup + timestamp fallback."
        ),
    )
    return parser.parse_args()


_ARGS = _parse_args()
RECORDING_NAME = _ARGS.recording


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
    """Copy the videos + frame_time_history.csv + camera_mapping.csv."""
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
        finite_per_kp = np.isfinite(kps[:, p, :, :]).all(axis=-1)  # (frames, kps)
        frame_valid = finite_per_kp.any(axis=-1)                    # (frames,)
        if not frame_valid.any():
            continue
        idx_valid = np.where(frame_valid)[0]
        first_i, last_i = int(idx_valid[0]), int(idx_valid[-1])

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


# SynthPose-52 skeleton connectivity (same as PoseDetectionWorker._SKELETON).
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 17), (17, 5), (17, 6), (17, 48),
    (5, 19), (6, 18),
    (5, 7), (7, 9), (7, 21), (7, 23), (9, 25), (9, 27),
    (6, 8), (8, 10), (8, 20), (8, 22), (10, 24), (10, 26),
    (5, 11), (6, 12), (11, 12),
    (48, 51), (51, 50), (50, 49), (49, 29), (49, 28), (29, 31), (28, 30),
    (11, 13), (13, 15), (13, 33), (13, 35), (15, 37), (15, 39),
    (12, 14), (14, 16), (14, 32), (14, 34), (16, 36), (16, 38),
    (15, 46), (15, 41), (41, 43), (43, 45),
    (16, 47), (16, 40), (40, 42), (42, 44),
    (5, 6),
]
L_ANKLE, R_ANKLE = 15, 16


def _generate_annotated_video(
    npz_path: Path,
    video_path: Path,
    output_path: Path,
    camera,
    view_R: np.ndarray | None,
    view_t: np.ndarray | None,
) -> None:
    """Render skeleton overlay on one camera's video from the 3D npz."""
    import cv2
    from calimerge.types import compute_projection_matrix

    data = np.load(str(npz_path))
    kps = data["keypoints_3d"]  # (n_frames, max_persons, n_kps, 3)
    n_frames, max_persons, n_kps, _ = kps.shape

    P = compute_projection_matrix(camera)  # 3x4

    # Precompute inverse view transform (view frame → world frame).
    if view_R is not None and view_t is not None:
        R_inv = view_R.T
        t_neg = -R_inv @ view_t
    else:
        R_inv = np.eye(3)
        t_neg = np.zeros(3)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    COLORS = [(80, 200, 120), (100, 160, 255), (255, 180, 80), (220, 100, 220)]

    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        for pi in range(max_persons):
            person_kps = kps[fi, pi]  # (n_kps, 3)
            valid = np.isfinite(person_kps).all(axis=1)
            if not valid.any():
                continue

            # Project all keypoints: view → world → pixel
            world_pts = (R_inv @ person_kps.T).T + t_neg  # (n_kps, 3)
            hom = P @ np.hstack([world_pts, np.ones((n_kps, 1))]).T  # (3, n_kps)
            px = np.zeros((n_kps, 2))
            for k in range(n_kps):
                if valid[k] and hom[2, k] > 0:
                    px[k] = hom[:2, k] / hom[2, k]
                else:
                    valid[k] = False

            color = COLORS[pi % len(COLORS)]
            kp_color = tuple(min(255, int(c * 1.3)) for c in color)

            for i, j in _SKELETON:
                if i >= n_kps or j >= n_kps:
                    continue
                if not (valid[i] and valid[j]):
                    continue
                pt1 = (int(px[i, 0]), int(px[i, 1]))
                pt2 = (int(px[j, 0]), int(px[j, 1]))
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

            for k in range(n_kps):
                if not valid[k]:
                    continue
                pt = (int(px[k, 0]), int(px[k, 1]))
                cv2.circle(frame, pt, 3, kp_color, -1, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()


def _generate_ankle_plot(
    npz_path: Path,
    output_path: Path,
) -> None:
    """Plot ankle x/y/z over time and save to PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(str(npz_path))
    kps = data["keypoints_3d"]
    timestamps = data["timestamps"]
    counts = data["person_count"]
    primary = data.get("primary_person_index",
                       np.zeros(len(timestamps), dtype=np.int32))

    n_frames = kps.shape[0]
    left_ankle = np.full((n_frames, 3), np.nan, dtype=np.float32)
    right_ankle = np.full((n_frames, 3), np.nan, dtype=np.float32)
    for fi in range(n_frames):
        if counts[fi] == 0:
            continue
        pi = int(primary[fi]) if primary[fi] < counts[fi] else 0
        left_ankle[fi] = kps[fi, pi, L_ANKLE]
        right_ankle[fi] = kps[fi, pi, R_ANKLE]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, axis_idx, axis_label in zip(axes, range(3), ("x (m)", "y (m)", "z (m)")):
        ax.plot(timestamps, left_ankle[:, axis_idx], color="#5099ff", label="L_Ankle")
        ax.plot(timestamps, right_ankle[:, axis_idx], color="#ff5050", label="R_Ankle")
        ax.set_ylabel(axis_label)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Ankle position over time (primary person)")
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=120)
    plt.close(fig)


def main() -> int:
    t0 = time.time()

    # ── 1. Locate source recording + copy into tests/data ──────────────
    print("=" * 70)
    print("offline pipeline test (unified worker — same path as GUI)")
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
        stem = p.stem
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

    # ── 3. Load extrinsic calibration ──────────────────────────────────
    try:
        from calimerge.config import (
            load_extrinsic_session,
            load_latest_extrinsic_session,
            load_view_transform,
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

    # Parse recording timestamp from folder name.
    rec_iso = None
    _stamp = RECORDING_NAME.split("_", 2)
    _date_str = _time_str = None
    if len(_stamp) >= 2 and len(_stamp[0]) >= 8 and _stamp[0].isdigit():
        _date_str = _stamp[0]
        _time_str = _stamp[1].split("_", 1)[0]
    elif len(_stamp) >= 3 and _stamp[1].isdigit() and len(_stamp[1]) == 8:
        _date_str = _stamp[1]
        _time_str = _stamp[2].split("_", 1)[0]
    if _date_str and _time_str and len(_time_str) >= 6:
        rec_iso = (
            f"{_date_str[:4]}-{_date_str[4:6]}-{_date_str[6:8]} "
            f"{_time_str[:2]}:{_time_str[2:4]}:{_time_str[4:6]}"
        )
        print(f"[calib] recording timestamp parsed as {rec_iso!r}")

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
    if calibrated_cams is None:
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
    if calibrated_cams is None and rec_iso is not None:
        try:
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

    # Remap calibrated cams by serial number if port numbers don't match.
    cal_ports = set(calibrated_cams.keys())
    vid_ports = set(port_to_video.keys())
    missing = vid_ports - cal_ports
    if missing:
        print(f"[calib] WARNING: video ports not in calibration: {missing}")
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

    # ── 3a. Normalise intrinsics to match the recorded video size. ─────
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

    # ── 3b. Load view transform ────────────────────────────────────────
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
            preset = load_view_transform(
                "synthpose", before=rec_iso,
            )
        except Exception as e:
            preset = None
            print(f"[calib] WARNING: load_view_transform raised: {e}")
        if preset is not None:
            view_R, view_t, _ = preset
            view_source = (
                f"view_transforms.db (model=synthpose, "
                f"before={rec_iso!r})"
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

    # ── 4. Pick backend ────────────────────────────────────────────────
    backend = _ARGS.backend
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

    # ── 5. Run the unified offline worker ──────────────────────────────
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

    last_step = ""
    t_milestones: dict[str, float] = {}

    def _on_prog(step: str, frac: float) -> None:
        nonlocal last_step
        if step != last_step:
            last_step = step
            t_milestones.setdefault(step, time.time())
            print(f"[prog] {step}  {frac:.2f}", flush=True)

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
        extrinsic_session_id=sess_id,
        extrinsic_created_at=str(created_at) if created_at else None,
    )

    if _ARGS.max_syncs > 0:
        _orig = worker._read_sync_to_ports
        cap = int(_ARGS.max_syncs)
        def _capped_read(_orig=_orig, cap=cap):
            full = _orig()
            keys = sorted(full.keys())[:cap]
            return {k: full[k] for k in keys}
        worker._read_sync_to_ports = _capped_read
        print(f"[run ] capping at {cap} sync indices", flush=True)

    try:
        worker.log_message.connect(
            lambda m: print(f"[work] {m}", flush=True)
        )
        worker.progress.connect(_on_prog)
    except Exception:
        pass

    t_pipe = time.time()
    try:
        worker.run()
    except Exception as e:
        print(traceback.format_exc())
        return _blocked(f"UnifiedOfflineWorker.run raised: {e}")
    t_pipe_end = time.time()
    pipe_secs = t_pipe_end - t_pipe

    t_detect = t_milestones.get("detect+triangulate", t_pipe)
    t_retrack = t_milestones.get("re-tracking", t_pipe_end)
    load_secs = t_detect - t_pipe
    infer_secs = t_retrack - t_detect
    post_secs = t_pipe_end - t_retrack
    # Estimate sync count for fps calculation — refined after npz load.
    n_syncs_actual = _ARGS.max_syncs if _ARGS.max_syncs > 0 else 0

    print(f"[run ] pipeline finished in {pipe_secs:.1f}s")
    print(f"[time]   model loading:      {load_secs:.1f}s")
    print(f"[time]   detect+triangulate: {infer_secs:.1f}s")
    print(f"[time]   retrack+stitch+io:  {post_secs:.1f}s")

    # ── 6. Inspect npz outputs ─────────────────────────────────────────
    raw_npz = test_data_dir / "keypoints_3d.raw.npz"
    npz = test_data_dir / "keypoints_3d.npz"
    print(f"[done] raw npz: {raw_npz} (exists={raw_npz.exists()})")
    print(f"[done] dense npz: {npz} (exists={npz.exists()})")

    n_distinct, per_person = _count_persons_in_npz(npz)

    if npz.exists() and n_syncs_actual == 0:
        d = np.load(str(npz))
        n_syncs_actual = d["keypoints_3d"].shape[0]

    # ── 6a. Generate annotated video + ankle plot ──────────────────────
    first_port = sorted(port_to_video.keys())[0]
    annotated_video = test_data_dir / f"annotated_port_{first_port}.mp4"
    ankle_plot = test_data_dir / "ankle_plot.png"

    if npz.exists():
        print("[viz ] generating annotated video...", flush=True)
        try:
            _generate_annotated_video(
                npz, port_to_video[first_port], annotated_video,
                calibrated_cams[first_port], view_R, view_t,
            )
            print(f"[viz ] wrote {annotated_video}")
        except Exception as e:
            print(f"[viz ] annotated video failed: {e}")

        print("[viz ] generating ankle plot...", flush=True)
        try:
            _generate_ankle_plot(npz, ankle_plot)
            print(f"[viz ] wrote {ankle_plot}")
        except Exception as e:
            print(f"[viz ] ankle plot failed: {e}")

    total_secs = time.time() - t0

    # ── 7. Final report ────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"recording dir:        {test_data_dir}")
    print(f"calibration session:  id={sess_id}  created_at={created_at}")
    print(f"backend:              {backend}")
    print(f"distinct persons:     {n_distinct}    <-- key fragmentation metric")
    print(f"annotated video:      {annotated_video} (exists={annotated_video.exists()})")
    print(f"ankle plot:           {ankle_plot} (exists={ankle_plot.exists()})")
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
    print(f"total wall time:      {total_secs:.1f}s")
    print(f"  model loading:      {load_secs:.1f}s")
    print(f"  detect+triangulate: {infer_secs:.1f}s  "
          f"({n_syncs_actual / max(0.01, infer_secs):.1f} syncs/s)")
    print(f"  retrack+stitch+io:  {post_secs:.1f}s")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
