"""Scratch tests for the active debugging session.

Throwaway. Overwrite freely. The current focus is figuring out why the
unified offline pipeline produces 3D ankle positions ~2 m off from the
online pipeline at every matching timestamp, even though both use
identical view transforms and identical calibration.

The newest test isolates the FIRST frame of each port and runs the
unified PyTorch detection path on it standalone — if it fails to find
the user there, the bug is at the per-frame primitive level (not in
the iteration loop or stitcher). The user is clearly visible at frame 0
in both videos.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDING = "zelda_20260428_152104_fga_horizontal_head_turns"
TEST_DIR = REPO_ROOT / "tests" / "data" / RECORDING


# ── 1. Are sync_indices contiguous? Bug hypothesis: gaps in
#       sync_index passed to CudaStreamPipeline.process_frame break
#       cross-frame tracker continuity.

def test_sync_index_continuity():
    csv_path = TEST_DIR / "frame_time_history.csv"
    if not csv_path.exists():
        return
    syncs: set[int] = set()
    with open(csv_path) as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or row[0].startswith("#") or row[0] == "sync_index":
                continue
            try:
                syncs.add(int(row[0]))
            except ValueError:
                pass
    ss = sorted(syncs)
    contiguous = len(ss) == ss[-1] - ss[0] + 1
    print(f"sync indices: {len(ss)} unique, range [{ss[0]}..{ss[-1]}]")
    print(f"contiguous? {contiguous}")
    if not contiguous:
        gaps = [(ss[i], ss[i + 1] - ss[i]) for i in range(len(ss) - 1)
                if ss[i + 1] - ss[i] > 1]
        print(f"first gaps: {gaps[:5]}")


# ── 2. Per-port frame counts in frame_time_history.csv vs encoded
#       frame counts in the .mp4. They MUST agree exactly or the
#       UnifiedOfflineWorker's "read once per port appearance"
#       iteration desyncs.

def test_per_port_frame_counts_match_videos():
    import cv2

    csv_path = TEST_DIR / "frame_time_history.csv"
    if not csv_path.exists():
        return

    port_appearance: dict[int, int] = {}
    with open(csv_path) as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or row[0].startswith("#") or row[0] == "sync_index":
                continue
            try:
                port = int(row[1])
            except (ValueError, IndexError):
                continue
            port_appearance[port] = port_appearance.get(port, 0) + 1

    print(f"per-port appearances in frame_time_history.csv: {port_appearance}")

    # Compare to encoded video frame count.
    for video in sorted(TEST_DIR.glob("port_*.mp4")):
        try:
            stem = video.stem
            after = stem[len("port_"):]
            port = int(after.split("_", 1)[0])
        except Exception:
            continue
        cap = cv2.VideoCapture(str(video))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        appearances = port_appearance.get(port, -1)
        match = "OK" if n_frames == appearances else "MISMATCH"
        print(f"  port {port}: video={n_frames} {w}x{h}, "
              f"csv_appearances={appearances}  {match}")


# ── 3. Frame-by-frame (timestamp-matched) ankle delta between
#       online and offline npzs. Reports the median + max.

def test_online_vs_offline_ankle_delta():
    online_root = (
        Path("~/OneDrive/Documents/calimerge/recordings/workouts").expanduser()
        / RECORDING / "keypoints_3d.npz"
    )
    offline = TEST_DIR / "keypoints_3d.npz"
    if not online_root.exists() or not offline.exists():
        return

    L_ANKLE = 15

    def _union_left(npz: Path):
        d = np.load(str(npz))
        kps = d["keypoints_3d"]
        t = d["timestamps"]
        n_frames, max_persons = kps.shape[0], kps.shape[1]
        out = np.full((n_frames, 3), np.nan, dtype=np.float64)
        for i in range(n_frames):
            for p in range(max_persons):
                la = kps[i, p, L_ANKLE]
                if np.all(np.isfinite(la)) and not np.all(np.isfinite(out[i])):
                    out[i] = la
        return np.asarray(t, dtype=np.float64), out

    on_t, on_l = _union_left(online_root)
    of_t, of_l = _union_left(offline)

    deltas: list[float] = []
    for j in range(len(of_t)):
        if not np.isfinite(of_l[j, 0]):
            continue
        i = int(np.argmin(np.abs(on_t - of_t[j])))
        if not np.isfinite(on_l[i, 0]):
            continue
        if abs(float(on_t[i] - of_t[j])) > 0.05:
            continue
        deltas.append(float(np.linalg.norm(on_l[i] - of_l[j])))

    if deltas:
        arr = np.asarray(deltas)
        print(f"timestamp-matched overlap: {len(deltas)} frames")
        print(f"  delta mean: {arr.mean():.3f} m")
        print(f"  delta median: {np.median(arr):.3f} m")
        print(f"  delta p95: {np.percentile(arr, 95):.3f} m")
        print(f"  delta max: {arr.max():.3f} m")
    else:
        print("no timestamp-matched overlap")


# ── 4. Sanity: shape + view_transform contents of both npzs.

def test_npz_metadata_summary():
    online_root = (
        Path("~/OneDrive/Documents/calimerge/recordings/workouts").expanduser()
        / RECORDING / "keypoints_3d.npz"
    )
    offline = TEST_DIR / "keypoints_3d.npz"

    for label, p in (("online", online_root), ("offline", offline)):
        if not p.exists():
            print(f"{label}: missing {p}")
            continue
        d = np.load(str(p))
        kps = d["keypoints_3d"]
        counts = d.get("person_count")
        print(f"{label}: kps shape {kps.shape}, "
              f"valid-person frames "
              f"{int((counts > 0).sum()) if counts is not None else 'n/a'}")
        if "view_transform_R" in d.files:
            R = d["view_transform_R"]
            t = d["view_transform_t"]
            print(f"  R det={np.linalg.det(R):+.6f}  "
                  f"t={t.tolist()}")
        for k in d.files:
            print(f"  field: {k} -> shape={getattr(d[k], 'shape', '?')}, dtype={getattr(d[k], 'dtype', '?')}")


# ── 5b. What backend is the user currently using? (Last persisted.)
#        Helps decide whether the apples-to-apples comparison should
#        use --unified-backend pytorch or --unified-backend cuda.
def test_first_frame_unified_pytorch_detection():
    """Read the first frame from each port .mp4, drive the unified
    PyTorch detection path on those two frames standalone, and report:
      - did YOLO find boxes in each frame?
      - did VitPose produce keypoints?
      - did _triangulate_live produce a 3D person?
      - what's the resulting hip-COM (in body frame after the saved
        view transform from the live npz)?

    If any of those steps fail on the first frame, the bug is at the
    per-frame primitive level, not in the iteration / stitcher.

    This test runs `pytest -s` against the YOLO + VitPose models on
    the device the worker auto-picks, so it'll take a few seconds the
    first time (model load).
    """
    import cv2

    if not TEST_DIR.exists():
        print(f"skipped: missing fixture {TEST_DIR}")
        return

    videos = sorted(TEST_DIR.glob("port_*.mp4"))
    if not videos:
        print(f"skipped: no port_*.mp4 under {TEST_DIR}")
        return

    # Read frame 0 from each port.
    port_frame: dict[int, "np.ndarray"] = {}
    for v in videos:
        try:
            stem = v.stem
            after = stem[len("port_"):]
            port = int(after.split("_", 1)[0])
        except Exception:
            continue
        cap = cv2.VideoCapture(str(v))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"FAIL: could not read frame 0 from {v.name}")
            return
        port_frame[port] = frame
        print(f"port {port}: frame 0 shape={frame.shape}")

    if len(port_frame) < 2:
        print(f"skipped: only {len(port_frame)} ports decoded")
        return

    # Load latest extrinsic, normalise intrinsics to video resolution
    # exactly as the test runner does for the unified worker.
    from calimerge.config import (
        load_latest_extrinsic_session,
        load_view_transform,
    )
    from calimerge.types import CalibratedCamera, scale_intrinsics
    import dataclasses as _dc

    sess = load_latest_extrinsic_session()
    if sess is None:
        print("skipped: no extrinsic session")
        return
    sess_id, created_at, cams_db = sess
    print(f"session id={sess_id} ports={sorted(cams_db.keys())}")

    # Remap calibration cameras to match the recording's port numbers
    # via serial — same path the runner takes when ports differ.
    mapping_csv = TEST_DIR / "camera_mapping.csv"
    port_to_serial: dict[int, str] = {}
    if mapping_csv.exists():
        with open(mapping_csv, newline="") as f:
            for row in csv.DictReader(f):
                port_to_serial[int(row["port"])] = row["serial_number"]

    serial_to_cam = {c.serial_number: c for c in cams_db.values()}
    cameras: dict[int, CalibratedCamera] = {}
    for vp in port_frame.keys():
        ser = port_to_serial.get(vp)
        base = serial_to_cam.get(ser) if ser else None
        if base is None and vp in cams_db:
            base = cams_db[vp]
        if base is None:
            print(f"FAIL: no calibration for port {vp}")
            return
        # Rescale to video resolution.
        target_h, target_w = next(iter(port_frame.values())).shape[:2]
        target_res = (int(target_w), int(target_h))
        if tuple(base.intrinsics.resolution) != target_res:
            scaled = scale_intrinsics(base.intrinsics, target_res)
            cameras[vp] = _dc.replace(base, intrinsics=scaled, port=vp)
        else:
            cameras[vp] = _dc.replace(base, port=vp)

    # Construct a PoseDetectionWorker exactly as UnifiedOfflineWorker
    # does — same model loading, no QThread start.
    from calimerge.gui.workers import PoseDetectionWorker
    from calimerge.tracking.pose_detector import setup_device, load_models

    worker = PoseDetectionWorker(device_name="auto", cameras=cameras)
    worker.confidence_threshold = 0.5

    device = setup_device("auto")
    worker._device = device
    print(f"loading models on device={device}...")
    person_model, pose_processor, pose_model = load_models(device=device)
    worker._models = (person_model, pose_processor, pose_model)
    worker._tracker.reset()

    # Capture keypoints_3d_ready emissions.
    collected: list = []

    def _collect(persons: list) -> None:
        collected.append(list(persons))

    worker.keypoints_3d_ready.connect(_collect)

    # ── 1. Per-frame YOLO + VitPose ────────────────────────────────
    print("\n=== STAGE 1: detect_and_draw_batch on frame 0 of each port ===")
    work = dict(port_frame)
    worker._detect_and_draw_batch(work)
    for port in sorted(port_frame.keys()):
        kps_list = worker._last_kps_per_port.get(port)
        if kps_list:
            print(f"  port {port}: detected {len(kps_list)} person(s)")
            for j, (kps, scores) in enumerate(kps_list):
                print(f"    person {j}: {kps.shape[0]} kps, "
                      f"mean score={float(np.mean(scores)):.3f}")
        else:
            print(f"  port {port}: NO DETECTION  <-- BUG if a person is visible")

    # ── 2. Triangulation ───────────────────────────────────────────
    print("\n=== STAGE 2: _triangulate_live ===")
    if len(worker._last_kps_per_port) < 2:
        print("  skipped: fewer than 2 ports have a detection (no 3D possible)")
        return
    worker._triangulate_live()
    if not collected:
        print("  _triangulate_live did NOT emit; bare 'except: pass' at "
              "workers.py:1252 swallowed the underlying error. Replaying "
              "the inner block directly so the traceback surfaces.")
        # Replay the body of _triangulate_live without the try/except —
        # whichever line raises is the real bug.
        from calimerge.tracking.triangulation import triangulate_keypoints
        from calimerge.tracking.tracker import (
            group_detections_across_views_bipartite, calculate_2d_com,
        )
        from calimerge.tracking.markers import HIP_INDICES
        worker._ensure_camera_caches()
        camera_params = worker._cached_cam_params
        port_to_cam_index = worker._cached_port_to_cam_index
        projection_matrices = worker._cached_proj_matrices
        print(f"  cam_params: {len(camera_params) if camera_params else None}")
        print(f"  port_to_cam_index: {port_to_cam_index}")
        port_persons = {
            port: persons
            for port, persons in worker._last_kps_per_port.items()
            if port in (port_to_cam_index or {}) and persons
        }
        print(f"  port_persons keys: {list(port_persons.keys())}")
        detected_persons_2d: dict[int, list[dict]] = {}
        for port, persons in port_persons.items():
            port_dets = []
            for kps, scores in persons:
                kps_with_score = np.concatenate([kps, scores[:, None]], axis=1)
                com_2d = calculate_2d_com(kps_with_score.tolist(), HIP_INDICES)
                print(f"    port {port}: kps shape={kps.shape}, "
                      f"com_2d={com_2d}")
                if com_2d is None:
                    continue
                port_dets.append({
                    "keypoints": kps_with_score,
                    "com_2d": com_2d,
                })
            if port_dets:
                detected_persons_2d[port] = port_dets
        print(f"  detected_persons_2d keys: {list(detected_persons_2d.keys())}")
        if len(detected_persons_2d) >= 2:
            for thr in (50.0, 100.0, 200.0, 500.0, 1000.0):
                grp = group_detections_across_views_bipartite(
                    detected_persons_2d,
                    projection_matrices,
                    port_to_cam_index,
                    camera_params,
                    epipolar_threshold=thr,
                )
                print(f"  epipolar_threshold={thr:>6.1f}  -> {len(grp)} group(s)")
            groups = group_detections_across_views_bipartite(
                detected_persons_2d,
                projection_matrices,
                port_to_cam_index,
                camera_params,
                epipolar_threshold=1000.0,
            )
            print(f"  using epipolar=1000 for further inspection:")
            for gi, group in enumerate(groups):
                print(f"    group {gi}: ports={list(group.keys())}")
                if len(group) >= 2:
                    kp_dict = {port: det["keypoints"]
                               for port, det in group.items()}
                    kps_3d = triangulate_keypoints(
                        kp_dict, port_to_cam_index,
                        camera_params, projection_matrices,
                    )
                    print(f"      triangulated to {len(kps_3d)} kp slots; "
                          f"first non-None: "
                          f"{next((k for k in kps_3d if k is not None), None)}")
        return
    persons_3d = collected[-1]
    print(f"  triangulated persons: {len(persons_3d)}")

    # ── 3. Apply the view transform from the live online npz so the
    #       result is in the SAME body frame the online file is in ──
    online_npz = (
        Path("~/OneDrive/Documents/calimerge/recordings/workouts").expanduser()
        / RECORDING / "keypoints_3d.npz"
    )
    R_view = None
    t_view = None
    if online_npz.exists():
        d = np.load(str(online_npz))
        if "view_transform_R" in d.files:
            R_view = d["view_transform_R"]
            t_view = d["view_transform_t"]

    HIP_L, HIP_R = 11, 12
    L_ANKLE = 15
    print("\n=== STAGE 3: per-person hip-COM + L_Ankle in body frame ===")
    for j, kps in enumerate(persons_3d):
        if not kps:
            continue
        valid = [k for k in (HIP_L, HIP_R, L_ANKLE)
                 if k < len(kps) and kps[k] is not None
                 and not np.isnan(kps[k]).any()]
        print(f"  person {j}: valid kp indices in (L_HIP, R_HIP, L_ANKLE) = {valid}")

        def _xform(p):
            if R_view is None or t_view is None:
                return np.asarray(p, dtype=float)
            return R_view @ np.asarray(p, dtype=float) + t_view

        if HIP_L < len(kps) and kps[HIP_L] is not None and not np.isnan(kps[HIP_L]).any():
            print(f"    L_Hip body: {_xform(kps[HIP_L])}")
        if L_ANKLE < len(kps) and kps[L_ANKLE] is not None and not np.isnan(kps[L_ANKLE]).any():
            print(f"    L_Ankle body: {_xform(kps[L_ANKLE])}")

    # ── 4. Compare to online's frame 0 ─────────────────────────────
    print("\n=== STAGE 4: online frame 0 ankle (body frame) ===")
    if online_npz.exists():
        d = np.load(str(online_npz))
        kps_arr = d["keypoints_3d"]
        print(f"  online npz frame 0: {kps_arr[0, 0, L_ANKLE]}")
    else:
        print(f"  no online npz at {online_npz}")


def test_session_extrinsic_provenance():
    """For the zelda fixture: which extrinsic was ACTIVE when the
    recording was made? Compare to the LATEST extrinsic that the
    offline runner loads. If they differ, the runner is using the
    wrong calibration on these videos.
    """
    import sqlite3
    from calimerge.config import (
        workouts_db_path, list_extrinsic_sessions,
        load_latest_extrinsic_session,
    )

    wdb = workouts_db_path()
    if not wdb.exists():
        print("workouts.db missing")
        return
    conn = sqlite3.connect(str(wdb))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, recording_path, created_at, extrinsic_session_id, "
            "       extrinsic_calibrated_at "
            "FROM sessions WHERE recording_path LIKE ?",
            (f"%{RECORDING}",),
        ).fetchall()
        for r in rows:
            print(f"session id={r['id']}  workouts.db_created={r['created_at']}")
            print(f"  recording_path = {r['recording_path']}")
            print(f"  extrinsic_session_id = {r['extrinsic_session_id']}")
            print(f"  extrinsic_calibrated_at = {r['extrinsic_calibrated_at']}")
    finally:
        conn.close()

    print()
    print("All extrinsic sessions in extrinsics.db (newest first):")
    for sess in list_extrinsic_sessions():
        print(f"  id={sess['id']}  created_at={sess['created_at']}  "
              f"rmse={sess['rmse']}  notes={sess.get('notes')}")

    print()
    latest = load_latest_extrinsic_session()
    if latest:
        sid, created_at, _ = latest
        print(f"runner loads LATEST: id={sid} created={created_at}")


def test_session_extrinsic_provenance():
    """For the zelda fixture: which extrinsic was ACTIVE when the
    recording was made? Compare to the LATEST extrinsic that the
    offline runner USED to load. They differ when the user recalibrated
    after recording — using the latest produces 3D points off by metres.
    """
    import sqlite3
    from calimerge.config import (
        workouts_db_path, list_extrinsic_sessions,
        load_latest_extrinsic_session,
    )

    wdb = workouts_db_path()
    if not wdb.exists():
        print("workouts.db missing")
        return
    conn = sqlite3.connect(str(wdb))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, recording_path, created_at, extrinsic_session_id, "
            "       extrinsic_calibrated_at "
            "FROM sessions WHERE recording_path LIKE ?",
            (f"%{RECORDING}",),
        ).fetchall()
        for r in rows:
            print(f"workouts session id={r['id']}  created={r['created_at']}")
            print(f"  recording_path = {r['recording_path']}")
            print(f"  extrinsic_session_id = {r['extrinsic_session_id']}")
            print(f"  extrinsic_calibrated_at = {r['extrinsic_calibrated_at']}")
    finally:
        conn.close()

    print("\nAll extrinsic sessions (newest first):")
    for sess in list_extrinsic_sessions():
        print(f"  id={sess['id']}  created_at={sess['created_at']}  "
              f"rmse={sess['rmse']}")

    latest = load_latest_extrinsic_session()
    if latest:
        sid, ts, _ = latest
        print(f"\nload_latest_extrinsic_session would return: "
              f"id={sid} created={ts}")


def test_app_settings_last_backend():
    from calimerge.config import load_app_settings
    settings = load_app_settings()
    keys_of_interest = (
        "last_detect_model",
        "last_detect_backend",
        "last_detect_confidence",
        "csv_export_immediate",
    )
    for k in keys_of_interest:
        print(f"  {k} = {settings.get(k)!r}")


# ── 5. Was the original online recording made with PyTorch or CUDA?
#       Check the workouts.db session row for this recording —
#       create_session stores model_backend in the config blob.
def test_session_backend_from_workouts_db():
    import sqlite3
    from calimerge.config import workouts_db_path
    db = workouts_db_path()
    if not db.exists():
        print("workouts.db missing")
        return
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, recording_path, workout_type, created_at "
            "FROM sessions WHERE recording_path LIKE ?",
            (f"%{RECORDING}",),
        ).fetchall()
        for r in rows:
            print(f"session id={r['id']} created={r['created_at']} "
                  f"workout={r['workout_type']} path={r['recording_path']}")
            results = conn.execute(
                "SELECT metric_name, metric_value FROM session_results "
                "WHERE session_id = ?",
                (r["id"],),
            ).fetchall()
            for res in results:
                print(f"  {res['metric_name']} = {res['metric_value']}")
    finally:
        conn.close()
