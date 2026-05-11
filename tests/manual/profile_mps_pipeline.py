"""
MPS pipeline per-stage performance profiler.

Runs the MPS streaming pipeline (CoreML / ANE) against the zelda fixture,
collects cumulative per-stage timing from the C side, and reports a
breakdown table so we can see where the time goes before deciding on
optimisation strategy (buffer pre-alloc, Metal shaders, full MPSGraph, etc).

Measurements come from two sources:
  * C-side cumulative stats (PT_MPS_StreamStats via MpsStreamPipeline.get_stats):
    preprocess, YOLO inference, VitPose inference, matching, triangulation,
    tracking — these are wall-clock timers inside pt_mps_stream_process_frame.
  * Python-side wall-clock around cv2.read() (video decode) and the full
    process_frame call (includes ctypes overhead + the C side total).

The script reuses the same calibration/setup logic as
run_offline_pipeline_on_test_data.py — if that script runs, this one will too.

Run:
    VIRTUAL_ENV= ~/.local/bin/uv run python tests/manual/profile_mps_pipeline.py

Options:
    --max-syncs N       Cap frames processed (default: all)
    --warmup N          Discard first N frames from stats (default: 5)
    --per-frame         Print per-frame timings (verbose)
    --vitpose-model P   Override VitPose .mlpackage path

Benchmark history (zelda fixture, 2 cameras, 640x480, M-series Mac):
  ┌────────────┬─────────────┬───────────┬────────────┬────────────┐
  │ Date       │ Model       │ VitPose   │ Total/frame│ Throughput │
  ├────────────┼─────────────┼───────────┼────────────┼────────────┤
  │ 2026-05-08 │ batch-16    │ 115.0 ms  │ 128.6 ms   │  7.8 fps  │
  │ (baseline) │ no caching  │           │            │            │
  ├────────────┼─────────────┼───────────┼────────────┼────────────┤
  │ 2026-05-08 │ batch-16    │  99.8 ms  │ 113.5 ms   │  8.8 fps  │
  │ Option A   │ + buf cache │           │            │            │
  ├────────────┼─────────────┼───────────┼────────────┼────────────┤
  │ 2026-05-09 │ batch-4     │  21.0 ms  │  34.8 ms   │ 28.7 fps  │
  │ rt_batch4  │ + buf cache │           │            │            │
  └────────────┴─────────────┴───────────┴────────────┴────────────┘
  Option E (ONNX Runtime CoreML EP): ruled out — VitPose unsupported,
    YOLO 8x slower due to graph partitioning.
  Option C (MPSGraph rewrite): ruled out — would lose ANE access,
    97% of time is actual model compute, not API overhead.
"""

from __future__ import annotations

import argparse
import csv as _csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = REPO_ROOT / "tests" / "data"
DEFAULT_RECORDING = "zelda_20260428_151934_fga_horizontal_head_turns"


def parse_args():
    p = argparse.ArgumentParser(description="MPS pipeline performance profiler")
    p.add_argument("recording", nargs="?", default=DEFAULT_RECORDING)
    p.add_argument("--max-syncs", type=int, default=0,
                   help="Cap sync indices processed (0 = all)")
    p.add_argument("--warmup", type=int, default=5,
                   help="Discard first N frames from per-frame stats")
    p.add_argument("--per-frame", action="store_true",
                   help="Print per-frame timing rows")
    p.add_argument("--person-confidence", type=float, default=0.5)
    p.add_argument("--max-track-distance", type=float, default=0.5)
    p.add_argument("--track-patience", type=int, default=60)
    p.add_argument("--extrinsic-session-id", type=int, default=None)
    p.add_argument("--vitpose-model", type=str, default=None,
                   help="Override VitPose .mlpackage path (default: models/coreml/vitpose_synthpose.mlpackage)")
    return p.parse_args()


def read_sync_to_ports(csv_path: Path) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    with open(csv_path, "r", newline="") as f:
        reader = _csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#") or row[0] == "sync_index":
                continue
            try:
                sync = int(row[0])
                port = int(row[1])
            except (ValueError, IndexError):
                continue
            out.setdefault(sync, []).append(port)
    return out


def resolve_calibration(recording_name: str, args):
    """Resolve extrinsic calibration — same priority as run_offline_pipeline_on_test_data."""
    from calimerge.config import (
        load_extrinsic_session,
        load_latest_extrinsic_session,
        load_view_transform,
        write_cuda_calibration_toml,
        list_extrinsic_sessions,
        extrinsics_db_path,
        workouts_db_path,
    )
    import tempfile

    sess_id = None
    calibrated_cams = None
    chosen_via = None

    if args.extrinsic_session_id is not None:
        loaded = load_extrinsic_session(args.extrinsic_session_id)
        if loaded is None:
            print(f"BLOCKED: --extrinsic-session-id {args.extrinsic_session_id} not found")
            sys.exit(2)
        _, calibrated_cams = loaded
        sess_id = args.extrinsic_session_id
        chosen_via = f"--extrinsic-session-id {sess_id}"
    else:
        # workouts.db lookup
        try:
            import sqlite3
            wdb = workouts_db_path()
            if wdb.exists():
                conn = sqlite3.connect(str(wdb))
                try:
                    row = conn.execute(
                        "SELECT extrinsic_session_id FROM sessions "
                        "WHERE recording_path LIKE ? ORDER BY created_at DESC LIMIT 1",
                        (f"%{recording_name}",),
                    ).fetchone()
                finally:
                    conn.close()
                if row is not None and row[0] is not None:
                    loaded = load_extrinsic_session(int(row[0]))
                    if loaded is not None:
                        _, calibrated_cams = loaded
                        sess_id = int(row[0])
                        chosen_via = "workouts.db"
        except Exception as e:
            print(f"[calib] workouts.db lookup raised: {e}")

        # timestamp fallback
        if calibrated_cams is None:
            try:
                stamp = recording_name.split("_", 2)
                date_str = time_str = None
                if len(stamp) >= 3 and stamp[1].isdigit() and len(stamp[1]) == 8:
                    date_str = stamp[1]
                    time_str = stamp[2].split("_", 1)[0]
                if date_str and time_str and len(time_str) >= 6:
                    rec_iso = (
                        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} "
                        f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                    )
                    for sess in list_extrinsic_sessions():
                        if str(sess["created_at"]) <= rec_iso:
                            loaded = load_extrinsic_session(int(sess["id"]))
                            if loaded is not None:
                                _, calibrated_cams = loaded
                                sess_id = int(sess["id"])
                                chosen_via = f"timestamp before {rec_iso}"
                                break
            except Exception:
                pass

        # last resort
        if calibrated_cams is None:
            latest = load_latest_extrinsic_session()
            if latest is None:
                print("BLOCKED: no extrinsic calibration in extrinsics.db")
                sys.exit(2)
            sess_id, _, calibrated_cams = latest
            chosen_via = "latest (last resort)"

    print(f"[calib] session={sess_id} via={chosen_via} "
          f"cameras={sorted(calibrated_cams.keys())}")

    # Load view transform
    view_R = view_t = None
    try:
        preset = load_view_transform("synthpose", extrinsic_session_id=sess_id)
        if preset is not None:
            view_R, view_t, _ = preset
    except Exception:
        pass

    return calibrated_cams, sess_id, view_R, view_t


def main() -> int:
    args = parse_args()
    recording_name = args.recording
    test_dir = TEST_DATA_DIR / recording_name

    if not test_dir.is_dir():
        print(f"BLOCKED: {test_dir} does not exist. Run run_offline_pipeline_on_test_data.py first.")
        return 2

    # Check MPS availability
    try:
        from calimerge.tracking.mps_stream_binding import MpsStreamPipeline, is_available
    except ImportError as e:
        print(f"BLOCKED: cannot import MPS stream binding: {e}")
        return 2
    if not is_available():
        print("BLOCKED: MPS pipeline dylib not available. Build with: bash src/mps_pipeline/build_mps.sh release")
        return 2

    # Discover videos
    port_to_video: dict[int, Path] = {}
    for p in sorted(test_dir.glob("port_*.mp4")):
        try:
            port = int(p.stem[len("port_"):].split("_", 1)[0])
        except Exception:
            continue
        port_to_video[port] = p
    if not port_to_video:
        print(f"BLOCKED: no port_*.mp4 in {test_dir}")
        return 2

    frame_time_csv = test_dir / "frame_time_history.csv"
    if not frame_time_csv.exists():
        print(f"BLOCKED: missing {frame_time_csv}")
        return 2

    print(f"[setup] recording: {test_dir}")
    print(f"[setup] videos: {list(port_to_video.keys())}")

    # Calibration
    calibrated_cams, sess_id, view_R, view_t = resolve_calibration(recording_name, args)

    # Remap ports by serial if needed
    cal_ports = set(calibrated_cams.keys())
    vid_ports = set(port_to_video.keys())
    if vid_ports - cal_ports:
        mapping_csv = test_dir / "camera_mapping.csv"
        port_to_serial: dict[int, str] = {}
        if mapping_csv.exists():
            with open(mapping_csv, "r", newline="") as f:
                for row in _csv.DictReader(f):
                    port_to_serial[int(row["port"])] = row["serial_number"]
        serial_to_cal = {c.serial_number: c for c in calibrated_cams.values()}
        remapped = {}
        for vp in vid_ports:
            ser = port_to_serial.get(vp)
            if ser and ser in serial_to_cal:
                from calimerge.types import CalibratedCamera
                base = serial_to_cal[ser]
                remapped[vp] = CalibratedCamera(
                    serial_number=base.serial_number, port=vp,
                    intrinsics=base.intrinsics, extrinsics=base.extrinsics,
                )
            elif vp in calibrated_cams:
                remapped[vp] = calibrated_cams[vp]
        if remapped:
            calibrated_cams = remapped

    # Normalise intrinsics to video resolution
    any_video = next(iter(port_to_video.values()))
    cap = cv2.VideoCapture(str(any_video))
    target_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if target_w > 0 and target_h > 0:
        from calimerge.types import scale_intrinsics
        import dataclasses as _dc
        target_res = (target_w, target_h)
        normalised = {}
        for port, cc in calibrated_cams.items():
            if tuple(cc.intrinsics.resolution) == target_res:
                normalised[port] = cc
            else:
                normalised[port] = _dc.replace(cc, intrinsics=scale_intrinsics(cc.intrinsics, target_res))
        calibrated_cams = normalised

    # Write calibration TOML
    from calimerge.config import write_cuda_calibration_toml, models_dir
    import tempfile
    cal_path = Path(tempfile.gettempdir()) / "calimerge_profile_mps_cal.toml"
    calibrated_ports = sorted(p for p in port_to_video.keys() if p in calibrated_cams)
    cal_subset = {p: calibrated_cams[p] for p in calibrated_ports}
    write_cuda_calibration_toml(cal_subset, cal_path)

    # Sync table
    sync_to_ports = read_sync_to_ports(frame_time_csv)
    sync_indices = sorted(sync_to_ports.keys())
    if args.max_syncs > 0:
        sync_indices = sync_indices[:args.max_syncs]
    n_syncs = len(sync_indices)
    print(f"[setup] {n_syncs} sync indices to process")

    # CoreML model paths
    coreml_dir = models_dir() / "coreml"
    yolo_pkg = coreml_dir / "yolo_v10s.mlpackage"
    if args.vitpose_model:
        vitpose_pkg = Path(args.vitpose_model)
    else:
        vitpose_rt = coreml_dir / "vitpose_synthpose_rt_batch4.mlpackage"
        vitpose_pkg = vitpose_rt if vitpose_rt.exists() else (
            coreml_dir / "vitpose_synthpose.mlpackage"
        )
    if not yolo_pkg.exists() or not vitpose_pkg.exists():
        print(f"BLOCKED: CoreML models not found: yolo={yolo_pkg}, vitpose={vitpose_pkg}")
        return 2
    print(f"[setup] VitPose model: {vitpose_pkg.name}")

    # Open video captures
    captures: dict[int, cv2.VideoCapture] = {}
    for port, vpath in port_to_video.items():
        c = cv2.VideoCapture(str(vpath))
        if c.isOpened():
            captures[port] = c

    w, h = target_w, target_h

    # Create pipeline
    print(f"\n{'='*70}")
    print(f"[profile] creating MPS pipeline ({len(calibrated_ports)} cameras, {w}x{h})")
    print(f"{'='*70}")
    t_create = time.perf_counter()
    pipeline = MpsStreamPipeline(
        num_cameras=len(calibrated_ports),
        frame_width=w, frame_height=h,
        calibration_toml_path=str(cal_path),
        yolo_model_path=str(yolo_pkg),
        vitpose_model_path=str(vitpose_pkg),
        max_persons=2,
        person_confidence=args.person_confidence,
        max_track_distance=args.max_track_distance,
        track_patience=args.track_patience,
        log_callback=lambda m: print(f"[mps] {m}"),
    )
    create_ms = (time.perf_counter() - t_create) * 1000
    print(f"[profile] pipeline created in {create_ms:.0f} ms")

    # Main profiling loop
    per_frame_decode_ms: list[float] = []
    per_frame_process_ms: list[float] = []
    per_frame_c_ms: list[float] = []  # from StreamFrameResult.processing_time_ms
    per_frame_n_persons: list[int] = []

    print(f"\n[profile] processing {n_syncs} sync frames...")
    t_loop_start = time.perf_counter()

    for i, sync in enumerate(sync_indices):
        wanted = [p for p in sync_to_ports[sync] if p in calibrated_ports]

        # Decode
        t_dec = time.perf_counter()
        frame_list: list[tuple[np.ndarray, int]] = []
        for port in wanted:
            ok, frame = captures[port].read()
            if not ok or frame is None:
                continue
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            frame_list.append((frame, port))
        decode_ms = (time.perf_counter() - t_dec) * 1000

        if len(frame_list) < 2:
            continue

        # Process
        t_proc = time.perf_counter()
        result = pipeline.process_frame(frame_list, sync_index=sync)
        process_ms = (time.perf_counter() - t_proc) * 1000

        per_frame_decode_ms.append(decode_ms)
        per_frame_process_ms.append(process_ms)
        per_frame_c_ms.append(result.processing_time_ms)
        per_frame_n_persons.append(result.num_persons)

        if args.per_frame and i >= args.warmup:
            print(f"  sync={sync:4d}  decode={decode_ms:6.1f}ms  "
                  f"process={process_ms:6.1f}ms  "
                  f"c_time={result.processing_time_ms:6.1f}ms  "
                  f"persons={result.num_persons}")

        if i % max(1, n_syncs // 20) == 0:
            pct = 100 * i / max(1, n_syncs)
            print(f"  [{pct:5.1f}%] sync {sync}", end="\r")

    t_loop_end = time.perf_counter()
    loop_wall_s = t_loop_end - t_loop_start

    # Get C-side cumulative stats
    stats = pipeline.get_stats()
    pipeline.close()

    for cap in captures.values():
        cap.release()

    # Trim warmup
    warmup = min(args.warmup, len(per_frame_process_ms) - 1)
    pf_decode = per_frame_decode_ms[warmup:]
    pf_process = per_frame_process_ms[warmup:]
    pf_c = per_frame_c_ms[warmup:]
    n_measured = len(pf_process)

    # Report
    print(f"\n\n{'='*70}")
    print("MPS PIPELINE PERFORMANCE REPORT")
    print(f"{'='*70}")
    print(f"recording:          {recording_name}")
    print(f"cameras:            {len(calibrated_ports)}")
    print(f"resolution:         {w}x{h}")
    print(f"total syncs:        {n_syncs}")
    print(f"measured syncs:     {n_measured} (after {warmup} warmup)")
    print(f"pipeline create:    {create_ms:.0f} ms")
    print(f"loop wall time:     {loop_wall_s:.2f} s")
    if n_syncs > 0:
        print(f"throughput:         {n_syncs / loop_wall_s:.1f} sync-frames/s")

    # C-side cumulative breakdown (from PT_MPS_StreamStats)
    print(f"\n--- C-side cumulative stats (all {stats.frames_processed} frames) ---")
    c_total = stats.total_ms
    stages = [
        ("preprocess (letterbox + vp crop)", stats.preprocess_ms),
        ("YOLO CoreML inference",            stats.coreml_yolo_ms),
        ("VitPose CoreML inference",         stats.coreml_vitpose_ms),
        ("matching (epipolar)",              stats.matching_ms),
        ("triangulation + tracking",         stats.triangulation_ms),
        ("track output",                     stats.tracking_ms),
    ]
    accounted = sum(v for _, v in stages)
    unaccounted = c_total - accounted

    print(f"  {'stage':<36} {'total_ms':>10} {'per_frame':>10} {'pct':>6}")
    print(f"  {'-'*36} {'-'*10} {'-'*10} {'-'*6}")
    n_c = max(1, stats.frames_processed)
    for label, ms in stages:
        print(f"  {label:<36} {ms:10.1f} {ms/n_c:10.2f} {100*ms/max(1,c_total):5.1f}%")
    print(f"  {'(unaccounted / overhead)':<36} {unaccounted:10.1f} {unaccounted/n_c:10.2f} {100*unaccounted/max(1,c_total):5.1f}%")
    print(f"  {'-'*36} {'-'*10} {'-'*10} {'-'*6}")
    print(f"  {'TOTAL (C-side)':<36} {c_total:10.1f} {c_total/n_c:10.2f} {'100%':>6}")

    # Python-side per-frame stats (post-warmup)
    if n_measured > 0:
        print(f"\n--- Python-side per-frame stats ({n_measured} frames, post-warmup) ---")
        print(f"  {'metric':<36} {'mean':>10} {'p50':>10} {'p95':>10} {'max':>10}")
        print(f"  {'-'*36} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for label, arr in [
            ("video decode (cv2.read)", pf_decode),
            ("process_frame (Python→C→Python)", pf_process),
            ("C-side frame time", pf_c),
        ]:
            a = np.array(arr)
            print(f"  {label:<36} {a.mean():10.2f} {np.median(a):10.2f} "
                  f"{np.percentile(a, 95):10.2f} {a.max():10.2f}")

        # Python overhead = process_frame wall - C-side frame time
        py_overhead = np.array(pf_process) - np.array(pf_c)
        print(f"  {'ctypes / Python overhead':<36} {py_overhead.mean():10.2f} "
              f"{np.median(py_overhead):10.2f} {np.percentile(py_overhead, 95):10.2f} "
              f"{py_overhead.max():10.2f}")

        total_per_frame = np.array(pf_decode) + np.array(pf_process)
        print(f"  {'total per sync (decode+process)':<36} {total_per_frame.mean():10.2f} "
              f"{np.median(total_per_frame):10.2f} {np.percentile(total_per_frame, 95):10.2f} "
              f"{total_per_frame.max():10.2f}")

        # Achievable FPS
        mean_total = total_per_frame.mean()
        if mean_total > 0:
            print(f"\n  achievable throughput: {1000/mean_total:.1f} sync-frames/s "
                  f"(based on mean {mean_total:.1f} ms/sync)")

    # Summary for decision-making
    print(f"\n--- Decision summary ---")
    if c_total > 0 and n_c > 0:
        preprocess_pct = 100 * stats.preprocess_ms / c_total
        inference_pct = 100 * (stats.coreml_yolo_ms + stats.coreml_vitpose_ms) / c_total
        matching_tri_pct = 100 * (stats.matching_ms + stats.triangulation_ms) / c_total
        print(f"  preprocess (letterbox+crop):     {preprocess_pct:5.1f}%  → if >25%, Metal shaders (option B) help")
        print(f"  CoreML inference (YOLO+VitPose): {inference_pct:5.1f}%  → if dominant, buffer pre-alloc (option A) or ONNX-RT (E)")
        print(f"  matching + triangulation:        {matching_tri_pct:5.1f}%  → CPU-bound, less optimisable")
        print(f"  If even split across stages:     full MPSGraph rewrite (option C) is the only way")

    print(f"\n{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
