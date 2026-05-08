"""
Render an offline-pipeline npz onto the source per-camera videos so we
can SEE whether the 3D keypoints land where the people actually are.

Inputs:
  * tests/data/<recording>/keypoints_3d.npz (or per-backend snapshot)
  * tests/data/<recording>/port_<port>_*.mp4
  * extrinsics db (for projection matrices)

Output:
  tests/data/<recording>/annotated/<backend>_port<n>.mp4
  with the active track ids' 2D-projected skeleton drawn each frame.

Run:
    uv run python3 tests/manual/annotate_offline_npz.py
    uv run python3 tests/manual/annotate_offline_npz.py --npz keypoints_3d.mps.npz

The video drawn:
  * per-track colour (from CAMERA_COLORS palette, recycled for tracks)
  * Hip-COM marked with a + sign
  * SynthPose-52 → COCO-17 skeleton edges drawn between visible kps
  * track id label near the head
  * frame index + sync index in the corner
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[2]
RECORDING_NAME = "zelda_20260428_151934_fga_horizontal_head_turns"
ZELDA = REPO / "tests" / "data" / RECORDING_NAME

# COCO-17 indices in SynthPose-52 — keep visualisation simple by drawing
# only the COCO bones; SynthPose's extra anatomical landmarks aren't part
# of the standard skeleton.
SKELETON_EDGES = [
    (5, 7), (7, 9),       # left arm
    (6, 8), (8, 10),      # right arm
    (5, 6),               # shoulders
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15),   # left leg
    (12, 14), (14, 16),   # right leg
    (0, 5), (0, 6),       # neck-ish
]
KP_RADIUS = 4
LINE_THICKNESS = 2
L_HIP = 11
R_HIP = 12
HEAD_KP = 0  # for track id label position

# Colours for tracks, BGR (OpenCV order). Recycled if there are more
# tracks than colours.
TRACK_COLORS_BGR = [
    (120, 200, 80),    # green
    (255, 160, 100),   # blue
    (80, 180, 255),    # orange
    (220, 100, 220),   # purple
    (100, 100, 255),   # red
    (220, 220, 100),   # cyan
    (80, 220, 255),    # yellow
    (255, 140, 180),   # lavender
]


def _project_3d(point: np.ndarray, P: np.ndarray) -> tuple[int, int] | None:
    """Project a single 3D world-frame point through 3×4 P; return (u, v) px or None."""
    if point is None or not np.isfinite(point).all():
        return None
    p4 = np.append(point, 1.0)
    p2h = P @ p4
    if abs(p2h[2]) < 1e-6:
        return None
    u = int(round(p2h[0] / p2h[2]))
    v = int(round(p2h[1] / p2h[2]))
    return u, v


def _undo_view_transform(kps: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Inverse of `p_view = R @ p_world + t` to recover calibration-world coords."""
    if np.allclose(R, np.eye(3)) and np.allclose(t, 0.0):
        return kps  # identity — npz already in world frame
    # kps shape (N, P, K, 3). Mask NaN, transform, restore NaN.
    out = np.empty_like(kps)
    mask = np.isfinite(kps).all(axis=-1)
    Rt = R.T
    flat_in = kps[mask].reshape(-1, 3).astype(np.float64)
    flat_out = (flat_in - t) @ Rt.T  # (p - t) R^T  ←→  R.T @ (p - t)
    out[mask] = flat_out.astype(kps.dtype)
    out[~mask] = np.nan
    return out


def _compute_proj_matrices(session_id: int, ports: list[int]) -> dict[int, np.ndarray]:
    """Build 3×4 projection matrices per port from the extrinsics db."""
    from calimerge.config import load_extrinsic_session
    from calimerge.types import compute_projection_matrix

    loaded = load_extrinsic_session(session_id)
    if loaded is None:
        raise RuntimeError(f"could not load extrinsic session id={session_id}")
    _created_at, cams = loaded
    out: dict[int, np.ndarray] = {}
    for p in ports:
        if p not in cams:
            continue
        out[p] = compute_projection_matrix(cams[p])
    return out


def _annotate_one_camera(
    video_path: Path,
    out_path: Path,
    P: np.ndarray,
    npz_kps: np.ndarray,
    backend_label: str,
) -> int:
    """Read video frame-by-frame, draw projected keypoints, write output."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  could not open {video_path}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"  VideoWriter failed to open {out_path}")
        cap.release()
        return 1

    n_frames_npz = npz_kps.shape[0]
    n_persons = npz_kps.shape[1]
    n_kps = npz_kps.shape[2]

    n_drawn = 0
    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if f_idx < n_frames_npz:
            persons = npz_kps[f_idx]  # (P, K, 3)
            for slot in range(n_persons):
                kps = persons[slot]
                color = TRACK_COLORS_BGR[slot % len(TRACK_COLORS_BGR)]
                # Project all keypoints once
                uv = [
                    _project_3d(kps[k], P) for k in range(n_kps)
                ]
                # Skeleton edges
                for a, b in SKELETON_EDGES:
                    if a < n_kps and b < n_kps and uv[a] and uv[b]:
                        cv2.line(frame, uv[a], uv[b], color, LINE_THICKNESS)
                # Keypoints
                for k in range(n_kps):
                    if uv[k] is not None:
                        cv2.circle(frame, uv[k], KP_RADIUS, color, -1)
                # Hip COM marker
                if uv[L_HIP] and uv[R_HIP]:
                    cx = (uv[L_HIP][0] + uv[R_HIP][0]) // 2
                    cy = (uv[L_HIP][1] + uv[R_HIP][1]) // 2
                    cv2.drawMarker(
                        frame, (cx, cy), color,
                        markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2,
                    )
                # Track id label near head
                if uv[HEAD_KP]:
                    label = f"t{slot}"
                    cv2.putText(
                        frame, label, (uv[HEAD_KP][0] + 8, uv[HEAD_KP][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                    )
                # Count this slot as drawn if any kp was visible
                if any(uv):
                    n_drawn += 1
        # Frame info banner
        cv2.rectangle(frame, (0, 0), (340, 28), (0, 0, 0), -1)
        cv2.putText(
            frame, f"{backend_label}  frame {f_idx:>4}/{n_frames_npz}",
            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        )
        writer.write(frame)
        f_idx += 1

    cap.release()
    writer.release()
    print(f"  wrote {out_path}  ({f_idx} frames, {n_drawn} slot-frames drawn)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz", type=str, default="keypoints_3d.npz",
        help="filename inside the recording dir (default: keypoints_3d.npz). "
             "Use e.g. keypoints_3d.mps.npz to render a per-backend snapshot.",
    )
    parser.add_argument(
        "--session-id", type=int, default=None,
        help="Force a specific extrinsic session id; default uses the same "
             "selection logic as run_offline_pipeline_on_test_data.py.",
    )
    parser.add_argument(
        "--out-name", type=str, default=None,
        help="Override the output filename stem; default derives from --npz.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory; default is <recording>/annotated/.",
    )
    args = parser.parse_args(argv)

    npz_path = ZELDA / args.npz
    if not npz_path.exists():
        print(f"missing: {npz_path}")
        return 1
    print(f"npz: {npz_path}")

    # Pick session id (timestamp-fallback by default). Mirror the runner's
    # logic; here we just inline the "before recording" branch.
    if args.session_id is not None:
        session_id = args.session_id
    else:
        from calimerge.config import list_extrinsic_sessions
        recording_iso = "2026-04-28 15:19:34"
        session_id = None
        for s in list_extrinsic_sessions():
            if str(s["created_at"]) <= recording_iso:
                session_id = int(s["id"])
                break
        if session_id is None:
            print("no extrinsic session predates the recording")
            return 2
    print(f"using extrinsic session id={session_id}")

    # Load + invert any view transform on the saved keypoints.
    d = np.load(npz_path)
    kps = d["keypoints_3d"]
    R = (
        d["view_transform_R"] if "view_transform_R" in d.files
        else np.eye(3)
    )
    t = (
        d["view_transform_t"] if "view_transform_t" in d.files
        else np.zeros(3)
    )
    backend = (
        str(d["model_backend"]) if "model_backend" in d.files else "?"
    )
    kps_world = _undo_view_transform(kps, R, t)
    print(f"  shape={kps.shape}  backend={backend}")

    # Find ports + their video files
    port_to_video: dict[int, Path] = {}
    for p in sorted(ZELDA.glob("port_*.mp4")):
        try:
            port = int(p.stem.split("_")[1].split("-")[0])
        except Exception:
            continue
        port_to_video[port] = p

    # Build per-port projection matrices
    Ps = _compute_proj_matrices(session_id, list(port_to_video.keys()))
    out_stem = args.out_name or Path(args.npz).stem  # e.g. 'keypoints_3d' or 'keypoints_3d.mps'

    out_dir = args.out_dir if args.out_dir is not None else ZELDA / "annotated"
    print(f"output dir: {out_dir}")

    rc = 0
    for port, vid in port_to_video.items():
        if port not in Ps:
            print(f"  port {port}: no projection matrix (uncalibrated); skipping")
            continue
        out_path = out_dir / f"{out_stem}_port{port}.mp4"
        rc |= _annotate_one_camera(
            vid, out_path, Ps[port], kps_world, f"{backend}|p{port}",
        )
    print("[done]")
    return rc


if __name__ == "__main__":
    sys.exit(main())
