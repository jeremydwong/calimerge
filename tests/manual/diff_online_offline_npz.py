"""Numerical per-sync diff of online vs offline npz outputs.

Loads both, finds the time-aligned overlap, and reports for each frame:
  - online ankle (L) position
  - offline ankle (L) position
  - delta in metres

Aim: surface "is offline broken at every frame, or only at some?"

Usage:
    VIRTUAL_ENV= ~/.local/bin/uv run python \\
        tests/manual/diff_online_offline_npz.py \\
        zelda_20260428_152104_fga_horizontal_head_turns
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
L_ANKLE = 15
HIP_L = 11
HIP_R = 12


def _union_ankle(npz_path: Path):
    data = np.load(str(npz_path))
    times = data["timestamps"]
    kps = data["keypoints_3d"]
    n_frames, max_persons, n_kps, _ = kps.shape

    left = np.full((n_frames, 3), np.nan, dtype=np.float64)
    hip = np.full((n_frames, 3), np.nan, dtype=np.float64)
    for i in range(n_frames):
        for p in range(max_persons):
            la = kps[i, p, L_ANKLE]
            if np.all(np.isfinite(la)) and not np.all(np.isfinite(left[i])):
                left[i] = la
            hl = kps[i, p, HIP_L]
            hr = kps[i, p, HIP_R]
            if np.all(np.isfinite(hl)) and np.all(np.isfinite(hr)):
                if not np.all(np.isfinite(hip[i])):
                    hip[i] = (hl + hr) / 2
    return np.asarray(times, dtype=np.float64), left, hip


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: diff_online_offline_npz.py <recording_name>")
        return 2
    recording = sys.argv[1]

    online = (
        Path("~/OneDrive/Documents/calimerge/recordings/workouts").expanduser()
        / recording / "keypoints_3d.npz"
    )
    offline = REPO_ROOT / "tests" / "data" / recording / "keypoints_3d.npz"
    if not online.exists():
        print(f"missing online npz at {online}")
        return 2
    if not offline.exists():
        print(f"missing offline npz at {offline}")
        return 2

    on_t, on_left, on_hip = _union_ankle(online)
    of_t, of_left, of_hip = _union_ankle(offline)

    print(f"online  : {len(on_t)} frames, t [{on_t[0]:.3f}, {on_t[-1]:.3f}]")
    print(f"offline : {len(of_t)} frames, t [{of_t[0]:.3f}, {of_t[-1]:.3f}]")

    on_data = np.load(str(online))
    of_data = np.load(str(offline))
    print(f"online R det = {np.linalg.det(on_data['view_transform_R']):+.6f}")
    print(f"offline R det = {np.linalg.det(of_data['view_transform_R']):+.6f}")
    print(f"R diff (max abs): {np.max(np.abs(on_data['view_transform_R'] - of_data['view_transform_R'])):.6e}")
    print(f"t diff (max abs): {np.max(np.abs(on_data['view_transform_t'] - of_data['view_transform_t'])):.6e}")

    # Online and offline don't share an index space:
    # online appends to _recording_keypoints only when a detection
    # emits, so its index space is (per-detection-event); offline's
    # is (per-sync-index, including empty syncs). Match by NEAREST
    # TIMESTAMP instead — accurate to ~1/fps.
    print()
    print("matched by nearest timestamp (not by index):")
    print(f"{'on_t':>7} {'of_t':>7} {'dt':>6}   "
          f"{'on_L_x':>8} {'on_L_y':>8} {'on_L_z':>8}   "
          f"{'of_L_x':>8} {'of_L_y':>8} {'of_L_z':>8}   "
          f"{'|delta|':>8}")
    deltas: list[float] = []
    matched_count = 0
    for j in range(len(of_t)):
        if not np.isfinite(of_left[j, 0]):
            continue
        # Find online index with nearest timestamp.
        i = int(np.argmin(np.abs(on_t - of_t[j])))
        if not np.isfinite(on_left[i, 0]):
            continue
        dt = abs(float(on_t[i] - of_t[j]))
        if dt > 0.05:  # > 1 frame at 30 fps; not a real match
            continue
        d = float(np.linalg.norm(on_left[i] - of_left[j]))
        deltas.append(d)
        if matched_count < 20:
            print(
                f"{on_t[i]:>7.3f} {of_t[j]:>7.3f} {dt:>6.3f}   "
                f"{on_left[i, 0]:>+8.3f} {on_left[i, 1]:>+8.3f} {on_left[i, 2]:>+8.3f}   "
                f"{of_left[j, 0]:>+8.3f} {of_left[j, 1]:>+8.3f} {of_left[j, 2]:>+8.3f}   "
                f"{d:>8.3f}"
            )
        matched_count += 1

    if deltas:
        arr = np.asarray(deltas)
        print()
        print(f"timestamp-matched overlap: {len(deltas)} frames")
        print(f"  delta mean: {arr.mean():.3f} m")
        print(f"  delta median: {np.median(arr):.3f} m")
        print(f"  delta p95: {np.percentile(arr, 95):.3f} m")
        print(f"  delta max: {arr.max():.3f} m")

    return 0


if __name__ == "__main__":
    sys.exit(main())
