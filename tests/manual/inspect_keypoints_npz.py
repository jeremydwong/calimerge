"""
Print summary stats of a keypoints_3d.npz dump.

Useful for diagnosing pipeline runs: tells you whether inference produced
any detections, how many person-slots are populated, and the per-slot
frame coverage. The script run_offline_pipeline_on_test_data.py already
does post-stitch reporting; this is the *pre-stitch* view.

Usage:
    uv run python3 tests/manual/inspect_keypoints_npz.py
       (defaults to the zelda recording in tests/data/)

    uv run python3 tests/manual/inspect_keypoints_npz.py path/to/keypoints_3d.raw.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


DEFAULT = (
    Path(__file__).resolve().parents[2]
    / "tests/data/zelda_20260428_151934_fga_horizontal_head_turns/keypoints_3d.raw.npz"
)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"not found: {path}")
        return 1
    print(f"file: {path}")

    d = np.load(str(path))
    print(f"keys: {list(d.keys())}")

    if "keypoints_3d" not in d.files:
        print("no keypoints_3d array; bailing")
        return 1

    kps = d["keypoints_3d"]
    pc = d["person_count"] if "person_count" in d.files else None

    n_frames, max_persons, n_kps, _ = kps.shape
    finite = np.isfinite(kps).all(axis=-1)
    print(f"\nshape: frames={n_frames} max_persons={max_persons} kps={n_kps}")
    print(f"finite-keypoint cells: {finite.sum()}/{finite.size}  ({100*finite.sum()/finite.size:.2f}%)")

    if pc is not None:
        print(f"\nperson_count: min={pc.min()} max={pc.max()} mean={pc.mean():.2f}")
        bins = np.bincount(pc, minlength=max(3, pc.max() + 1))
        for i, n in enumerate(bins):
            if n > 0:
                print(f"  {i} persons: {n} frames")

    print("\nper-slot coverage:")
    for p in range(max_persons):
        slot_valid = finite[:, p, :].any(axis=-1)
        if slot_valid.any():
            idx = np.where(slot_valid)[0]
            kpts_per_frame = finite[:, p, :].sum(axis=-1)
            avg_kp = kpts_per_frame[slot_valid].mean()
            print(
                f"  slot {p}: {slot_valid.sum()} frames "
                f"(idx {idx[0]}..{idx[-1]}), "
                f"avg {avg_kp:.1f}/{n_kps} keypoints per frame"
            )
        else:
            print(f"  slot {p}: empty")

    return 0


if __name__ == "__main__":
    sys.exit(main())
