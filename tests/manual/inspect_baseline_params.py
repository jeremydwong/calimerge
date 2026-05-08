"""Verify the new baseline npz has params + 4 stitched tracks."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
NPZ = REPO / "tests/data/zelda_20260428_151934_fga_horizontal_head_turns/keypoints_3d.npz"


def main() -> int:
    if not NPZ.exists():
        print(f"missing: {NPZ}")
        return 1
    d = np.load(NPZ)
    print(f"keys: {list(d.keys())}")
    print(f"shape: {d['keypoints_3d'].shape}")
    for k in (
        "model_backend", "model_name",
        "person_confidence", "max_track_distance", "track_patience",
    ):
        if k in d.files:
            v = d[k]
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: MISSING")

    finite = np.isfinite(d["keypoints_3d"]).all(axis=-1)
    print(f"\nper-slot coverage:")
    for p in range(d["keypoints_3d"].shape[1]):
        valid = finite[:, p, :].any(axis=-1)
        if valid.any():
            idx = np.where(valid)[0]
            print(f"  slot {p}: {valid.sum()} frames (idx {idx[0]}..{idx[-1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
