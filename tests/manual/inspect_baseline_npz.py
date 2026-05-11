"""Inspect the baseline npz structure for regression test design."""
import subprocess, tempfile, numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURE = "tests/data/zelda_20260428_151934_fga_horizontal_head_turns"

for name in ("keypoints_3d.npz", "keypoints_3d.raw.npz"):
    # Use working-tree copy (just produced by a test run)
    src = REPO / FIXTURE / name
    if not src.exists():
        print(f"\n=== {name} === MISSING")
        continue
    d = np.load(str(src), allow_pickle=True)
    print(f"\n=== {name} ===")
    print(f"  keys: {d.files}")
    for k in d.files:
        arr = d[k]
        if arr.dtype.kind == 'f' and arr.size > 0:
            print(f"  {k}: shape={arr.shape} dtype={arr.dtype}"
                  f"  finite={np.isfinite(arr).sum()}/{arr.size}"
                  f"  range=[{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")
        elif arr.ndim == 0:
            print(f"  {k}: scalar  value={arr.item()!r}")
        else:
            print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")
