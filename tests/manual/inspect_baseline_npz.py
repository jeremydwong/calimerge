"""Inspect the baseline npz structure for regression test design."""
import subprocess, tempfile, numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURE = "tests/data/zelda_20260428_151934_fga_horizontal_head_turns"

for name in ("keypoints_3d.npz", "keypoints_3d.raw.npz"):
    git_path = f"{FIXTURE}/{name}"
    tmp = Path(tempfile.gettempdir()) / f"_baseline_{name}"
    subprocess.run(["git", "show", f"HEAD:{git_path}"], stdout=open(tmp, "wb"),
                   cwd=str(REPO), check=True)
    d = np.load(str(tmp))
    print(f"\n=== {name} ===")
    print(f"  keys: {d.files}")
    for k in d.files:
        arr = d[k]
        print(f"  {k}: shape={arr.shape} dtype={arr.dtype}"
              f"  finite={np.isfinite(arr).sum()}/{arr.size}"
              f"  range=[{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]"
              if arr.dtype.kind == 'f' else
              f"  {k}: shape={arr.shape} dtype={arr.dtype}")
