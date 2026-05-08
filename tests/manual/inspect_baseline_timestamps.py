"""Print the timestamp range of the git-committed baseline npz."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

BASELINE = Path("/tmp/baseline_zelda_20260428_151934_fga_horizontal_head_turns_keypoints_3d.npz")


def main() -> int:
    if not BASELINE.exists():
        print(f"missing: {BASELINE}")
        return 1
    d = np.load(BASELINE)
    print(f"keys: {list(d.keys())}")
    ts = d["timestamps"]
    print(f"timestamps: shape={ts.shape}  min={ts.min():.3f}s  max={ts.max():.3f}s  span={ts.max()-ts.min():.3f}s")
    print(f"first 5: {ts[:5]}")
    print(f"last 5:  {ts[-5:]}")
    print(f"median delta: {np.median(np.diff(ts)):.4f}s  (1/fps would be ~0.0333s for 30fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
