"""Quick: dump shape + per-track stats for git-baseline npz vs current."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ZELDA = REPO / "tests/data/zelda_20260428_151934_fga_horizontal_head_turns"
BASELINE = Path("/tmp/baseline_zelda.npz")


def _summary(label: str, path: Path) -> None:
    print(f"\n=== {label}: {path}")
    if not path.exists():
        print("  (missing)")
        return
    d = np.load(path)
    kps = d["keypoints_3d"]
    print(f"  shape: {kps.shape}  dtype: {kps.dtype}")
    if "model_backend" in d.files:
        print(f"  model_backend: {d['model_backend']}")
    if "model_name" in d.files:
        print(f"  model_name:    {d['model_name']}")
    finite = np.isfinite(kps).all(axis=-1)
    print(f"  finite (frame, person, kp): {finite.sum()}/{finite.size}")
    for p in range(kps.shape[1]):
        slot_valid = finite[:, p, :].any(axis=-1)
        if slot_valid.any():
            idx = np.where(slot_valid)[0]
            print(
                f"    slot {p}: {slot_valid.sum()} frames "
                f"(range {idx[0]}..{idx[-1]})"
            )


def main() -> int:
    _summary("baseline (git HEAD)", BASELINE)
    _summary("current dense (overwritten by last test)", ZELDA / "keypoints_3d.npz")
    _summary("current raw (overwritten or stale)", ZELDA / "keypoints_3d.raw.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
