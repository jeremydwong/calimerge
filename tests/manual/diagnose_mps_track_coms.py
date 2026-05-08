"""Print per-track Hip-COM trajectories for each MPS-produced track.

Lets us see whether the 4 fragmented MPS tracks are spatially close
(same person, fragmented) or spatially far (different detections).
If they're close, the retracker is stitching wrong; if far, the C
pipeline is producing spurious detections that retrack can't merge.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
NPZ = REPO / "tests/data/zelda_20260428_151934_fga_horizontal_head_turns/keypoints_3d.mps.npz"
L_HIP, R_HIP = 11, 12


def main() -> int:
    if not NPZ.exists():
        print(f"missing: {NPZ}")
        return 1

    d = np.load(NPZ)
    kps = d["keypoints_3d"]  # (N, P, K, 3)
    print(f"shape: {kps.shape}")

    for p in range(kps.shape[1]):
        finite_per_kp = np.isfinite(kps[:, p, :, :]).all(axis=-1)  # (N, K)
        slot_valid = finite_per_kp.any(axis=-1)
        if not slot_valid.any():
            continue
        idx = np.where(slot_valid)[0]
        print(f"\nslot {p}: {slot_valid.sum()} frames, range [{idx[0]}..{idx[-1]}]")

        # Per-frame Hip COM
        coms = []
        for i in idx:
            l = kps[i, p, L_HIP]
            r = kps[i, p, R_HIP]
            ok_l = np.isfinite(l).all()
            ok_r = np.isfinite(r).all()
            if ok_l and ok_r:
                coms.append((i, (l + r) * 0.5))
            elif ok_l:
                coms.append((i, l))
            elif ok_r:
                coms.append((i, r))

        if not coms:
            print("  (no valid hip COMs)")
            continue
        # Print first, last, mean, range
        first_i, first_c = coms[0]
        last_i, last_c = coms[-1]
        all_c = np.stack([c for _, c in coms], axis=0)
        c_min = all_c.min(axis=0)
        c_max = all_c.max(axis=0)
        c_mean = all_c.mean(axis=0)
        print(f"  first @ frame {first_i:>4}: ({first_c[0]:+.2f}, {first_c[1]:+.2f}, {first_c[2]:+.2f})")
        print(f"  last  @ frame {last_i:>4}: ({last_c[0]:+.2f}, {last_c[1]:+.2f}, {last_c[2]:+.2f})")
        print(f"  mean: ({c_mean[0]:+.2f}, {c_mean[1]:+.2f}, {c_mean[2]:+.2f})")
        print(f"  bbox: x[{c_min[0]:+.2f}, {c_max[0]:+.2f}]  y[{c_min[1]:+.2f}, {c_max[1]:+.2f}]  z[{c_min[2]:+.2f}, {c_max[2]:+.2f}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
