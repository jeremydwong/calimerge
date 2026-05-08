"""Compare two extrinsic sessions field by field to see if they differ at all."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    from calimerge.config import load_extrinsic_session

    for sid in (10, 11, 12, 13, 4, 1):
        loaded = load_extrinsic_session(sid)
        if loaded is None:
            print(f"id={sid}: not found")
            continue
        created_at, cams = loaded
        print(f"\n--- session id={sid}  created_at={created_at} ---")
        for port in sorted(cams.keys()):
            cam = cams[port]
            R = cam.extrinsics.rotation
            t = cam.extrinsics.translation
            K = cam.intrinsics.matrix
            print(f"  port {port}:")
            print(f"    K diag: ({K[0,0]:.2f}, {K[1,1]:.2f})  c=({K[0,2]:.2f}, {K[1,2]:.2f})")
            print(f"    t:    ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})")
            print(f"    R[0]: ({R[0,0]:+.4f}, {R[0,1]:+.4f}, {R[0,2]:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
