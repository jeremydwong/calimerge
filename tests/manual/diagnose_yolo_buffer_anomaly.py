"""
Inspect the YOLO mlpackage's raw output for anomalies.

The C-side filter at pt_preprocess.m:137 prints `top_conf=640` when
iterating all 300 detections — but reads correct values for det[0]
(conf=0.935). Either the model genuinely emits absurd values past
det[0], or the C-side reads garbage somewhere in the buffer.

This script answers it from the Python side: load the mlpackage, run
on a real frame stacked into a batch=2, and report:

  * max conf, max cls across all 300 detections per slot
  * count of detections with conf > 1.0 or cls > 80 (= sentinel signs
    of pixel coordinates leaking into the conf/cls fields)
  * the first detection where conf > 1.0 (if any), with surrounding
    context to spot layout drift

Run:
    uv run python3 tests/manual/diagnose_yolo_buffer_anomaly.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MLPKG = REPO_ROOT / "models" / "coreml" / "yolo_v10s.mlpackage"
ZELDA_DIR = REPO_ROOT / "tests" / "data" / "zelda_20260428_151934_fga_horizontal_head_turns"


def _frame() -> np.ndarray:
    cap = cv2.VideoCapture(str(sorted(ZELDA_DIR.glob("port_0*.mp4"))[0]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ok, f = cap.read()
    cap.release()
    assert ok
    return f


def _preproc(bgr: np.ndarray, size: int = 640) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = min(size / h, size / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(bgr, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    py, px = (size - nh) // 2, (size - nw) // 2
    canvas[py:py + nh, px:px + nw] = r
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def main() -> int:
    if not MLPKG.exists():
        print(f"missing: {MLPKG}")
        return 1

    import coremltools as ct
    ml = ct.models.MLModel(str(MLPKG))

    one = _preproc(_frame())
    batch = np.concatenate([one, one], axis=0)
    out = next(iter(ml.predict({"images": batch}).values())).astype(np.float32)
    print(f"output shape: {out.shape}  dtype: {out.dtype}")

    for slot in range(out.shape[0]):
        s = out[slot]  # (300, 6)
        conf = s[:, 4]
        cls = s[:, 5]
        print(f"\n=== slot {slot} ===")
        print(f"  conf:    min={conf.min():.4f}  max={conf.max():.4f}  mean={conf.mean():.4f}")
        print(f"  cls:     min={cls.min():.1f}   max={cls.max():.1f}   unique={len(np.unique(cls.astype(int)))}")
        print(f"  conf>1:  {(conf > 1.0).sum()}     cls>80: {(cls > 80).sum()}")

        # Top-5 detections by conf
        order = np.argsort(-conf)[:5]
        print(f"  top-5 by conf:")
        for r in order:
            print(
                f"    det[{int(r):3d}]: "
                f"box=({s[r,0]:6.1f}, {s[r,1]:6.1f}, {s[r,2]:6.1f}, {s[r,3]:6.1f})  "
                f"conf={s[r,4]:.4f}  cls={int(s[r,5])}"
            )

        # If any high "conf" looks like a pixel coordinate, dump that row + neighbours
        anomalies = np.where(conf > 1.0)[0]
        if len(anomalies) > 0:
            r = int(anomalies[0])
            lo, hi = max(0, r - 2), min(300, r + 3)
            print(f"  first anomaly at det[{r}] — context:")
            for i in range(lo, hi):
                marker = " ← " if i == r else "   "
                print(
                    f"    det[{i:3d}]:{marker}"
                    f"({s[i,0]:6.1f}, {s[i,1]:6.1f}, {s[i,2]:6.1f}, {s[i,3]:6.1f}, "
                    f"{s[i,4]:8.3f}, {s[i,5]:6.1f})"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
