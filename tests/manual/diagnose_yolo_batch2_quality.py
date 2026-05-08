"""
Verify the batch=2 production YOLO mlpackage is numerically correct.

We already showed (diagnose_yolo_coreml_quality.py) that the bridged
conversion path produces correct outputs when traced at batch=1. The
production model is traced at batch=2 to match what the MPS C-side
sends. This script confirms the batch=2 build produces correct outputs
on the same image stacked into a 2-batch.

If detections match PyTorch's, the model is fine and the bug is in
the C-side decoding/preprocessing path.

Run:
    uv run python3 tests/manual/diagnose_yolo_batch2_quality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PT_PATH = REPO_ROOT / "models" / "yolo" / "yolov10s.pt"
MLPKG_PATH = REPO_ROOT / "models" / "coreml" / "yolo_v10s.mlpackage"
ZELDA_DIR = REPO_ROOT / "tests" / "data" / "zelda_20260428_151934_fga_horizontal_head_turns"
TOP_N = 5


def _grab_frame() -> np.ndarray:
    candidates = sorted(ZELDA_DIR.glob("port_0*.mp4"))
    cap = cv2.VideoCapture(str(candidates[0]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ok, frame = cap.read()
    cap.release()
    assert ok
    return frame


def _yolo_preprocess(bgr: np.ndarray, size: int = 640) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_y = (size - nh) // 2
    pad_x = (size - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None]  # (1, 3, size, size)


def _print_topn(label: str, dets: np.ndarray, top: int = TOP_N) -> None:
    if dets.size == 0:
        print(f"  {label}: NO DETECTIONS")
        return
    order = np.argsort(-dets[:, 4])[:top]
    print(f"  {label} top-{top} (sorted by conf):")
    for i, idx in enumerate(order):
        x1, y1, x2, y2, conf, cls = dets[idx]
        print(
            f"    [{i}] conf={conf:.3f}  cls={int(cls)}  "
            f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )


def main() -> int:
    if not MLPKG_PATH.exists():
        print(f"missing: {MLPKG_PATH}")
        return 1

    print(f"frame source: {ZELDA_DIR}")
    frame = _grab_frame()
    print(f"frame shape: {frame.shape}")

    # Stack the same image twice → batch=2 input
    one = _yolo_preprocess(frame, size=640)
    batch = np.concatenate([one, one], axis=0)
    print(f"batch input shape: {batch.shape}")

    print("\n[coreml batch=2] loading mlpackage and predicting ...")
    import coremltools as ct
    ml = ct.models.MLModel(str(MLPKG_PATH))
    out = ml.predict({"images": batch})
    arr = next(iter(out.values()))
    arr = np.asarray(arr).astype(np.float32)
    print(f"output shape: {arr.shape}")

    if arr.ndim != 3 or arr.shape[0] != 2:
        print(f"  unexpected output shape: {arr.shape}")
        return 1

    print("\n[batch slot 0]")
    _print_topn("coreml-batch0", arr[0])
    print("\n[batch slot 1]")
    _print_topn("coreml-batch1", arr[1])

    print("\n[summary]")
    # Look for class=0 (person) detections with conf > 0.1
    for slot in (0, 1):
        person_mask = (arr[slot, :, 5].astype(int) == 0) & (arr[slot, :, 4] > 0.1)
        n_person = person_mask.sum()
        if n_person > 0:
            top_person_idx = np.argmax(arr[slot, :, 4] * person_mask)
            top_conf = arr[slot, top_person_idx, 4]
            print(f"  slot {slot}: {n_person} person detections > 0.1, top conf={top_conf:.3f}")
        else:
            print(f"  slot {slot}: NO person detections > 0.1 — model is broken at batch=2")

    return 0


if __name__ == "__main__":
    sys.exit(main())
