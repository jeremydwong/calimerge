"""
Diagnose whether the onnx2torch-bridged CoreML YOLO produces correct outputs.

The MPS pipeline previously loaded our bridged YOLO mlpackage cleanly but
returned 0 detections on the zelda recording. The PyTorch backend on the
same frame found 3 people. Question: is the converted CoreML model itself
broken, or is the C-side preprocessing wrong?

This script answers it by running both backends on the SAME frame and
printing top-N detections from each:

  1. PyTorch YOLO via ultralytics (canonical reference)
  2. Bridged CoreML YOLO via in-memory ct.convert (no save → no fork)

If the two agree, the model conversion is fine and the bug is C-side.
If they diverge, the onnx2torch + freeze + ct.convert chain lost fidelity.

Run:
    uv run python3 tests/manual/diagnose_yolo_coreml_quality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PT_PATH = REPO_ROOT / "models" / "yolo" / "yolov10s.pt"
ONNX_PATH = REPO_ROOT / "models" / "onnx" / "yolo_v10s.onnx"
ZELDA_DIR = REPO_ROOT / "tests" / "data" / "zelda_20260428_151934_fga_horizontal_head_turns"
TOP_N = 5


def _grab_frame() -> np.ndarray:
    """Pull a single BGR frame from port_0 of the zelda recording."""
    candidates = sorted(ZELDA_DIR.glob("port_0*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"no port_0*.mp4 in {ZELDA_DIR}")
    cap = cv2.VideoCapture(str(candidates[0]))
    # Skip into the recording so the subject is in frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame from {candidates[0]}")
    return frame  # BGR (H, W, 3)


def _yolo_preprocess(bgr: np.ndarray, size: int = 640) -> np.ndarray:
    """YOLO letterbox to (size, size), BGR→RGB, /255, NCHW float32."""
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


def _print_topn(label: str, detections: np.ndarray, top: int = TOP_N) -> None:
    """detections: (N, 6) — [x1, y1, x2, y2, conf, class]"""
    if detections.size == 0:
        print(f"  {label}: NO DETECTIONS")
        return
    order = np.argsort(-detections[:, 4])[:top]
    print(f"  {label} top-{top} (sorted by conf):")
    for i, idx in enumerate(order):
        x1, y1, x2, y2, conf, cls = detections[idx]
        print(
            f"    [{i}] conf={conf:.3f}  cls={int(cls)}  "
            f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})  "
            f"size={x2-x1:.0f}x{y2-y1:.0f}"
        )


def _pytorch_predict(bgr: np.ndarray) -> np.ndarray:
    """Run ultralytics YOLO and return (N, 6) detections in 640-letterboxed coords."""
    from ultralytics import YOLO
    model = YOLO(str(PT_PATH))
    res = model.predict(bgr, imgsz=640, conf=0.05, verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy().reshape(-1, 1)
    cls = boxes.cls.cpu().numpy().reshape(-1, 1)
    return np.concatenate([xyxy, conf, cls], axis=1)


def _build_bridged_coreml():
    """Reproduce the onnx2torch + trace + freeze + ct.convert chain in-memory."""
    import onnx
    import onnx2torch
    import torch
    import coremltools as ct

    onnx_model = onnx.load(str(ONNX_PATH))
    # strip Reshape allowzero attrs (same fix the production path uses)
    for node in onnx_model.graph.node:
        if node.op_type == "Reshape":
            for i in range(len(node.attribute) - 1, -1, -1):
                if node.attribute[i].name == "allowzero":
                    del node.attribute[i]

    torch_model = onnx2torch.convert(onnx_model).eval()
    example = torch.randn(1, 3, 640, 640, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(torch_model, example, strict=False)
        traced = torch.jit.freeze(traced)
        torch._C._jit_pass_inline(traced.graph)

    ml = ct.convert(
        traced,
        source="pytorch",
        inputs=[ct.TensorType(name="images", shape=(1, 3, 640, 640))],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS14,
    )
    return ml


def _coreml_predict(ml, preproc: np.ndarray) -> np.ndarray:
    """Run the in-memory mlmodel and decode YOLOv10 output into (N, 6)."""
    out = ml.predict({"images": preproc})
    arr = next(iter(out.values()))
    arr = np.asarray(arr).astype(np.float32)
    # YOLOv10 raw output: (1, 300, 6) — [x1, y1, x2, y2, conf, class]
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def main() -> int:
    if not PT_PATH.exists():
        print(f"missing: {PT_PATH}")
        return 1
    if not ONNX_PATH.exists():
        print(f"missing: {ONNX_PATH}")
        return 1

    print(f"frame source: {ZELDA_DIR}")
    frame = _grab_frame()
    print(f"frame shape: {frame.shape}  (BGR, will letterbox to 640x640)")

    print("\n[pytorch] running ultralytics YOLO on .pt ...")
    pt_dets = _pytorch_predict(frame)
    print(f"  total dets above conf=0.05: {len(pt_dets)}")
    _print_topn("pytorch", pt_dets)

    print("\n[coreml] building bridged mlmodel in-memory (no save) ...")
    ml = _build_bridged_coreml()
    print("  built. running prediction ...")
    preproc = _yolo_preprocess(frame, size=640)
    cm_dets = _coreml_predict(ml, preproc)
    cm_dets_filtered = cm_dets[cm_dets[:, 4] > 0.05]
    print(f"  total dets above conf=0.05: {len(cm_dets_filtered)}")
    _print_topn("coreml-bridged", cm_dets)

    print("\n[summary]")
    if len(pt_dets) > 0 and len(cm_dets_filtered) == 0:
        print("  → PyTorch finds people, bridged CoreML does not.")
        print("    Conclusion: the onnx2torch+ct.convert chain is producing")
        print("    a numerically broken model. Fix by switching YOLO to a")
        print("    different conversion path (Ultralytics direct works on")
        print("    a different env, or pre-shipped mlpackage).")
    elif len(pt_dets) > 0 and len(cm_dets_filtered) > 0:
        # Compare top scores
        pt_top = pt_dets[np.argmax(pt_dets[:, 4])]
        cm_top = cm_dets_filtered[np.argmax(cm_dets_filtered[:, 4])]
        print(f"  PyTorch top conf:        {pt_top[4]:.3f}")
        print(f"  CoreML  top conf:        {cm_top[4]:.3f}")
        if abs(pt_top[4] - cm_top[4]) < 0.1:
            print("  → Both backends agree. Bridged CoreML model is correct;")
            print("    bug is on the C-side (preprocessing / coordinate decoding).")
        else:
            print("  → Detections present but scores diverge significantly.")
            print("    Likely cause: FP16 quantisation drift or trace fidelity loss.")
    else:
        print("  → PyTorch found no detections — frame may be a bad sample.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
