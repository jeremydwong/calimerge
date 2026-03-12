#!/usr/bin/env python3
"""
Convert ONNX models (YOLO v10s, VitPose-Base) to CoreML format for macOS.

Usage:
    python convert_models_coreml.py --yolo models/yolov10s.onnx --vitpose models/vitpose-base-simple.onnx --output models/coreml/

Requires: coremltools, onnx
    pip install coremltools onnx
"""

import argparse
import sys
from pathlib import Path

try:
    import coremltools as ct
    import numpy as np
except ImportError:
    print("ERROR: coremltools required. Install with: pip install coremltools")
    sys.exit(1)


def convert_yolo(onnx_path: str, output_dir: str, num_cameras: int = 4) -> str:
    """Convert YOLO v10s ONNX to CoreML mlpackage.

    YOLO v10s input:  (batch, 3, 640, 640) float32 or float16
    YOLO v10s output: (batch, 300, 6) float32 [x1, y1, x2, y2, conf, cls]

    We convert with a fixed batch size = num_cameras (typically 3-4).
    """
    out_path = str(Path(output_dir) / "yolov10s.mlpackage")
    print(f"Converting YOLO: {onnx_path} -> {out_path}")

    model = ct.converters.convert(
        onnx_path,
        inputs=[
            ct.TensorType(
                name="images",
                shape=(num_cameras, 3, 640, 640),
                dtype=np.float32,
            )
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )

    model.save(out_path)
    print(f"  Saved: {out_path}")
    return out_path


def convert_vitpose(onnx_path: str, output_dir: str, max_persons: int = 8) -> str:
    """Convert VitPose-Base ONNX to CoreML mlpackage.

    VitPose input:  (batch, 3, 256, 192) float32  (ImageNet-normalized)
    VitPose output: (batch, 52, 64, 48) float32    (heatmaps)

    Batch size is variable (= number of detected person crops).
    We use an EnumeratedShapes range to handle variable batch.
    """
    out_path = str(Path(output_dir) / "vitpose_base.mlpackage")
    print(f"Converting VitPose: {onnx_path} -> {out_path}")

    # CoreML doesn't support truly dynamic shapes on ANE.
    # Use EnumeratedShapes with common batch sizes.
    batch_sizes = [1, 2, 3, 4, 6, 8]
    batch_sizes = [b for b in batch_sizes if b <= max_persons]
    if max_persons not in batch_sizes:
        batch_sizes.append(max_persons)

    shapes = [(b, 3, 256, 192) for b in batch_sizes]

    model = ct.converters.convert(
        onnx_path,
        inputs=[
            ct.TensorType(
                name="input",
                shape=ct.EnumeratedShapes(shapes=shapes),
                dtype=np.float32,
            )
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )

    model.save(out_path)
    print(f"  Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX models to CoreML")
    parser.add_argument("--yolo", type=str, help="Path to YOLO v10s ONNX model")
    parser.add_argument("--vitpose", type=str, help="Path to VitPose-Base ONNX model")
    parser.add_argument("--output", type=str, default="models/coreml/",
                        help="Output directory for CoreML models")
    parser.add_argument("--num-cameras", type=int, default=4,
                        help="Fixed batch size for YOLO (= number of cameras)")
    parser.add_argument("--max-persons", type=int, default=8,
                        help="Max batch size for VitPose (= max detected persons)")
    args = parser.parse_args()

    if not args.yolo and not args.vitpose:
        parser.error("At least one of --yolo or --vitpose must be specified")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.yolo:
        convert_yolo(args.yolo, args.output, args.num_cameras)

    if args.vitpose:
        convert_vitpose(args.vitpose, args.output, args.max_persons)

    print("\nDone. Compile models for deployment:")
    print(f"  xcrun coremlcompiler compile {out}/*.mlpackage {out}/")


if __name__ == "__main__":
    main()
