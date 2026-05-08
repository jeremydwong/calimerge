"""
Build the CoreML .mlpackages for the macOS MPS pipeline.

Run on macOS (with coremltools installed) to materialise:

    models/coreml/yolo_v10s.mlpackage
    models/coreml/vitpose_synthpose.mlpackage

from the ONNX sources under models/onnx/ (or the user's data dir).

This is the macOS counterpart to the TensorRT engine build that happens
automatically on the CUDA pipeline's first run. It must be invoked
manually because:

  - coremltools is not installed in the standard calimerge dev env
  - conversion takes a few minutes and only needs to happen on
    model-version bumps

Usage:

    python tests/manual/build_coreml_models.py
    # or, with explicit ONNX paths:
    python tests/manual/build_coreml_models.py \
        --yolo-onnx models/onnx/yolo_v10s.onnx \
        --vitpose-onnx models/onnx/vitpose_synthpose.onnx

The script is import-clean on Windows; it just prints a friendly
message and exits non-zero if coremltools cannot be imported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_onnx(filename: str, override: Path | None) -> Path:
    if override is not None:
        return override

    # 1. App data dir (preferred)
    try:
        from calimerge.config import models_dir
        primary = models_dir() / "onnx" / filename
        if primary.exists():
            return primary
    except Exception:
        pass

    # 2. Repo's models/onnx/
    legacy = REPO_ROOT / "models" / "onnx" / filename
    return legacy


def _resolve_yolo_pt(override: Path | None) -> Path:
    """Locate the Ultralytics YOLO .pt checkpoint."""
    if override is not None:
        return override

    try:
        from calimerge.config import models_dir
        primary = models_dir() / "yolo" / "yolov10s.pt"
        if primary.exists():
            return primary
    except Exception:
        pass

    return REPO_ROOT / "models" / "yolo" / "yolov10s.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-onnx", type=Path, default=None)
    parser.add_argument("--vitpose-onnx", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "models" / "coreml",
        help="Where to write the .mlpackages (default: <repo>/models/coreml)",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 quantisation",
    )
    args = parser.parse_args(argv)

    from calimerge.tracking.convert_onnx_to_coreml import (
        convert_yolo_to_coreml,
        convert_vitpose_to_coreml,
        is_coremltools_available,
    )

    if not is_coremltools_available():
        print(
            "coremltools is not installed in this environment.\n"
            "Run this on macOS to materialise the .mlpackages:\n"
            "    pip install coremltools\n"
            "    python tests/manual/build_coreml_models.py"
        )
        return 1

    fp16 = not args.no_fp16

    yolo_onnx = _resolve_onnx("yolo_v10s.onnx", args.yolo_onnx)
    vitpose_onnx = _resolve_onnx("vitpose_synthpose.onnx", args.vitpose_onnx)

    print(f"YOLO ONNX:    {yolo_onnx}")
    print(f"VitPose ONNX: {vitpose_onnx}")
    print(f"Output dir:   {args.output_dir}")
    print(f"FP16:         {fp16}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not yolo_onnx.exists():
        print(f"ERROR: YOLO ONNX not found at {yolo_onnx}")
        return 2
    if not vitpose_onnx.exists():
        print(f"ERROR: VitPose ONNX not found at {vitpose_onnx}")
        return 2

    print("Converting YOLO v10s (onnx2torch + ct.convert bridge) ...")
    yolo_out = convert_yolo_to_coreml(
        yolo_onnx,
        args.output_dir / "yolo_v10s.mlpackage",
        fp16=fp16,
    )
    print(f"  -> {yolo_out}")

    print("Converting VitPose ...")
    vitpose_out = convert_vitpose_to_coreml(
        vitpose_onnx,
        args.output_dir / "vitpose_synthpose.mlpackage",
        fp16=fp16,
    )
    print(f"  -> {vitpose_out}")

    print()
    print("Done. The .mlpackages are first-run-cached by CoreML on each")
    print("Mac, so the first GUI launch with backend=Hardware (MPS) will")
    print("take an extra ~5-15s while CoreML compiles to .mlmodelc.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
