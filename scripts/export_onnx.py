"""
Export YOLO v10s and VitPose-Base ONNX models for TensorRT.

Produces two ONNX files suitable for the CUDA pose tracking pipeline:
  - yolo_v10s.onnx          (dynamic batch, 640x640 input)
  - vitpose_base_coco_wholebody.onnx  (dynamic batch, 256x192 input, 52 keypoints)

Models are auto-downloaded if not already present.

Usage:
    uv run python scripts/export_onnx.py [--output-dir DIR]

The output directory defaults to models/onnx/ at the project root.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def export_yolo(output_dir: Path) -> Path:
    """
    Export YOLO v10s to ONNX format.

    Uses the ultralytics library to download the model (if needed) and
    export it with dynamic batch dimension and opset 17.

    Returns:
        Path to the exported ONNX file.
    """
    from ultralytics import YOLO

    # Model source: ultralytics will auto-download yolov10s.pt
    model_filename = "yolov10s.pt"
    model_cache = output_dir.parent / "yolo"
    model_cache.mkdir(parents=True, exist_ok=True)
    model_path = model_cache / model_filename

    if not model_path.exists():
        # Download from GitHub releases
        download_url = (
            "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt"
        )
        print(f"Downloading YOLO v10s from: {download_url}")
        import urllib.request
        urllib.request.urlretrieve(download_url, str(model_path))
        print(f"Saved to: {model_path}")
    else:
        print(f"Using cached YOLO model: {model_path}")

    print("Exporting YOLO v10s to ONNX...")
    model = YOLO(str(model_path))

    # Export with dynamic batch dimension for TensorRT
    # opset 17 is required for TensorRT 10+ compatibility
    export_path = model.export(
        format="onnx",
        opset=17,
        dynamic=True,
        simplify=True,
    )

    # Move to output directory with canonical name
    src = Path(export_path)
    dst = output_dir / "yolo_v10s.onnx"
    if src != dst:
        import shutil
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    print(f"YOLO ONNX exported to: {dst}")
    print(f"  Input:  images (batch, 3, 640, 640)")
    print(f"  Output: detections (batch, 300, 6)  [x1, y1, x2, y2, conf, class]")
    return dst


def export_vitpose(output_dir: Path) -> Path:
    """
    Export VitPose-Base (COCO, 17 keypoints) to ONNX format.

    Uses the HuggingFace transformers library to download the model
    and exports with dynamic batch dimension.

    The model is usyd-community/vitpose-base-simple, which outputs
    17 COCO keypoints. The CSV export pads to 52 SynthPose columns
    with NaN (matching the Python pipeline behavior).

    Returns:
        Path to the exported ONNX file.
    """
    import torch
    from transformers import VitPoseForPoseEstimation

    model_id = "usyd-community/vitpose-base-simple"

    # Check for local cache
    local_dir = output_dir.parent / "vitpose"
    config_file = local_dir / "config.json"

    if config_file.exists():
        print(f"Loading VitPose from local cache: {local_dir}")
        model = VitPoseForPoseEstimation.from_pretrained(
            str(local_dir), local_files_only=True
        )
    else:
        print(f"Downloading VitPose from HuggingFace: {model_id}")
        model = VitPoseForPoseEstimation.from_pretrained(model_id)
        # Save locally for future runs
        local_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(local_dir))
        print(f"VitPose saved to: {local_dir}")

    model.eval()
    model.cpu()

    # VitPose input: (batch, 3, 256, 192) - height x width
    # This matches PT_VITPOSE_INPUT_H=256, PT_VITPOSE_INPUT_W=192 in pt_common.h
    dummy_input = torch.randn(1, 3, 256, 192)

    output_path = output_dir / "vitpose_base_coco.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Exporting VitPose to ONNX...")

    # The VitPose HuggingFace model wraps the backbone + head.
    # We need to export the underlying torch model, not the HF wrapper,
    # because the HF forward() method returns a dict, not raw tensors.
    # Use a wrapper that extracts heatmaps from the model output.

    class VitPoseONNXWrapper(torch.nn.Module):
        """Wrapper that returns raw heatmap tensor for ONNX export."""

        def __init__(self, hf_model):
            super().__init__()
            self.model = hf_model

        def forward(self, pixel_values):
            outputs = self.model(pixel_values=pixel_values)
            # VitPoseForPoseEstimation returns VitPoseEstimatorOutput
            # with .heatmaps attribute: (batch, num_keypoints, H, W)
            return outputs.heatmaps

    wrapper = VitPoseONNXWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        input_names=["images"],
        output_names=["heatmaps"],
        dynamic_axes={
            "images": {0: "batch"},
            "heatmaps": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify the export
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    print(f"VitPose ONNX exported to: {output_path}")
    print(f"  Input:  images   (batch, 3, 256, 192)")
    print(f"  Output: heatmaps (batch, 17, 64, 48)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLO v10s and VitPose-Base to ONNX for the CUDA pipeline."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for ONNX files (default: models/onnx/)",
    )
    parser.add_argument(
        "--yolo-only",
        action="store_true",
        help="Only export YOLO model",
    )
    parser.add_argument(
        "--vitpose-only",
        action="store_true",
        help="Only export VitPose model",
    )

    args = parser.parse_args()

    # Default output directory: project_root/models/onnx/
    if args.output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "models" / "onnx"
    else:
        output_dir = args.output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    export_both = not args.yolo_only and not args.vitpose_only

    if export_both or args.yolo_only:
        print("=" * 60)
        print("  YOLO v10s Export")
        print("=" * 60)
        try:
            yolo_path = export_yolo(output_dir)
            file_size = yolo_path.stat().st_size / (1024 * 1024)
            print(f"  Size: {file_size:.1f} MB")
        except Exception as e:
            print(f"ERROR exporting YOLO: {e}", file=sys.stderr)
            if not export_both:
                return 1
        print()

    if export_both or args.vitpose_only:
        print("=" * 60)
        print("  VitPose-Base (SynthPose 52kp) Export")
        print("=" * 60)
        try:
            vitpose_path = export_vitpose(output_dir)
            file_size = vitpose_path.stat().st_size / (1024 * 1024)
            print(f"  Size: {file_size:.1f} MB")
        except Exception as e:
            print(f"ERROR exporting VitPose: {e}", file=sys.stderr)
            if not export_both:
                return 1
        print()

    print("=" * 60)
    print("  Export complete.")
    print()
    print("  Copy the ONNX files to your recording directory or")
    print("  set the paths via --yolo / --vitpose in pt_main.exe,")
    print("  or pass yolo_onnx / vitpose_onnx to run_cuda_pipeline().")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
