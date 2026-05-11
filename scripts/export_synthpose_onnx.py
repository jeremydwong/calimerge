"""
Export the SynthPose VitPose model to ONNX for use with the CUDA TensorRT pipeline.

Usage:
    uv run python scripts/export_synthpose_onnx.py

Output:
    models/onnx/vitpose_synthpose.onnx

The exported model takes input shape (batch, 3, 256, 192) and outputs
(batch, 52, 64, 48) heatmaps — 52 SynthPose keypoints.

Requirements:
    - transformers, torch, onnx (already in pyproject.toml)
    - Internet access to download the model on first run
"""

import sys
from pathlib import Path

import torch


def export(output_dir: Path | None = None):
    from transformers import VitPoseForPoseEstimation

    model_id = "stanfordmimi/synthpose-vitpose-base-hf"
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "models" / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vitpose_synthpose.onnx"

    print(f"Loading model: {model_id}")
    model = VitPoseForPoseEstimation.from_pretrained(model_id)
    model.eval()

    # Input: (batch, 3, 256, 192) matching PT_VITPOSE_INPUT_H/W
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 256, 192)

    print(f"Exporting to: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # Verify output shape
    import onnx
    m = onnx.load(str(output_path))
    output_shape = m.graph.output[0].type.tensor_type.shape
    dims = [d.dim_value or d.dim_param for d in output_shape.dim]
    print(f"Output shape: {dims}")
    print(f"Expected: [batch, 52, 64, 48]")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    print("Done. Use this model with the CUDA pipeline:")
    print(f"  --vitpose {output_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: <repo>/models/onnx/)")
    args = p.parse_args()
    export(output_dir=args.output_dir)
