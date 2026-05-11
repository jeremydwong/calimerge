"""
Re-export the VitPose/SynthPose CoreML model with batch=4 instead of batch=16.

For a 2-camera rig tracking 1-2 people, batch=16 wastes ~75% of inference
computing zero-padded crops. Batch=4 gives headroom for 2 detections per
camera while cutting total compute roughly proportionally.

The existing batch=16 model is left untouched.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = REPO_ROOT / "models" / "onnx" / "vitpose_synthpose.onnx"
OUTPUT_PATH = REPO_ROOT / "models" / "coreml" / "vitpose_synthpose_rt_batch4.mlpackage"

BATCH = 4
INPUT_NAME = "input"
INPUT_SHAPE = (BATCH, 3, 256, 192)


def main() -> int:
    if not ONNX_PATH.exists():
        logger.error("ONNX model not found: %s", ONNX_PATH)
        return 1

    if OUTPUT_PATH.exists():
        logger.error("Output already exists: %s — delete it first if you want to re-convert", OUTPUT_PATH)
        return 1

    try:
        import coremltools as ct
        import onnx
        import onnx2torch
        import torch
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        return 1

    logger.info("Loading ONNX model: %s", ONNX_PATH)
    onnx_model = onnx.load(str(ONNX_PATH))

    # Strip allowzero attrs that onnx2torch doesn't handle
    for node in onnx_model.graph.node:
        if node.op_type == "Reshape":
            for i in range(len(node.attribute) - 1, -1, -1):
                if node.attribute[i].name == "allowzero":
                    del node.attribute[i]

    logger.info("Converting ONNX → PyTorch (onnx2torch)")
    torch_model = onnx2torch.convert(onnx_model).eval()

    logger.info("Tracing with input shape %s", INPUT_SHAPE)
    example = torch.randn(*INPUT_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(torch_model, example, strict=False)
        traced = torch.jit.freeze(traced)
        torch._C._jit_pass_inline(traced.graph)

    logger.info("Converting to CoreML mlprogram (FP16, cpuAndGPU)")
    mlmodel = ct.convert(
        traced,
        source="pytorch",
        inputs=[ct.TensorType(name=INPUT_NAME, shape=INPUT_SHAPE)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=ct.target.macOS14,
    )

    mlmodel.short_description = f"Calimerge vitpose/synthpose model (FP16=True, batch={BATCH})"
    mlmodel.author = "Calimerge"
    mlmodel.version = "1.0"

    logger.info("Saving to %s", OUTPUT_PATH)
    mlmodel.save(str(OUTPUT_PATH))

    logger.info("Done. Model saved with batch=%d", BATCH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
