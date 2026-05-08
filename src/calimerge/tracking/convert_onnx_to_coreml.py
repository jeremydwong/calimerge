"""
Convert ONNX models to CoreML .mlpackage for the macOS MPS pipeline.

The CUDA pipeline ships ONNX models that TensorRT compiles into FP16 engines
on first run. The MPS (Apple Silicon) pipeline needs the equivalent ONNX
models converted to CoreML so the Apple Neural Engine and integrated GPU can
execute them.

Usage (from a Mac with coremltools installed):

    from calimerge.tracking.convert_onnx_to_coreml import (
        convert_yolo_to_coreml, convert_vitpose_to_coreml,
    )
    convert_yolo_to_coreml(
        "models/onnx/yolo_v10s.onnx",
        "models/coreml/yolo_v10s.mlpackage",
    )
    convert_vitpose_to_coreml(
        "models/onnx/vitpose_synthpose.onnx",
        "models/coreml/vitpose_synthpose.mlpackage",
    )

This module is import-clean on Windows even when coremltools is not
installed; the actual conversion call lazily imports it and surfaces a
clear "run this on macOS" message if the import fails.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Compute-unit targets ──
#
# We deliberately split YOLO and VitPose:
#
#   YOLO v10s:        cpuAndNeuralEngine — pure conv, fits the ANE well, low
#                     latency the ANE was built for.
#   VitPose (transformer): cpuAndGPU — transformer attention is awkward on the
#                     ANE and frequently runs faster on the M-series GPU.
#
# Caller can override via the ``compute_units`` argument.
COMPUTE_UNIT_YOLO_DEFAULT = "cpuAndNeuralEngine"
COMPUTE_UNIT_VITPOSE_DEFAULT = "cpuAndGPU"


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────


def convert_yolo_to_coreml(
    onnx_path: str | Path,
    output_path: str | Path,
    compute_units: str = COMPUTE_UNIT_YOLO_DEFAULT,
    fp16: bool = True,
) -> Path:
    """Convert the YOLOv10 ONNX detector to a CoreML .mlpackage.

    Goes via the onnx2torch + jit.trace + ct.convert bridge (same path as
    VitPose). The bridge is numerically correct — verified by a side-by-side
    diagnostic against ultralytics PyTorch inference (see
    tests/manual/diagnose_yolo_coreml_quality.py). Ultralytics' direct
    CoreML exporter would be cleaner architecturally, but its save phase
    forks an ObjC-touching child that the macOS fork-safety check kills.

    Args:
        onnx_path: Path to ``yolo_v10s.onnx``.
        output_path: Path to write ``yolo_v10s.mlpackage``.
        compute_units: One of ``"cpuOnly"``, ``"cpuAndGPU"``,
            ``"cpuAndNeuralEngine"``, ``"all"``.
        fp16: If True (default), quantise weights to FP16.

    Returns:
        The output ``.mlpackage`` path.
    """
    return _convert(
        onnx_path=Path(onnx_path),
        output_path=Path(output_path),
        compute_units=compute_units,
        fp16=fp16,
        kind="yolo",
    )


def convert_vitpose_to_coreml(
    onnx_path: str | Path,
    output_path: str | Path,
    compute_units: str = COMPUTE_UNIT_VITPOSE_DEFAULT,
    fp16: bool = True,
) -> Path:
    """Convert the VitPose SynthPose ONNX model to a CoreML .mlpackage.

    Args:
        onnx_path: Path to ``vitpose_synthpose.onnx``.
        output_path: Path to write ``vitpose_synthpose.mlpackage``.
        compute_units: See :func:`convert_yolo_to_coreml`.
        fp16: If True (default), quantise weights to FP16.

    Returns:
        The output ``.mlpackage`` path.
    """
    return _convert(
        onnx_path=Path(onnx_path),
        output_path=Path(output_path),
        compute_units=compute_units,
        fp16=fp16,
        kind="vitpose",
    )


def is_coremltools_available() -> bool:
    """Return True if coremltools can be imported in the current environment."""
    try:
        import coremltools  # noqa: F401
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────────


def _resolve_compute_units(name: str):
    """Map a string name to a ``coremltools.ComputeUnit`` enum value."""
    import coremltools as ct

    table = {
        "cpuOnly": ct.ComputeUnit.CPU_ONLY,
        "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
        "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
    }
    if name not in table:
        raise ValueError(
            f"Unknown compute_units {name!r}; valid: {sorted(table)}"
        )
    return table[name]


# Per-kind input contract: (input_name, (batch, C, H, W)).
# Used for jit-tracing the ONNX-derived PyTorch model. coremltools dropped
# native ONNX ingestion in v6+, so we go ONNX → PyTorch (via onnx2torch) →
# trace → ct.convert(source="pytorch").
#
# Batch is FIXED here, not dynamic, because onnx2torch bakes literal batch
# constants into intermediate Reshape ops (e.g. YOLO's anchor-grid reshape
# becomes (1, 4, 128, 400) regardless of input batch). RangeDim does not
# override that. Sized for the streaming pipeline:
#
#   YOLO:     batch = num_cameras           (one frame per camera per sync)
#   VitPose:  batch = PT_MAX_DETECTIONS     (one crop per detected person across all cams;
#                                            C side zero-pads to this fixed batch)
#
# Bump _NUM_CAMERAS_DEFAULT in lockstep with the rig you're targeting.
# _VITPOSE_MAX_BATCH must mirror PT_MAX_DETECTIONS in src/pt_shared/pt_common.h.
_NUM_CAMERAS_DEFAULT = 2
_VITPOSE_MAX_BATCH = 16

_INPUT_SHAPES: dict[str, tuple[str, tuple[int, int, int, int]]] = {
    "yolo":    ("images", (_NUM_CAMERAS_DEFAULT, 3, 640, 640)),
    "vitpose": ("input",  (_VITPOSE_MAX_BATCH, 3, 256, 192)),
}


def _strip_reshape_allowzero(onnx_model) -> None:
    for node in onnx_model.graph.node:
        if node.op_type != "Reshape":
            continue
        for i in range(len(node.attribute) - 1, -1, -1):
            if node.attribute[i].name == "allowzero":
                del node.attribute[i]


def _convert(
    onnx_path: Path,
    output_path: Path,
    compute_units: str,
    fp16: bool,
    kind: str,
) -> Path:
    """Convert a single ONNX file to CoreML via an in-memory PyTorch bridge."""
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if kind not in _INPUT_SHAPES:
        raise ValueError(f"unknown kind {kind!r}; valid: {sorted(_INPUT_SHAPES)}")

    try:
        import coremltools as ct
    except ImportError as e:
        raise RuntimeError(
            "coremltools is required for ONNX→CoreML conversion. "
            "Install with `pip install coremltools` (macOS only). "
            f"Original error: {e}"
        ) from e

    try:
        import onnx
        import onnx2torch
        import torch
    except ImportError as e:
        raise RuntimeError(
            "onnx2torch + onnx + torch are required to bridge ONNX→PyTorch. "
            "Install with `pip install onnx2torch onnx torch` "
            "(or run via `uv run --with onnx2torch ...`). "
            f"Original error: {e}"
        ) from e

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_name, input_shape = _INPUT_SHAPES[kind]

    logger.info(
        "Converting %s ONNX → CoreML: %s -> %s (input %s shape %s)",
        kind, onnx_path, output_path, input_name, input_shape,
    )

    onnx_model = onnx.load(str(onnx_path))
    _strip_reshape_allowzero(onnx_model)
    torch_model = onnx2torch.convert(onnx_model).eval()

    example = torch.randn(*input_shape, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(torch_model, example, strict=False)
        # Freeze + inline graph passes fold away view-ops like aten::resolve_conj
        # that coremltools' torch frontend doesn't implement. Without this the
        # convert step trips on phantom complex-number plumbing inserted by the
        # tracer for real-tensor view operations.
        traced = torch.jit.freeze(traced)
        torch._C._jit_pass_inline(traced.graph)

    cu = _resolve_compute_units(compute_units)
    precision = ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced,
        source="pytorch",
        inputs=[ct.TensorType(name=input_name, shape=input_shape)],
        convert_to="mlprogram",
        compute_precision=precision,
        compute_units=cu,
        minimum_deployment_target=ct.target.macOS14,
    )

    mlmodel.short_description = f"Calimerge {kind} model (FP16={fp16})"
    mlmodel.author = "Calimerge"
    mlmodel.version = "1.0"

    mlmodel.save(str(output_path))

    logger.info("Saved %s", output_path)
    return output_path


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Tiny CLI for ad-hoc conversions. Use the manual test in
    ``tests/manual/build_coreml_models.py`` for the full conversion."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert ONNX → CoreML for Calimerge")
    parser.add_argument("--yolo-onnx", type=Path, default=None, help="YOLO v10s ONNX path")
    parser.add_argument("--vitpose-onnx", type=Path, default=None, help="VitPose ONNX path")
    parser.add_argument("--output-dir", type=Path, default=Path("models/coreml"))
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args(argv)

    if not is_coremltools_available():
        print("coremltools is not installed in this environment.")
        print("Run this on macOS to materialise the .mlpackages.")
        return 1

    fp16 = not args.no_fp16
    if args.yolo_onnx is not None:
        convert_yolo_to_coreml(
            args.yolo_onnx,
            args.output_dir / "yolo_v10s.mlpackage",
            fp16=fp16,
        )
    if args.vitpose_onnx is not None:
        convert_vitpose_to_coreml(
            args.vitpose_onnx,
            args.output_dir / "vitpose_synthpose.mlpackage",
            fp16=fp16,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
