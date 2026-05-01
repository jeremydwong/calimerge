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


def _convert(
    onnx_path: Path,
    output_path: Path,
    compute_units: str,
    fp16: bool,
    kind: str,
) -> Path:
    """Convert a single ONNX file to CoreML.

    The actual conversion happens here. Imports coremltools lazily so this
    module is import-clean on Windows without coremltools installed.
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    try:
        import coremltools as ct
    except ImportError as e:
        raise RuntimeError(
            "coremltools is required for ONNX→CoreML conversion. "
            "Install with `pip install coremltools` (macOS only). "
            f"Original error: {e}"
        ) from e

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Converting %s ONNX → CoreML: %s -> %s", kind, onnx_path, output_path)

    # coremltools 7+ accepts the source ONNX path directly via the ML
    # Program backend (mlpackage). For broader version support we go through
    # the unified ct.convert API.
    cu = _resolve_compute_units(compute_units)

    precision = ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32

    mlmodel = ct.convert(
        str(onnx_path),
        convert_to="mlprogram",
        compute_precision=precision,
        compute_units=cu,
        minimum_deployment_target=ct.target.macOS14,
    )

    # Tag with metadata so the consumer can sanity-check at load time.
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
