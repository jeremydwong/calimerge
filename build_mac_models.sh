#!/usr/bin/env bash
# macOS: build the model artifacts needed by the MPS pose pipeline.
#
# Two-stage chain, each idempotent (skips when the output already exists):
#
#   1. PyTorch  →  ONNX     via scripts/export_onnx.py + scripts/export_synthpose_onnx.py
#   2. ONNX     →  CoreML   via tests/manual/build_coreml_models.py
#
# Usage:
#   bash build_mac_models.sh           # build only what's missing
#   bash build_mac_models.sh --force   # rebuild everything from scratch
#
# Slow (~5-15 min for CoreML conversion). Run once per model-version bump,
# not every launch — that's why this is split out from run_mac.sh.
#
# Requires: macOS (uses --with coremltools so coremltools doesn't pollute
# the project lockfile).
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "build_mac_models.sh: macOS only (detected $(uname -s))" >&2
    exit 1
fi

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
[[ -x "$UV_BIN" ]] || UV_BIN="$(command -v uv || true)"
if [[ -z "${UV_BIN:-}" ]]; then
    echo "build_mac_models.sh: uv not found" >&2
    exit 1
fi

ONNX_DIR="models/onnx"
COREML_DIR="models/coreml"
mkdir -p "$ONNX_DIR" "$COREML_DIR"

YOLO_ONNX="$ONNX_DIR/yolo_v10s.onnx"
VITPOSE_ONNX="$ONNX_DIR/vitpose_synthpose.onnx"
YOLO_MLPKG="$COREML_DIR/yolo_v10s.mlpackage"
VITPOSE_MLPKG="$COREML_DIR/vitpose_synthpose.mlpackage"

# ---- Stage 1: PyTorch → ONNX ----
if [[ "$FORCE" == "1" ]] || [[ ! -f "$YOLO_ONNX" ]]; then
    echo "→ exporting YOLO v10s → ONNX"
    "$UV_BIN" run python3 scripts/export_onnx.py
else
    echo "→ $YOLO_ONNX exists (skip)"
fi

if [[ "$FORCE" == "1" ]] || [[ ! -f "$VITPOSE_ONNX" ]]; then
    echo "→ exporting VitPose SynthPose → ONNX"
    "$UV_BIN" run python3 scripts/export_synthpose_onnx.py
else
    echo "→ $VITPOSE_ONNX exists (skip)"
fi

# ---- Stage 2: ONNX → CoreML ----
need_coreml=0
if [[ "$FORCE" == "1" ]] || [[ ! -d "$YOLO_MLPKG" ]] || [[ ! -d "$VITPOSE_MLPKG" ]]; then
    need_coreml=1
fi

if [[ "$need_coreml" == "1" ]]; then
    echo "→ converting ONNX → CoreML (this is the slow step, 5-15 min)"
    # coremltools dropped ONNX ingestion in v6, so we bridge via onnx2torch.
    # Both deps are pulled in transiently with --with so they don't pollute uv.lock.
    "$UV_BIN" run --with coremltools --with onnx2torch python3 tests/manual/build_coreml_models.py
else
    echo "→ CoreML mlpackages exist (skip)"
fi

echo
echo "Done."
echo "  $YOLO_ONNX"
echo "  $VITPOSE_ONNX"
echo "  $YOLO_MLPKG"
echo "  $VITPOSE_MLPKG"
