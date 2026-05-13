#!/usr/bin/env bash
# macOS: build native library if stale, sync Python deps, then launch calimerge.
#
# Usage:
#   bash run_mac.sh              # launches GUI
#   bash run_mac.sh gui          # same
#   bash run_mac.sh clock        # sync verification clock
#   bash run_mac.sh --rebuild    # force-rebuild native library even if fresh
#   bash run_mac.sh --no-build   # skip native build entirely
#
# Everything after recognised flags is forwarded to `calimerge`.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "run_mac.sh: this script is macOS only (detected $(uname -s))" >&2
    exit 1
fi

FORCE_REBUILD=0
SKIP_BUILD=0
args=()
for a in "$@"; do
    case "$a" in
        --rebuild)  FORCE_REBUILD=1 ;;
        --no-build) SKIP_BUILD=1 ;;
        *)          args+=("$a") ;;
    esac
done

# Default subcommand is `gui` when nothing else was passed through.
if [[ ${#args[@]} -eq 0 ]]; then
    args=("gui")
fi

NATIVE_LIB="build/native/libcalimerge.dylib"
MPS_LIB="build/mps/libcalimerge_mps.dylib"

# Returns 0 (true) if $1 (the dylib) is missing or older than any matching
# source under $2 (the source dir). Globs are passed as remaining args.
is_stale() {
    local target="$1"; shift
    local src_dir="$1"; shift
    [[ ! -f "$target" ]] && return 0
    local found
    found=$(find "$src_dir" -type f \( "$@" \) -newer "$target" -print -quit 2>/dev/null || true)
    [[ -n "$found" ]]
}

# ---- native camera dylib ----
needs_native=0
if [[ "$SKIP_BUILD" == "1" ]]; then
    needs_native=0
elif [[ "$FORCE_REBUILD" == "1" ]]; then
    needs_native=1
elif is_stale "$NATIVE_LIB" "src/native" -name '*.mm' -o -name '*.cpp' -o -name '*.h'; then
    needs_native=1
fi
if [[ "$needs_native" == "1" ]]; then
    echo "→ building native camera dylib…"
    (cd src/native && ./build_macos.sh release)
else
    echo "→ native camera dylib up to date"
fi

# ---- mps pose pipeline dylib ----
needs_mps=0
if [[ "$SKIP_BUILD" == "1" ]]; then
    needs_mps=0
elif [[ "$FORCE_REBUILD" == "1" ]]; then
    needs_mps=1
elif is_stale "$MPS_LIB" "src/mps_pipeline" -name '*.m' -o -name '*.c' -o -name '*.h'; then
    needs_mps=1
elif is_stale "$MPS_LIB" "src/pt_shared" -name '*.cpp' -o -name '*.h'; then
    needs_mps=1
fi
if [[ "$needs_mps" == "1" ]]; then
    echo "→ building mps pose pipeline dylib…"
    bash src/mps_pipeline/build_mps.sh release
else
    echo "→ mps pose pipeline dylib up to date"
fi

# ---- coreml model artifacts (warning only — slow to build, not auto) ----
DATA_DIR="${CALIMERGE_DATA_DIR:-$HOME/Library/Application Support/Calimerge}"
COREML_DIR="$DATA_DIR/models/coreml"
if [[ ! -d "$COREML_DIR/yolo_v10s.mlpackage" || ! -d "$COREML_DIR/vitpose_synthpose.mlpackage" ]]; then
    echo "→ NOTE: CoreML mlpackages missing under $COREML_DIR/"
    echo "        MPS backend will not work until you run:  bash build_mac_models.sh"
fi

# Conda's VIRTUAL_ENV env var confuses uv into using the wrong Python.
unset VIRTUAL_ENV CONDA_PREFIX

# Catch Intel conda on an arm64 Mac before it poisons the whole venv.
if [[ "$(uname -m)" == "arm64" ]]; then
    bad_python="$(command -v python3 2>/dev/null || true)"
    if [[ -n "$bad_python" ]] && file -b "$bad_python" 2>/dev/null | grep -q x86_64; then
        echo "ERROR: Your default python3 is Intel (x86_64) on an arm64 Mac." >&2
        echo "       This is usually Intel Miniconda/Anaconda running under Rosetta." >&2
        echo "       It will pull the wrong wheels and break native extensions." >&2
        echo "" >&2
        echo "  Fix: uninstall Intel conda, or install the arm64 (Apple Silicon) version." >&2
        echo "       uv manages its own Python — conda is not needed for calimerge." >&2
        exit 1
    fi
fi

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
[[ -x "$UV_BIN" ]] || UV_BIN="$(command -v uv || true)"
if [[ -z "${UV_BIN:-}" ]]; then
    echo "run_mac.sh: uv not found (looked in ~/.local/bin and \$PATH)" >&2
    exit 1
fi

echo "→ uv sync"
"$UV_BIN" sync

echo "→ launching: calimerge ${args[*]:-}"
exec "$UV_BIN" run calimerge "${args[@]}"
