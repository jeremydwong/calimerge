#!/usr/bin/env bash
# Windows (Git Bash): build native + CUDA DLLs if stale, sync Python deps,
# then launch calimerge.
#
# Usage:
#   bash run_win.sh                # launches GUI
#   bash run_win.sh gui            # same
#   bash run_win.sh clock          # sync verification clock
#   bash run_win.sh --rebuild      # force-rebuild both native + CUDA libs
#   bash run_win.sh --no-build     # skip both builds entirely
#   bash run_win.sh --no-cuda-build # build only the camera lib, skip CUDA
#
# Everything after recognised flags is forwarded to `calimerge`.
set -euo pipefail
cd "$(dirname "$0")"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) ;;
    *)
        echo "run_win.sh: this script is Windows (Git Bash) only (detected $(uname -s))" >&2
        exit 1
        ;;
esac

FORCE_REBUILD=0
SKIP_BUILD=0
SKIP_CUDA_BUILD=0
args=()
for a in "$@"; do
    case "$a" in
        --rebuild)        FORCE_REBUILD=1 ;;
        --no-build)       SKIP_BUILD=1 ;;
        --no-cuda-build)  SKIP_CUDA_BUILD=1 ;;
        *)                args+=("$a") ;;
    esac
done

if [[ ${#args[@]} -eq 0 ]]; then
    args=("gui")
fi

# ─── Camera native lib (always required) ──────────────────────────────
NATIVE_LIB="build/native/calimerge.dll"

needs_build=0
if [[ "$SKIP_BUILD" == "1" ]]; then
    needs_build=0
elif [[ "$FORCE_REBUILD" == "1" ]]; then
    needs_build=1
elif [[ ! -f "$NATIVE_LIB" ]]; then
    echo "→ native library missing"
    needs_build=1
elif [[ -n "$(find src/native -type f \( -name '*.cpp' -o -name '*.c' -o -name '*.h' \) -newer "$NATIVE_LIB" -print -quit 2>/dev/null)" ]]; then
    echo "→ native source newer than $NATIVE_LIB"
    needs_build=1
fi

if [[ "$needs_build" == "1" ]]; then
    echo "→ building native library…"
    (cd src/native && cmd //c build_win32.bat release)
else
    echo "→ native library up to date"
fi

# ─── CUDA pipeline lib (optional — only when toolchain is present) ────
# Hardcoded toolchain locations on this machine. If you install the
# toolchain elsewhere, edit these paths or replace this whole block with
# auto-detection later.
CUDA_TOOLKIT_PATH='/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9'
TENSORRT_PATH_WIN='/c/TensorRT'
OPENCV_PATH_WIN='/c/OpenCV/opencv/build'
CUDA_LIB="build/cuda/calimerge_cuda.dll"

cuda_check_paths=(
    "$CUDA_TOOLKIT_PATH/bin/nvcc.exe"
    "$TENSORRT_PATH_WIN/lib/nvinfer_10.dll"
    "$OPENCV_PATH_WIN/x64/vc16/bin"
)
cuda_toolchain_present=1
cuda_missing=()
for p in "${cuda_check_paths[@]}"; do
    if ! ls "$p"* >/dev/null 2>&1; then
        cuda_toolchain_present=0
        cuda_missing+=("$p")
    fi
done

if [[ "$SKIP_BUILD" == "1" || "$SKIP_CUDA_BUILD" == "1" ]]; then
    echo "→ skipping CUDA build (flag)"
elif [[ "$cuda_toolchain_present" != "1" ]]; then
    echo "→ skipping CUDA build (toolchain not found):"
    for p in "${cuda_missing[@]}"; do
        echo "    missing: $p"
    done
    echo "  (PyTorch detection backend will work; CUDA TensorRT backend will not)"
else
    cuda_needs_build=0
    if [[ "$FORCE_REBUILD" == "1" ]]; then
        cuda_needs_build=1
    elif [[ ! -f "$CUDA_LIB" ]]; then
        echo "→ CUDA library missing"
        cuda_needs_build=1
    elif [[ -n "$(find src/cuda_pipeline src/pt_shared -type f \( -name '*.cpp' -o -name '*.cu' -o -name '*.h' \) -newer "$CUDA_LIB" -print -quit 2>/dev/null)" ]]; then
        echo "→ CUDA source newer than $CUDA_LIB"
        cuda_needs_build=1
    fi

    if [[ "$cuda_needs_build" == "1" ]]; then
        echo "→ building CUDA pipeline (this may take a minute)…"
        # build_cuda_win32.bat sets up MSVC + CUDA + TensorRT + OpenCV env on
        # its own. Pass our hardcoded paths via env so it picks them up.
        export TENSORRT_PATH='C:\TensorRT'
        export OPENCV_PATH='C:\OpenCV\opencv\build'
        (cd src/cuda_pipeline && cmd //c build_cuda_win32.bat release)
    else
        echo "→ CUDA library up to date"
    fi
fi

# Anaconda sets VIRTUAL_ENV / CONDA_PREFIX to its base path on Windows;
# uv picks the wrong interpreter if either is set.
unset VIRTUAL_ENV CONDA_PREFIX

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
[[ -x "$UV_BIN" ]] || UV_BIN="$(command -v uv || true)"
if [[ -z "${UV_BIN:-}" ]]; then
    echo "run_win.sh: uv not found (looked in ~/.local/bin and \$PATH)" >&2
    exit 1
fi

echo "→ uv sync"
"$UV_BIN" sync

echo "→ launching: calimerge ${args[*]:-}"
exec "$UV_BIN" run calimerge "${args[@]}"
