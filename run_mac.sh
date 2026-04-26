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

needs_build=0
if [[ "$SKIP_BUILD" == "1" ]]; then
    needs_build=0
elif [[ "$FORCE_REBUILD" == "1" ]]; then
    needs_build=1
elif [[ ! -f "$NATIVE_LIB" ]]; then
    echo "→ native library missing"
    needs_build=1
elif [[ -n "$(find src/native -type f \( -name '*.mm' -o -name '*.cpp' -o -name '*.h' \) -newer "$NATIVE_LIB" -print -quit 2>/dev/null)" ]]; then
    echo "→ native source newer than $NATIVE_LIB"
    needs_build=1
fi

if [[ "$needs_build" == "1" ]]; then
    echo "→ building native library…"
    (cd src/native && ./build_macos.sh release)
else
    echo "→ native library up to date"
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
