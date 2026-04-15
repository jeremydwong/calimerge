#!/bin/bash
#
# build_macos.sh - Unity build for calimerge camera module (macOS)
#
# Usage: ./build_macos.sh [debug|release]
# Output: build/native/ (relative to repo root)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/native"

mkdir -p "$BUILD_DIR"

BUILD_TYPE="${1:-release}"

echo "Building calimerge for macOS ($BUILD_TYPE)..."
echo "Output: $BUILD_DIR"

if [ "$BUILD_TYPE" = "debug" ]; then
    CFLAGS="-g -O0 -DDEBUG"
else
    CFLAGS="-O2 -DNDEBUG"
fi

cd "$SCRIPT_DIR"

clang++ $CFLAGS -std=c++17 \
    -fobjc-arc \
    -framework AVFoundation \
    -framework CoreMedia \
    -framework CoreVideo \
    -framework Foundation \
    -framework IOKit \
    -shared -fPIC \
    -o "$BUILD_DIR/libcalimerge.dylib" \
    calimerge_macos.mm

echo "Built: $BUILD_DIR/libcalimerge.dylib"

# Show exported symbols
echo ""
echo "Exported symbols:"
nm -gU "$BUILD_DIR/libcalimerge.dylib" | grep " T " | head -20
