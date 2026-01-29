#!/bin/bash
#
# build_macos.sh - Unity build for calimerge camera module (macOS)
#
# Usage: ./build_macos.sh [debug|release]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_TYPE="${1:-release}"

echo "Building calimerge for macOS ($BUILD_TYPE)..."

if [ "$BUILD_TYPE" = "debug" ]; then
    CFLAGS="-g -O0 -DDEBUG"
else
    CFLAGS="-O2 -DNDEBUG"
fi

clang++ $CFLAGS -std=c++17 \
    -fobjc-arc \
    -framework AVFoundation \
    -framework CoreMedia \
    -framework CoreVideo \
    -framework Foundation \
    -shared -fPIC \
    -o libcalimerge.dylib \
    calimerge_macos.mm

echo "Built: $SCRIPT_DIR/libcalimerge.dylib"

# Show exported symbols
echo ""
echo "Exported symbols:"
nm -gU libcalimerge.dylib | grep " T " | head -20
