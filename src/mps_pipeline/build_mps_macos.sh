#!/bin/bash
# build_mps_macos.sh - Build the MPS pose tracking pipeline dylib (macOS)
#
# Usage: ./build_mps_macos.sh [release]
#
# Requires: macOS 15+ with Apple Silicon (M1+)

set -e

echo "============================================================"
echo " MPS Pose Tracking Pipeline - macOS Build"
echo "============================================================"
echo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SHARED_DIR="../pt_shared"

# Build mode
if [ "$1" = "release" ]; then
    CFLAGS="-O2 -DNDEBUG"
    echo "Build mode: RELEASE"
else
    CFLAGS="-O0 -g -DDEBUG"
    echo "Build mode: DEBUG"
fi

COMMON_FLAGS="-std=c17 -Wall -Wextra -Wno-unused-parameter"
OBJC_FLAGS="-std=gnu17 -fobjc-arc -Wall -Wextra -Wno-unused-parameter"
CXX_FLAGS="-std=c++17 -Wall -Wextra -Wno-unused-parameter"

FRAMEWORKS="-framework Foundation -framework CoreML -framework Accelerate -framework AVFoundation -framework CoreVideo -framework CoreMedia"

echo
echo "---- Compiling shared C++ sources ----"

# Shared platform-independent code (C++)
clang++ -c $CFLAGS $CXX_FLAGS -I"$SHARED_DIR" \
    -o pt_matching.o "$SHARED_DIR/pt_matching.cpp"
echo "  pt_matching.o"

clang++ -c $CFLAGS $CXX_FLAGS -I"$SHARED_DIR" \
    -o pt_triangulation.o "$SHARED_DIR/pt_triangulation.cpp"
echo "  pt_triangulation.o"

clang++ -c $CFLAGS $CXX_FLAGS -I"$SHARED_DIR" \
    -o pt_tracker.o "$SHARED_DIR/pt_tracker.cpp"
echo "  pt_tracker.o"

clang++ -c $CFLAGS $CXX_FLAGS -I"$SHARED_DIR" \
    -o pt_export.o "$SHARED_DIR/pt_export.cpp"
echo "  pt_export.o"

echo
echo "---- Compiling MPS pipeline sources ----"

# Plain C sources
clang -c $CFLAGS $COMMON_FLAGS -I"$SHARED_DIR" \
    -o pt_calibration.o pt_calibration.c
echo "  pt_calibration.o"

clang -c $CFLAGS $COMMON_FLAGS -I"$SHARED_DIR" \
    -o pt_heatmap.o pt_heatmap.c
echo "  pt_heatmap.o"

# Objective-C sources
clang -c $CFLAGS $OBJC_FLAGS -I"$SHARED_DIR" \
    -o pt_coreml.o pt_coreml.m
echo "  pt_coreml.o"

clang -c $CFLAGS $OBJC_FLAGS -I"$SHARED_DIR" \
    -o pt_preprocess.o pt_preprocess.m
echo "  pt_preprocess.o"

clang -c $CFLAGS $OBJC_FLAGS -I"$SHARED_DIR" \
    -o pt_videodecode.o pt_videodecode.m
echo "  pt_videodecode.o"

clang -c $CFLAGS $OBJC_FLAGS -I"$SHARED_DIR" \
    -o pt_stream_mps.o pt_stream_mps.m
echo "  pt_stream_mps.o"

echo
echo "---- Linking dylib ----"

# Export only the API functions
clang++ -dynamiclib -arch arm64 \
    -exported_symbols_list calimerge_mps.def \
    -install_name @rpath/calimerge_mps.dylib \
    -o calimerge_mps.dylib \
    pt_matching.o pt_triangulation.o pt_tracker.o pt_export.o \
    pt_calibration.o pt_heatmap.o \
    pt_coreml.o pt_preprocess.o pt_videodecode.o pt_stream_mps.o \
    $FRAMEWORKS \
    -lc++

if [ $? -ne 0 ]; then
    echo "LINK FAILED"
    exit 1
fi

echo
echo "---- Exported symbols ----"
nm -gU calimerge_mps.dylib | grep "_pt_"

echo
echo "---- Building streaming test (pt_stream_main_mps) ----"

clang $CFLAGS $OBJC_FLAGS -I"$SHARED_DIR" \
    -o pt_stream_main_mps pt_stream_main_mps.m \
    pt_matching.o pt_triangulation.o pt_tracker.o pt_export.o \
    pt_calibration.o pt_heatmap.o \
    pt_coreml.o pt_preprocess.o pt_videodecode.o pt_stream_mps.o \
    $FRAMEWORKS \
    -lc++ 2>/dev/null

if [ $? -ne 0 ]; then
    echo "  WARNING: pt_stream_main_mps build failed (non-fatal, test harness may not exist yet)"
else
    echo "  pt_stream_main_mps built successfully"
fi

echo
echo "============================================================"
echo " Build complete: calimerge_mps.dylib"
echo "============================================================"
