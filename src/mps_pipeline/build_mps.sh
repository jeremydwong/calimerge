#!/bin/bash
# build_mps.sh - Build the MPS / CoreML pose tracking pipeline dylib (macOS).
#
# Usage:  bash src/mps_pipeline/build_mps.sh [release]
# Output: build/mps/libcalimerge_mps.dylib  (relative to repo root)
#
# Mirrors src/cuda_pipeline/build_cuda_win32.bat's release/debug split and
# output layout, so the Python binding's lib-search logic
# (build/<backend>/lib...) finds both backends with the same code path.
#
# Requires: macOS 14+ on Apple Silicon (M1+).
#
# Frameworks: CoreML, Metal, AVFoundation, Accelerate.

set -e

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "build_mps.sh: macOS only — current uname is $(uname -s)"
    exit 1
fi

echo "============================================================"
echo " MPS Pose Tracking Pipeline - macOS Build"
echo "============================================================"
echo

# ---- Resolve paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_DIR="$REPO_ROOT/src/pt_shared"
OUT_DIR="$REPO_ROOT/build/mps"

mkdir -p "$OUT_DIR"

# ---- Build mode ----
if [[ "$1" == "release" ]]; then
    CFLAGS_OPT="-O2 -DNDEBUG"
    echo "Build mode: RELEASE"
else
    CFLAGS_OPT="-O0 -g -DDEBUG"
    echo "Build mode: DEBUG"
fi

COMMON_FLAGS="-std=c17 -Wall -Wextra -Wno-unused-parameter"
OBJC_FLAGS="-std=gnu17 -fobjc-arc -Wall -Wextra -Wno-unused-parameter"
CXX_FLAGS="-std=c++17 -Wall -Wextra -Wno-unused-parameter"

FRAMEWORKS=(
    -framework Foundation
    -framework CoreML
    -framework Metal
    -framework MetalPerformanceShaders
    -framework Accelerate
    -framework AVFoundation
    -framework CoreVideo
    -framework CoreMedia
)

cd "$SCRIPT_DIR"

echo
echo "---- Compiling shared C++ sources ----"

clang++ -c $CFLAGS_OPT $CXX_FLAGS -I"$SHARED_DIR" \
    -o "$OUT_DIR/pt_matching.o" "$SHARED_DIR/pt_matching.cpp"
echo "  pt_matching.o"

clang++ -c $CFLAGS_OPT $CXX_FLAGS -I"$SHARED_DIR" \
    -o "$OUT_DIR/pt_triangulation.o" "$SHARED_DIR/pt_triangulation.cpp"
echo "  pt_triangulation.o"

clang++ -c $CFLAGS_OPT $CXX_FLAGS -I"$SHARED_DIR" \
    -o "$OUT_DIR/pt_tracker.o" "$SHARED_DIR/pt_tracker.cpp"
echo "  pt_tracker.o"

clang++ -c $CFLAGS_OPT $CXX_FLAGS -I"$SHARED_DIR" \
    -o "$OUT_DIR/pt_export.o" "$SHARED_DIR/pt_export.cpp"
echo "  pt_export.o"

echo
echo "---- Compiling MPS pipeline sources ----"

# Plain C
clang -c $CFLAGS_OPT $COMMON_FLAGS -I"$SHARED_DIR" \
    -o "$OUT_DIR/pt_calibration.o" pt_calibration.c
echo "  pt_calibration.o"

clang -c $CFLAGS_OPT $COMMON_FLAGS -I"$SHARED_DIR" \
    -o "$OUT_DIR/pt_heatmap.o" pt_heatmap.c
echo "  pt_heatmap.o"

# Objective-C / Objective-C++
clang -c $CFLAGS_OPT $OBJC_FLAGS -I"$SHARED_DIR" \
    -o "$OUT_DIR/pt_coreml.o" pt_coreml.m
echo "  pt_coreml.o"

clang -c $CFLAGS_OPT $OBJC_FLAGS -I"$SHARED_DIR" \
    -o "$OUT_DIR/pt_preprocess.o" pt_preprocess.m
echo "  pt_preprocess.o"

clang -c $CFLAGS_OPT $OBJC_FLAGS -I"$SHARED_DIR" \
    -o "$OUT_DIR/pt_videodecode.o" pt_videodecode.m
echo "  pt_videodecode.o"

clang -c $CFLAGS_OPT $OBJC_FLAGS -I"$SHARED_DIR" \
    -o "$OUT_DIR/pt_stream_mps.o" pt_stream_mps.m
echo "  pt_stream_mps.o"

clang -c $CFLAGS_OPT $OBJC_FLAGS -I"$SHARED_DIR" \
    -o "$OUT_DIR/pt_offline_mps.o" pt_offline_mps.m
echo "  pt_offline_mps.o"

echo
echo "---- Linking dylib ----"

clang++ -dynamiclib -arch arm64 \
    -exported_symbols_list calimerge_mps.def \
    -install_name @rpath/libcalimerge_mps.dylib \
    -o "$OUT_DIR/libcalimerge_mps.dylib" \
    "$OUT_DIR"/pt_matching.o "$OUT_DIR"/pt_triangulation.o "$OUT_DIR"/pt_tracker.o "$OUT_DIR"/pt_export.o \
    "$OUT_DIR"/pt_calibration.o "$OUT_DIR"/pt_heatmap.o \
    "$OUT_DIR"/pt_coreml.o "$OUT_DIR"/pt_preprocess.o "$OUT_DIR"/pt_videodecode.o \
    "$OUT_DIR"/pt_stream_mps.o "$OUT_DIR"/pt_offline_mps.o \
    "${FRAMEWORKS[@]}" \
    -lc++

echo
echo "---- Exported symbols ----"
nm -gU "$OUT_DIR/libcalimerge_mps.dylib" | grep "_pt_" || true

echo
echo "---- Building streaming test (pt_stream_main_mps) ----"

if clang $CFLAGS_OPT $OBJC_FLAGS -I"$SHARED_DIR" \
    -o "$OUT_DIR/pt_stream_main_mps" pt_stream_main_mps.m \
    "$OUT_DIR"/pt_matching.o "$OUT_DIR"/pt_triangulation.o "$OUT_DIR"/pt_tracker.o "$OUT_DIR"/pt_export.o \
    "$OUT_DIR"/pt_calibration.o "$OUT_DIR"/pt_heatmap.o \
    "$OUT_DIR"/pt_coreml.o "$OUT_DIR"/pt_preprocess.o "$OUT_DIR"/pt_videodecode.o "$OUT_DIR"/pt_stream_mps.o \
    "${FRAMEWORKS[@]}" \
    -lc++ 2>/dev/null
then
    echo "  pt_stream_main_mps built"
else
    echo "  WARNING: pt_stream_main_mps build failed (non-fatal)"
fi

echo
echo "---- Building offline test (pt_main_mps) ----"

if clang $CFLAGS_OPT $OBJC_FLAGS -I"$SHARED_DIR" \
    -o "$OUT_DIR/pt_main_mps" pt_main_mps.m \
    "$OUT_DIR"/pt_matching.o "$OUT_DIR"/pt_triangulation.o "$OUT_DIR"/pt_tracker.o "$OUT_DIR"/pt_export.o \
    "$OUT_DIR"/pt_calibration.o "$OUT_DIR"/pt_heatmap.o \
    "$OUT_DIR"/pt_coreml.o "$OUT_DIR"/pt_preprocess.o "$OUT_DIR"/pt_videodecode.o \
    "$OUT_DIR"/pt_stream_mps.o "$OUT_DIR"/pt_offline_mps.o \
    "${FRAMEWORKS[@]}" \
    -lc++ 2>/dev/null
then
    echo "  pt_main_mps built"
else
    echo "  WARNING: pt_main_mps build failed (non-fatal)"
fi

echo
echo "============================================================"
echo " Build complete:"
echo "   $OUT_DIR/libcalimerge_mps.dylib"
echo "   $OUT_DIR/pt_stream_main_mps   (streaming smoke test)"
echo "   $OUT_DIR/pt_main_mps          (offline batch smoke test)"
echo "============================================================"
