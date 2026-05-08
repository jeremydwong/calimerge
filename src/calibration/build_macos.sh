#!/usr/bin/env bash
# build_macos.sh - Unity build for cm_calibration library (macOS)
#
# Usage:
#   bash build_macos.sh           (release build, default)
#   bash build_macos.sh debug     (debug build with -g -O0)
#
# Output:
#   build/calibration/libcm_calibration.dylib
#   build/calibration/test_calibration
#
# Requirements:
#   clang++ (Xcode Command Line Tools)
#   OpenCV via Homebrew (brew install opencv)
#     Apple Silicon: /opt/homebrew/opt/opencv
#     Intel:         /usr/local/opt/opencv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/calibration"

mkdir -p "${BUILD_DIR}"

# Detect OpenCV prefix — try Apple Silicon path first, then Intel
if [ -d /opt/homebrew/opt/opencv ]; then
    OPENCV_PREFIX=/opt/homebrew/opt/opencv
elif [ -d /usr/local/opt/opencv ]; then
    OPENCV_PREFIX=/usr/local/opt/opencv
else
    echo "ERROR: OpenCV not found. Install with: brew install opencv"
    exit 1
fi

OPENCV_INCLUDE="${OPENCV_PREFIX}/include/opencv4"
OPENCV_LIB_DIR="${OPENCV_PREFIX}/lib"

# Use pkg-config if available; otherwise fall back to hard-coded flags
if pkg-config --exists opencv4 2>/dev/null; then
    OPENCV_CFLAGS=$(pkg-config --cflags opencv4)
    OPENCV_LIBS=$(pkg-config --libs opencv4)
else
    OPENCV_CFLAGS="-I${OPENCV_INCLUDE}"
    OPENCV_LIBS="-L${OPENCV_LIB_DIR} -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_aruco -lopencv_objdetect"
fi

if [ "${1:-release}" = "debug" ]; then
    CFLAGS="-Wall -Wextra -Wno-unused-parameter -g -O0 -DDEBUG"
else
    CFLAGS="-Wall -Wextra -Wno-unused-parameter -O2 -DNDEBUG"
fi

STDFLAGS="-std=c++17"

echo "OpenCV prefix: ${OPENCV_PREFIX}"
echo "Building..."

# ============================================================
# Shared library (unity build)
# ============================================================
clang++ ${CFLAGS} ${STDFLAGS} \
    ${OPENCV_CFLAGS} \
    -dynamiclib \
    -o "${BUILD_DIR}/libcm_calibration.dylib" \
    "${SCRIPT_DIR}/calibration_unity.cpp" \
    ${OPENCV_LIBS}

echo "Built: ${BUILD_DIR}/libcm_calibration.dylib"

# ============================================================
# Test binary
# ============================================================
clang++ ${CFLAGS} ${STDFLAGS} \
    ${OPENCV_CFLAGS} \
    -o "${BUILD_DIR}/test_calibration" \
    "${SCRIPT_DIR}/test_calibration.cpp" \
    ${OPENCV_LIBS} \
    -L"${BUILD_DIR}" \
    -lcm_calibration \
    -Wl,-rpath,"${BUILD_DIR}"

echo "Built: ${BUILD_DIR}/test_calibration"

echo ""
echo "Build output: ${BUILD_DIR}"
echo "  libcm_calibration.dylib"
echo "  test_calibration"
echo ""
echo "To run tests:"
echo "  ${BUILD_DIR}/test_calibration"
