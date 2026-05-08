#!/usr/bin/env bash
# build_macos.sh - Unity build for the Calimerge Qt6 GUI (macOS / clang++)
#
# Usage:
#   bash build_macos.sh          (debug, default)
#   bash build_macos.sh release  (optimised, NDEBUG)
#
# Prerequisites:
#   - Xcode Command Line Tools (clang++)
#   - Qt 6.x installed via Homebrew or the Qt online installer.
#     Either:  export QT_DIR=/path/to/Qt/6.x.x/macos  before running, or
#              install via: brew install qt@6
#     Default search paths tried in order:
#       $QT_DIR  (if set)
#       /usr/local/opt/qt@6      (Homebrew on Intel)
#       /opt/homebrew/opt/qt@6   (Homebrew on Apple Silicon)
#       ~/Qt/6.*/macos           (Qt online installer)
#     If Qt is not found the script prints an install URL and exits with 1.
#
# Output: ../../build/app/calimerge
#
# Style: mirrors build_win32.bat — no CMake, direct compiler invocation.
# See design_cpp.md §5.

set -e

echo "============================================================"
echo " Calimerge Qt6 GUI - macOS Build"
echo "============================================================"
echo

# ---- Locate Qt6 -----------------------------------------------------------
find_qt() {
    local candidates=(
        "${QT_DIR:-}"
        "/usr/local/opt/qt@6"
        "/opt/homebrew/opt/qt@6"
    )
    # Also try ~/Qt/6.*/macos from the Qt online installer
    for d in ~/Qt/6.*/macos; do
        candidates+=("$d")
    done

    for dir in "${candidates[@]}"; do
        if [[ -n "$dir" && -f "$dir/include/QtCore/qobject.h" ]]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

QT_FOUND=$(find_qt) || true
if [[ -z "$QT_FOUND" ]]; then
    echo
    echo "ERROR: Qt6 not found."
    echo
    echo " Fix options:"
    echo "  1. Install via Homebrew:  brew install qt@6"
    echo "  2. Download from: https://www.qt.io/download-open-source"
    echo "     Select 'macOS' component in the installer."
    echo "  3. Set QT_DIR to the correct path before running:"
    echo "       export QT_DIR=/path/to/Qt/6.x.x/macos"
    echo "       bash src/app/build_macos.sh release"
    echo
    exit 1
fi

echo "Qt6 found at: $QT_FOUND"
QT_DIR="$QT_FOUND"
MOCBIN="$QT_DIR/bin/moc"
RCCBIN="$QT_DIR/bin/rcc"

# ---- Build mode -----------------------------------------------------------
BUILD_MODE="${1:-debug}"
if [[ "$BUILD_MODE" == "release" ]]; then
    CFLAGS="-O2 -DNDEBUG"
    echo "Build mode: RELEASE"
else
    CFLAGS="-O0 -g -DDEBUG"
    echo "Build mode: DEBUG"
fi

# ---- Paths ----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."
BUILD_DIR="$REPO_ROOT/build/app"
GEN_DIR="$SCRIPT_DIR/gen"

mkdir -p "$BUILD_DIR"
mkdir -p "$GEN_DIR"

cd "$SCRIPT_DIR"

# ---- Step 1: MOC ----------------------------------------------------------
echo
echo "---- Running moc on Q_OBJECT headers ----"

run_moc() {
    local header="$1"
    local basename
    basename="$(basename "$header" .h)"
    echo "  moc $header"
    "$MOCBIN" "$header" -o "$GEN_DIR/moc_${basename}.cpp"
}

# Headers in src/app/ root
for h in *.h; do
    [[ -f "$h" ]] && run_moc "$h"
done

# Sub-directories — skip gracefully if not present yet
for subdir in tabs widgets workers; do
    if [[ -d "$subdir" ]]; then
        for h in "$subdir"/*.h; do
            [[ -f "$h" ]] && run_moc "$h"
        done
    fi
done

# ---- Step 2: RCC ----------------------------------------------------------
if [[ -f resources.qrc ]]; then
    echo
    echo "---- Running rcc on resources.qrc ----"
    "$RCCBIN" resources.qrc -o "$GEN_DIR/resources.cpp"
fi

# ---- Step 3: Unity compile ------------------------------------------------
echo
echo "---- Compiling app_unity.cpp ----"

QT_INCLUDES=(
    "-I$QT_DIR/include"
    "-I$QT_DIR/include/QtCore"
    "-I$QT_DIR/include/QtWidgets"
    "-I$QT_DIR/include/QtGui"
)

QT_FRAMEWORKS=(
    "-F$QT_DIR/lib"
    "-framework QtCore"
    "-framework QtWidgets"
    "-framework QtGui"
)

# Qt on macOS ships as frameworks; add rpath so the binary can find them
QT_RPATH="-Wl,-rpath,$QT_DIR/lib"

clang++ \
    -std=c++17 \
    $CFLAGS \
    "${QT_INCLUDES[@]}" \
    "-I." \
    -o "$BUILD_DIR/calimerge" \
    app_unity.cpp \
    "${QT_FRAMEWORKS[@]}" \
    $QT_RPATH

echo
echo "============================================================"
echo " Build OK: $BUILD_DIR/calimerge"
echo "============================================================"
