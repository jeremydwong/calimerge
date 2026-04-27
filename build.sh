#!/usr/bin/env bash
# build.sh - Cross-platform native library build dispatcher.
#
# Detects the host OS and runs the matching native build script in src/native/.
# Any arguments are forwarded to the platform script (e.g. `bash build.sh release`).
#
# Usage:
#   bash build.sh           # release build (default)
#   bash build.sh release   # explicit release
#   bash build.sh debug     # debug build
set -euo pipefail
cd "$(dirname "$0")"

# Default flavor matches the platform scripts: both treat "release" as default
# when no arg is given, so it's safe to forward whatever the caller passed
# (including nothing).
case "$(uname -s)" in
    Darwin)
        cd src/native && exec ./build_macos.sh "$@"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        # build_win32.bat must be invoked via `./` from Git Bash. A bare
        # `cmd //c build_win32.bat ...` fails because cmd /c searches PATH,
        # not the cwd, for the script name.
        cd src/native && exec ./build_win32.bat "$@"
        ;;
    Linux)
        echo "build.sh: Linux native build is not implemented yet." >&2
        exit 1
        ;;
    *)
        echo "build.sh: unsupported platform: $(uname -s)" >&2
        exit 1
        ;;
esac
