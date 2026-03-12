#!/bin/bash
# Rebuild the native DLL, handling locked file if needed
cd "$(dirname "$0")"
if [ -f calimerge.dll ]; then
    mv calimerge.dll calimerge_old.dll 2>/dev/null || true
fi
cmd //c "build_win32.bat release"
