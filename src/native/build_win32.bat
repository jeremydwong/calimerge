@echo off
::
:: build_win32.bat - Unity build for calimerge camera module (Windows)
::
:: Usage: build_win32.bat [debug|release]
::
:: Requires: Visual Studio Developer Command Prompt (or vcvarsall.bat)
::

setlocal

cd /d "%~dp0"

set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=release

echo Building calimerge for Windows (%BUILD_TYPE%)...

if "%BUILD_TYPE%"=="debug" (
    set CFLAGS=/Zi /Od /DDEBUG /W3
) else (
    set CFLAGS=/O2 /DNDEBUG /W3
)

:: Check that cl.exe is available
where cl >nul 2>&1
if errorlevel 1 (
    echo Error: cl.exe not found. Run from a Visual Studio Developer Command Prompt.
    echo   Or run: vcvarsall.bat x64
    exit /b 1
)

:: Build shared library (DLL)
cl %CFLAGS% /EHsc /LD /Fe:calimerge.dll calimerge_win32.cpp ^
    mfplat.lib mfreadwrite.lib mf.lib mfuuid.lib ole32.lib ^
    /link /DLL

if errorlevel 1 (
    echo Build FAILED.
    exit /b 1
)

echo.
echo Built: %~dp0calimerge.dll

:: Show exported symbols
echo.
echo Exported symbols:
dumpbin /exports calimerge.dll 2>nul | findstr "cm_"

:: Build test executables
echo.
echo Building test executables...

cl %CFLAGS% /Fe:test_enumerate.exe test_enumerate.c /link calimerge.lib
cl %CFLAGS% /Fe:test_capture.exe test_capture.c /link calimerge.lib
cl %CFLAGS% /Fe:test_multi.exe test_multi.c /link calimerge.lib
cl %CFLAGS% /Fe:test_sync_log.exe test_sync_log.c /link calimerge.lib

echo.
echo Done.

endlocal
