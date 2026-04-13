@echo off
:: build_win32.bat - Unity build for calimerge camera module (Windows)
::
:: Usage: build_win32.bat [debug]
:: Output: build\native\ (relative to repo root)

echo Build script starting...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64

:: Resolve repo root (two levels up from src/native/)
set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..\..\
set BUILD_DIR=%REPO_ROOT%build\native

:: Create output directory
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

pushd %SCRIPT_DIR%

if "%1"=="debug" (
    set CFLAGS=-W4 -WX -wd4100 -wd4201 -wd4189 /Zi /Od /DDEBUG
) else (
    set CFLAGS=-W4 -WX -wd4100 -wd4201 -wd4189 /O2 /DNDEBUG
)

:: DLL (unity build - single translation unit)
cl %CFLAGS% /EHsc /LD /Fe:"%BUILD_DIR%\calimerge.dll" calimerge_win32.cpp ^
    mfplat.lib mfreadwrite.lib mf.lib mfuuid.lib ole32.lib strmiids.lib ^
    /link /DLL /DEF:calimerge.def /IMPLIB:"%BUILD_DIR%\calimerge.lib"

if errorlevel 1 (
    echo BUILD FAILED
    popd
    exit /b 1
)

echo.
dumpbin /exports "%BUILD_DIR%\calimerge.dll" 2>nul | findstr "cm_"

:: Test programs (linked against calimerge.dll)
cl %CFLAGS% /Fe:"%BUILD_DIR%\test_enumerate.exe" test_enumerate.c /link "%BUILD_DIR%\calimerge.lib"
cl %CFLAGS% /Fe:"%BUILD_DIR%\test_capture.exe" test_capture.c /link "%BUILD_DIR%\calimerge.lib"
cl %CFLAGS% /Fe:"%BUILD_DIR%\test_multi.exe" test_multi.c /link "%BUILD_DIR%\calimerge.lib"
cl %CFLAGS% /Fe:"%BUILD_DIR%\test_sync_log.exe" test_sync_log.c /link "%BUILD_DIR%\calimerge.lib"

:: Standalone diagnostic tools (no dependency on calimerge.dll)
cl %CFLAGS% /EHsc /Fe:"%BUILD_DIR%\test_usb_serials.exe" test_usb_serials.cpp ^
    /link mf.lib mfplat.lib mfuuid.lib ole32.lib setupapi.lib

:: Clean up .obj files left in source dir by cl
del /q *.obj 2>nul

popd

echo.
echo Build output: %BUILD_DIR%
