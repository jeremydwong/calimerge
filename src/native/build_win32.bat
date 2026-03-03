@echo off
:: build_win32.bat - Unity build for calimerge camera module (Windows)
::
:: Usage: build_win32.bat [debug]

echo Build script starting...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64

pushd %~dp0

if "%1"=="debug" (
    set CFLAGS=-W4 -WX -wd4100 -wd4201 -wd4189 /Zi /Od /DDEBUG
) else (
    set CFLAGS=-W4 -WX -wd4100 -wd4201 -wd4189 /O2 /DNDEBUG
)

:: DLL (unity build - single translation unit)
cl %CFLAGS% /EHsc /LD /Fe:calimerge.dll calimerge_win32.cpp ^
    mfplat.lib mfreadwrite.lib mf.lib mfuuid.lib ole32.lib strmiids.lib ^
    /link /DLL /DEF:calimerge.def

if errorlevel 1 (
    echo BUILD FAILED
    popd
    exit /b 1
)

echo.
dumpbin /exports calimerge.dll 2>nul | findstr "cm_"

:: Test programs
cl %CFLAGS% /Fe:test_enumerate.exe test_enumerate.c /link calimerge.lib
cl %CFLAGS% /Fe:test_capture.exe test_capture.c /link calimerge.lib
cl %CFLAGS% /Fe:test_multi.exe test_multi.c /link calimerge.lib
cl %CFLAGS% /Fe:test_sync_log.exe test_sync_log.c /link calimerge.lib

popd
