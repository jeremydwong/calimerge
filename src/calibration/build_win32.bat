@echo off
:: build_win32.bat - Unity build for cm_calibration library (Windows)
::
:: Usage:
::   build_win32.bat           (release build, default)
::   build_win32.bat debug     (debug build with /Zi)
::
:: Output:
::   build\calibration\cm_calibration.dll
::   build\calibration\cm_calibration.lib
::   build\calibration\test_calibration.exe
::
:: Requirements:
::   MSVC Build Tools 2022 at:
::     C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\
::   OpenCV at OPENCV_PATH (default: C:\OpenCV\opencv\build)

echo Build script starting...
:: vswhere.exe is not on PATH by default; prepend its dir to suppress the
:: harmless "not recognized" warning VsDevCmd.bat emits when it can't find it.
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64

:: Resolve repo root (two levels up from src/calibration/)
set SCRIPT_DIR=%~dp0
:: Resolve BUILD_DIR to a fully absolute path (removes ..\ sequences) so the
:: linker never receives an unresolved relative path as an argument.
for %%I in ("%SCRIPT_DIR%..\..\build\calibration") do set BUILD_DIR=%%~fI

:: OpenCV — default to C:\OpenCV\opencv\build, allow override via environment
if "%OPENCV_PATH%"=="" set OPENCV_PATH=C:\OpenCV\opencv\build

set OPENCV_INCLUDE=%OPENCV_PATH%\include
set OPENCV_LIB_DIR=%OPENCV_PATH%\x64\vc16\lib

:: Find the OpenCV world lib (version-agnostic glob)
:: cl.exe doesn't support wildcards, so we detect the lib name with a for loop.
set OPENCV_LIB=
for %%F in (%OPENCV_LIB_DIR%\opencv_world*.lib) do (
    :: skip debug libs (ending in 'd.lib')
    echo %%F | findstr /i /c:"d.lib" >nul 2>&1
    if errorlevel 1 set OPENCV_LIB=%%F
)
if "%OPENCV_LIB%"=="" (
    echo ERROR: Could not find opencv_world*.lib under %OPENCV_LIB_DIR%
    echo        Set OPENCV_PATH to your OpenCV build directory.
    exit /b 1
)
echo Using OpenCV lib: %OPENCV_LIB%

:: Create output directory
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

pushd %SCRIPT_DIR%

if "%1"=="debug" (
    set CFLAGS=/W4 /WX /wd4100 /wd4201 /wd4189 /Zi /Od /DDEBUG /D_CRT_SECURE_NO_WARNINGS
) else (
    set CFLAGS=/W4 /WX /wd4100 /wd4201 /wd4189 /O2 /DNDEBUG /D_CRT_SECURE_NO_WARNINGS
)

:: ============================================================
:: DLL — explicit compile then link (avoids /LD implib placement ambiguity)
:: ============================================================
cl %CFLAGS% /EHsc /std:c++17 /c ^
    /DCM_CALIBRATION_BUILDING_DLL ^
    /I"%OPENCV_INCLUDE%" ^
    /Fo:"%BUILD_DIR%\calibration_unity.obj" ^
    calibration_unity.cpp

if errorlevel 1 (
    echo BUILD FAILED: compile
    popd
    exit /b 1
)

link /DLL /NOLOGO ^
    /IMPLIB:"%BUILD_DIR%\cm_calibration.lib" ^
    /OUT:"%BUILD_DIR%\cm_calibration.dll" ^
    "%BUILD_DIR%\calibration_unity.obj" ^
    "%OPENCV_LIB%"

if errorlevel 1 (
    echo BUILD FAILED: DLL link
    popd
    exit /b 1
)

echo.
dumpbin /exports "%BUILD_DIR%\cm_calibration.dll" 2>nul | findstr "cm_"

:: ============================================================
:: Test binary
:: ============================================================
cl %CFLAGS% /EHsc /std:c++17 ^
    /I"%OPENCV_INCLUDE%" ^
    /Fe:"%BUILD_DIR%\test_calibration.exe" ^
    test_calibration.cpp ^
    "%OPENCV_LIB%" ^
    /link "%BUILD_DIR%\cm_calibration.lib"

if errorlevel 1 (
    echo BUILD FAILED: test_calibration.exe
    popd
    exit /b 1
)

:: Clean up .obj files left in source dir by cl
del /q *.obj 2>nul

popd

echo.
echo Build output: %BUILD_DIR%
echo   cm_calibration.dll
echo   cm_calibration.lib
echo   test_calibration.exe
echo.
echo To run tests (ensure OpenCV DLLs are on PATH):
echo   set PATH=%OPENCV_PATH%\x64\vc16\bin;%%PATH%%
echo   %BUILD_DIR%\test_calibration.exe
