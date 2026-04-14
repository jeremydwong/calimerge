@echo off
:: build_cuda_win32.bat - Build the CUDA pose tracking pipeline DLL (Windows)
::
:: Usage: build_cuda_win32.bat [release]
:: Output: build\cuda\ (relative to repo root)
::
:: Requires:
::   - MSVC Build Tools 2022
::   - CUDA Toolkit 12.9
::   - TensorRT (set TENSORRT_PATH env var)
::   - OpenCV (optional, set OPENCV_PATH for CPU fallback decode)

echo ============================================================
echo  CUDA Pose Tracking Pipeline - Windows Build
echo ============================================================
echo.

setlocal

:: ---- MSVC ----
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64

:: ---- CUDA 12.9 ----
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set PATH=%CUDA_PATH%\bin;%PATH%

:: Verify nvcc
where nvcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc not found. Is CUDA Toolkit 12.9 installed?
    exit /b 1
)

:: ---- TensorRT ----
if not defined TENSORRT_PATH (
    echo WARNING: TENSORRT_PATH not set. TensorRT features will be stubbed out.
    set TENSORRT_PATH=C:\TensorRT
)

:: ---- OpenCV (optional) ----
if not defined OPENCV_PATH (
    echo WARNING: OPENCV_PATH not set. OpenCV CPU fallback will be disabled.
)

:: ---- Build mode ----
if "%1"=="release" (
    set OPTIMIZE=/O2
    set NVCC_OPT=-O2
    set CFLAGS=/O2 /MD /EHsc /DNDEBUG
    echo Build mode: RELEASE
) else (
    set OPTIMIZE=/Od /Zi
    set NVCC_OPT=-O0 -g -G
    set CFLAGS=/Od /Zi /MD /EHsc /DDEBUG
    echo Build mode: DEBUG
)

:: ---- Resolve paths ----
set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..\..\
set BUILD_DIR=%REPO_ROOT%build\cuda
set SRC_DIR=%SCRIPT_DIR%

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

pushd %SRC_DIR%

echo.
echo ---- Compiling CUDA sources (.cu) ----

:: sm_120 = Blackwell (RTX 5070 Ti)
nvcc -c %NVCC_OPT% --use_fast_math ^
    -gencode arch=compute_120,code=sm_120 ^
    -Xcompiler "/MD %OPTIMIZE% /EHsc" ^
    -I"%CUDA_PATH%\include" ^
    -I"%TENSORRT_PATH%\include" ^
    -I"..\pt_shared" ^
    -o "%BUILD_DIR%\pt_arena.obj" pt_arena.cu

if errorlevel 1 (
    echo BUILD FAILED: pt_arena.cu
    popd
    exit /b 1
)

nvcc -c %NVCC_OPT% --use_fast_math ^
    -gencode arch=compute_120,code=sm_120 ^
    -Xcompiler "/MD %OPTIMIZE% /EHsc" ^
    -I"%CUDA_PATH%\include" ^
    -I"%TENSORRT_PATH%\include" ^
    -I"..\pt_shared" ^
    -o "%BUILD_DIR%\pt_kernels.obj" pt_kernels.cu

if errorlevel 1 (
    echo BUILD FAILED: pt_kernels.cu
    popd
    exit /b 1
)

echo.
echo ---- Compiling C++ sources (.cpp) ----

:: Build cl flags with optional OpenCV include
set CL_INCLUDES=/I"%CUDA_PATH%\include" /I"%TENSORRT_PATH%\include" /I"..\pt_shared"
if defined OPENCV_PATH (
    set CL_INCLUDES=%CL_INCLUDES% /I"%OPENCV_PATH%\include" /DHAS_OPENCV
)

cl /c %CFLAGS% %CL_INCLUDES% /Fo:"%BUILD_DIR%\pt_tensorrt.obj" pt_tensorrt.cpp
if errorlevel 1 ( echo BUILD FAILED: pt_tensorrt.cpp & popd & exit /b 1 )

cl /c %CFLAGS% %CL_INCLUDES% /Fo:"%BUILD_DIR%\pt_nvdec.obj" pt_nvdec.cpp
if errorlevel 1 ( echo BUILD FAILED: pt_nvdec.cpp & popd & exit /b 1 )

cl /c %CFLAGS% %CL_INCLUDES% /Fo:"%BUILD_DIR%\pt_matching.obj" ..\pt_shared\pt_matching.cpp
if errorlevel 1 ( echo BUILD FAILED: pt_matching.cpp & popd & exit /b 1 )

cl /c %CFLAGS% %CL_INCLUDES% /Fo:"%BUILD_DIR%\pt_triangulation.obj" ..\pt_shared\pt_triangulation.cpp
if errorlevel 1 ( echo BUILD FAILED: pt_triangulation.cpp & popd & exit /b 1 )

cl /c %CFLAGS% %CL_INCLUDES% /Fo:"%BUILD_DIR%\pt_tracker.obj" ..\pt_shared\pt_tracker.cpp
if errorlevel 1 ( echo BUILD FAILED: pt_tracker.cpp & popd & exit /b 1 )

cl /c %CFLAGS% %CL_INCLUDES% /Fo:"%BUILD_DIR%\pt_export.obj" ..\pt_shared\pt_export.cpp
if errorlevel 1 ( echo BUILD FAILED: pt_export.cpp & popd & exit /b 1 )

cl /c %CFLAGS% %CL_INCLUDES% /Fo:"%BUILD_DIR%\pt_pipeline.obj" pt_pipeline.cpp
if errorlevel 1 ( echo BUILD FAILED: pt_pipeline.cpp & popd & exit /b 1 )

cl /c %CFLAGS% %CL_INCLUDES% /Fo:"%BUILD_DIR%\pt_stream.obj" pt_stream.cpp
if errorlevel 1 ( echo BUILD FAILED: pt_stream.cpp & popd & exit /b 1 )

echo.
echo ---- Linking DLL ----

:: Build link flags with optional OpenCV lib
set LINK_LIBS=cuda.lib cudart.lib nvinfer.lib nvonnxparser.lib
set LINK_PATHS=/LIBPATH:"%CUDA_PATH%\lib\x64" /LIBPATH:"%TENSORRT_PATH%\lib"

if defined OPENCV_PATH (
    REM Use release lib only - no 'd' suffix. Debug/release CRT mismatch causes heap corruption.
    if exist "%OPENCV_PATH%\x64\vc16\lib" (
        set LINK_PATHS=%LINK_PATHS% /LIBPATH:"%OPENCV_PATH%\x64\vc16\lib"
        set LINK_LIBS=%LINK_LIBS% opencv_world4130.lib
    ) else (
        set LINK_PATHS=%LINK_PATHS% /LIBPATH:"%OPENCV_PATH%\lib"
        set LINK_LIBS=%LINK_LIBS% opencv_world4130.lib
    )
)

link /DLL /DEF:calimerge_cuda.def /OUT:"%BUILD_DIR%\calimerge_cuda.dll" ^
    "%BUILD_DIR%\pt_arena.obj" "%BUILD_DIR%\pt_kernels.obj" ^
    "%BUILD_DIR%\pt_tensorrt.obj" "%BUILD_DIR%\pt_nvdec.obj" "%BUILD_DIR%\pt_matching.obj" ^
    "%BUILD_DIR%\pt_triangulation.obj" "%BUILD_DIR%\pt_tracker.obj" "%BUILD_DIR%\pt_export.obj" ^
    "%BUILD_DIR%\pt_pipeline.obj" "%BUILD_DIR%\pt_stream.obj" ^
    %LINK_LIBS% ^
    %LINK_PATHS%

if errorlevel 1 (
    echo LINK FAILED
    popd
    exit /b 1
)

echo.
echo ---- DLL Exports ----
dumpbin /exports "%BUILD_DIR%\calimerge_cuda.dll" 2>nul | findstr "pt_"

echo.
echo ---- Building test program (pt_main.exe) ----

cl %CFLAGS% %CL_INCLUDES% /Fe:"%BUILD_DIR%\pt_main.exe" pt_main.cpp ^
    "%BUILD_DIR%\pt_arena.obj" "%BUILD_DIR%\pt_kernels.obj" ^
    "%BUILD_DIR%\pt_tensorrt.obj" "%BUILD_DIR%\pt_nvdec.obj" "%BUILD_DIR%\pt_matching.obj" ^
    "%BUILD_DIR%\pt_triangulation.obj" "%BUILD_DIR%\pt_tracker.obj" "%BUILD_DIR%\pt_export.obj" ^
    "%BUILD_DIR%\pt_pipeline.obj" "%BUILD_DIR%\pt_stream.obj" ^
    /link %LINK_LIBS% %LINK_PATHS%

set BUILD_ERRORS=0

if errorlevel 1 (
    echo FAILED: pt_main.exe
    set /a BUILD_ERRORS+=1
) else (
    echo OK: pt_main.exe
)

echo.
echo ---- Building streaming test program (pt_stream_main.exe) ----

cl %CFLAGS% %CL_INCLUDES% /Fe:"%BUILD_DIR%\pt_stream_main.exe" pt_stream_main.cpp ^
    "%BUILD_DIR%\pt_arena.obj" "%BUILD_DIR%\pt_kernels.obj" ^
    "%BUILD_DIR%\pt_tensorrt.obj" "%BUILD_DIR%\pt_nvdec.obj" "%BUILD_DIR%\pt_matching.obj" ^
    "%BUILD_DIR%\pt_triangulation.obj" "%BUILD_DIR%\pt_tracker.obj" "%BUILD_DIR%\pt_export.obj" ^
    "%BUILD_DIR%\pt_pipeline.obj" "%BUILD_DIR%\pt_stream.obj" ^
    /link %LINK_LIBS% %LINK_PATHS%

if errorlevel 1 (
    echo FAILED: pt_stream_main.exe
    set /a BUILD_ERRORS+=1
) else (
    echo OK: pt_stream_main.exe
)

:: Clean up .obj files left in source dir by cl
del /q *.obj 2>nul

popd

echo.
echo ============================================================
echo  Build Summary
echo ============================================================
echo  Output:  %BUILD_DIR%
echo  Errors:  %BUILD_ERRORS%
echo.
echo  Artifacts:
if exist "%BUILD_DIR%\calimerge_cuda.dll" (echo    OK  calimerge_cuda.dll) else (echo    MISSING  calimerge_cuda.dll)
if exist "%BUILD_DIR%\pt_main.exe"        (echo    OK  pt_main.exe)        else (echo    MISSING  pt_main.exe)
if exist "%BUILD_DIR%\pt_stream_main.exe" (echo    OK  pt_stream_main.exe) else (echo    MISSING  pt_stream_main.exe)
echo ============================================================

if %BUILD_ERRORS% GTR 0 (
    echo  *** BUILD HAD %BUILD_ERRORS% ERROR(S) ***
    exit /b 1
)
