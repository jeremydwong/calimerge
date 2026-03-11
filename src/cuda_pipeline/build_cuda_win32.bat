@echo off
:: build_cuda_win32.bat - Build the CUDA pose tracking pipeline DLL (Windows)
::
:: Usage: build_cuda_win32.bat [release]
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

pushd %~dp0

echo.
echo ---- Compiling CUDA sources (.cu) ----

:: sm_120 = Blackwell (RTX 5070 Ti)
nvcc -c %NVCC_OPT% --use_fast_math ^
    -gencode arch=compute_120,code=sm_120 ^
    -Xcompiler "/MD %OPTIMIZE% /EHsc" ^
    -I"%CUDA_PATH%\include" ^
    -I"%TENSORRT_PATH%\include" ^
    -o pt_arena.obj pt_arena.cu

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
    -o pt_kernels.obj pt_kernels.cu

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

cl /c %CFLAGS% %CL_INCLUDES% /Fo:pt_tensorrt.obj pt_tensorrt.cpp
if errorlevel 1 ( echo BUILD FAILED: pt_tensorrt.cpp & popd & exit /b 1 )

cl /c %CFLAGS% %CL_INCLUDES% /Fo:pt_nvdec.obj pt_nvdec.cpp
if errorlevel 1 ( echo BUILD FAILED: pt_nvdec.cpp & popd & exit /b 1 )

cl /c %CFLAGS% %CL_INCLUDES% /Fo:pt_matching.obj ..\pt_shared\pt_matching.cpp
if errorlevel 1 ( echo BUILD FAILED: pt_matching.cpp & popd & exit /b 1 )

cl /c %CFLAGS% %CL_INCLUDES% /Fo:pt_triangulation.obj ..\pt_shared\pt_triangulation.cpp
if errorlevel 1 ( echo BUILD FAILED: pt_triangulation.cpp & popd & exit /b 1 )

cl /c %CFLAGS% %CL_INCLUDES% /Fo:pt_tracker.obj ..\pt_shared\pt_tracker.cpp
if errorlevel 1 ( echo BUILD FAILED: pt_tracker.cpp & popd & exit /b 1 )

cl /c %CFLAGS% %CL_INCLUDES% /Fo:pt_export.obj ..\pt_shared\pt_export.cpp
if errorlevel 1 ( echo BUILD FAILED: pt_export.cpp & popd & exit /b 1 )

cl /c %CFLAGS% %CL_INCLUDES% /Fo:pt_pipeline.obj pt_pipeline.cpp
if errorlevel 1 ( echo BUILD FAILED: pt_pipeline.cpp & popd & exit /b 1 )

cl /c %CFLAGS% %CL_INCLUDES% /Fo:pt_stream.obj pt_stream.cpp
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

link /DLL /DEF:calimerge_cuda.def /OUT:calimerge_cuda.dll ^
    pt_arena.obj pt_kernels.obj ^
    pt_tensorrt.obj pt_nvdec.obj pt_matching.obj ^
    pt_triangulation.obj pt_tracker.obj pt_export.obj ^
    pt_pipeline.obj pt_stream.obj ^
    %LINK_LIBS% ^
    %LINK_PATHS%

if errorlevel 1 (
    echo LINK FAILED
    popd
    exit /b 1
)

echo.
echo ---- DLL Exports ----
dumpbin /exports calimerge_cuda.dll 2>nul | findstr "pt_"

echo.
echo ---- Building test program (pt_main.exe) - monolithic build ----
echo      (links .obj files directly, bypasses DLL boundary)

cl %CFLAGS% %CL_INCLUDES% /Fe:pt_main.exe pt_main.cpp ^
    pt_arena.obj pt_kernels.obj ^
    pt_tensorrt.obj pt_nvdec.obj pt_matching.obj ^
    pt_triangulation.obj pt_tracker.obj pt_export.obj ^
    pt_pipeline.obj pt_stream.obj ^
    /link %LINK_LIBS% %LINK_PATHS%

if errorlevel 1 (
    echo WARNING: pt_main.exe build failed (non-fatal)
) else (
    echo pt_main.exe built successfully
)

echo.
echo ---- Building streaming test program (pt_stream_main.exe) ----

cl %CFLAGS% %CL_INCLUDES% /Fe:pt_stream_main.exe pt_stream_main.cpp ^
    pt_arena.obj pt_kernels.obj ^
    pt_tensorrt.obj pt_nvdec.obj pt_matching.obj ^
    pt_triangulation.obj pt_tracker.obj pt_export.obj ^
    pt_pipeline.obj pt_stream.obj ^
    /link %LINK_LIBS% %LINK_PATHS%

if errorlevel 1 (
    echo WARNING: pt_stream_main.exe build failed (non-fatal)
) else (
    echo pt_stream_main.exe built successfully
)

popd

echo.
echo ============================================================
echo  Build complete.
echo ============================================================
