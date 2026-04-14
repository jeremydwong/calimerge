@echo off
:: One-line CUDA pipeline rebuild. Run from repo root.
:: Usage: build_cuda.bat [release]
if not defined OPENCV_PATH set OPENCV_PATH=C:\OpenCV\opencv\build
if not defined TENSORRT_PATH set TENSORRT_PATH=C:\TensorRT
pushd src\cuda_pipeline
call build_cuda_win32.bat %1
set BUILD_RC=%errorlevel%
popd
if %BUILD_RC% NEQ 0 (
    echo.
    echo === CUDA BUILD FAILED ===
    exit /b 1
)
echo.
echo === CUDA BUILD SUCCEEDED (v0.2.1) ===
