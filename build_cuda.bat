@echo off
:: One-line CUDA pipeline rebuild. Run from repo root.
:: Usage: build_cuda.bat [release]
set OPENCV_PATH=C:\OpenCV\opencv\build
set TENSORRT_PATH=C:\TensorRT
cd src\cuda_pipeline && call build_cuda_win32.bat %1
if errorlevel 1 (
    echo.
    echo *** CUDA BUILD FAILED ***
    cd ..\..
    exit /b 1
)
cd ..\..
echo.
echo *** CUDA BUILD SUCCEEDED (v0.2.0) ***
