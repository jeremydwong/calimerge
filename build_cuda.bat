@echo off
:: One-line CUDA pipeline rebuild. Run from repo root.
:: Usage: build_cuda.bat [release]
if not defined OPENCV_PATH set OPENCV_PATH=C:\OpenCV\opencv\build
if not defined TENSORRT_PATH set TENSORRT_PATH=C:\TensorRT

:: Delete old DLL so we can tell if the build actually produced a new one
del /q build\cuda\calimerge_cuda.dll 2>nul

pushd src\cuda_pipeline
call build_cuda_win32.bat %1
popd

echo.
if exist build\cuda\calimerge_cuda.dll (
    echo === CUDA BUILD SUCCEEDED (v0.2.1) ===
) else (
    echo === CUDA BUILD FAILED ===
    exit /b 1
)
