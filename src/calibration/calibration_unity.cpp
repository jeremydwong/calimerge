/*
 * calibration_unity.cpp
 *
 * Single compilation unit for the cm_calibration DLL.
 *
 * Following the same unity-build pattern as calimerge_win32.cpp in src/native/:
 * compile only this file — it pulls in the full implementation via #include.
 *
 * Build:
 *   Windows: src/calibration/build_win32.bat
 *   macOS:   src/calibration/build_macos.sh
 *
 * Output:
 *   Windows: build/calibration/cm_calibration.dll + cm_calibration.lib
 *   macOS:   build/calibration/libcm_calibration.dylib
 */

#include "charuco.cpp"
#include "intrinsic.cpp"
