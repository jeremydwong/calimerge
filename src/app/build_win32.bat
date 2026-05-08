@echo off
:: build_win32.bat - Unity build for the Calimerge Qt6 GUI (Windows / MSVC)
::
:: Usage:
::   build_win32.bat          (debug, default)
::   build_win32.bat release  (optimised, NDEBUG)
::
:: Prerequisites:
::   - MSVC Build Tools 2022 (installs automatically via VsDevCmd.bat below)
::   - Qt 6.x for MSVC 2022 64-bit
::     Either:  set QT_DIR=C:\Qt\6.x.x\msvc2022_64  before running, or
::              install to the default path C:\Qt\6.9.0\msvc2022_64
::     If Qt is not found the script prints an install URL and exits with 1.
::
:: Output: ..\..\build\app\calimerge.exe
::
:: Style: follows src/native/build_win32.bat and
::        src/cuda_pipeline/build_cuda_win32.bat — no CMake, no Makefiles,
::        direct MSVC invocation.  See design_cpp.md §5.

echo ============================================================
echo  Calimerge Qt6 GUI - Windows Build
echo ============================================================
echo.

setlocal

:: ---- MSVC ---------------------------------------------------------------
:: vswhere.exe is not on PATH by default; prepend its dir to suppress the
:: harmless "not recognized" warning VsDevCmd.bat emits when it can't find it.
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64
if errorlevel 1 (
    echo ERROR: VsDevCmd.bat failed. Is MSVC Build Tools 2022 installed?
    echo        Install from: https://aka.ms/vs/17/release/vs_BuildTools.exe
    exit /b 1
)

:: ---- Qt6 ----------------------------------------------------------------
:: Prefer caller-supplied QT_DIR; fall back to a known default location.
if not defined QT_DIR (
    set QT_DIR=C:\Qt\6.9.0\msvc2022_64
)

:: Validate Qt headers exist before doing any real work.
if not exist "%QT_DIR%\include\QtCore\qobject.h" (
    echo.
    echo ERROR: Qt6 not found at QT_DIR="%QT_DIR%"
    echo.
    echo  Fix options:
    echo   1. Install Qt 6 from https://www.qt.io/download-open-source
    echo      and select "MSVC 2022 64-bit" component.
    echo   2. Set QT_DIR to the correct path before running this script, e.g.:
    echo        set QT_DIR=C:\Qt\6.x.x\msvc2022_64
    echo        call src\app\build_win32.bat release
    echo.
    exit /b 1
)

set MOCBIN=%QT_DIR%\bin\moc.exe
set RCCBIN=%QT_DIR%\bin\rcc.exe

:: ---- Build mode ---------------------------------------------------------
if "%1"=="release" (
    set CFLAGS=/W4 /WX /wd4100 /wd4201 /wd4189 /O2 /MD /EHsc /DNDEBUG
    echo Build mode: RELEASE
) else (
    set CFLAGS=/W4 /WX /wd4100 /wd4201 /wd4189 /Od /Zi /MDd /EHsc /DDEBUG
    echo Build mode: DEBUG
)

:: ---- Paths --------------------------------------------------------------
set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..\..\
set BUILD_DIR=%REPO_ROOT%build\app
set GEN_DIR=%SCRIPT_DIR%gen

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%GEN_DIR%"   mkdir "%GEN_DIR%"

pushd %SCRIPT_DIR%

:: ---- Step 1: MOC --------------------------------------------------------
echo.
echo ---- Running moc on Q_OBJECT headers ----

:: Run moc on every .h in src/app/ root
for %%H in (*.h) do (
    echo   moc %%H
    "%MOCBIN%" %%H -o "%GEN_DIR%\moc_%%~nH.cpp"
    if errorlevel 1 (
        echo MOC FAILED: %%H
        popd
        exit /b 1
    )
)

:: Run moc on tabs/ headers (skip if directory not present yet)
if exist tabs\ (
    for %%H in (tabs\*.h) do (
        echo   moc %%H
        "%MOCBIN%" %%H -o "%GEN_DIR%\moc_%%~nH.cpp"
        if errorlevel 1 (
            echo MOC FAILED: %%H
            popd
            exit /b 1
        )
    )
)

:: Run moc on widgets/ headers
if exist widgets\ (
    for %%H in (widgets\*.h) do (
        echo   moc %%H
        "%MOCBIN%" %%H -o "%GEN_DIR%\moc_%%~nH.cpp"
        if errorlevel 1 (
            echo MOC FAILED: %%H
            popd
            exit /b 1
        )
    )
)

:: Run moc on workers/ headers
if exist workers\ (
    for %%H in (workers\*.h) do (
        echo   moc %%H
        "%MOCBIN%" %%H -o "%GEN_DIR%\moc_%%~nH.cpp"
        if errorlevel 1 (
            echo MOC FAILED: %%H
            popd
            exit /b 1
        )
    )
)

:: ---- Step 2: RCC (Qt resources) ----------------------------------------
if exist resources.qrc (
    echo.
    echo ---- Running rcc on resources.qrc ----
    "%RCCBIN%" resources.qrc -o "%GEN_DIR%\resources.cpp"
    if errorlevel 1 (
        echo RCC FAILED: resources.qrc
        popd
        exit /b 1
    )
)

:: ---- Step 3: Unity compile ----------------------------------------------
echo.
echo ---- Compiling app_unity.cpp ----

set QT_INC=/I"%QT_DIR%\include" ^
           /I"%QT_DIR%\include\QtCore" ^
           /I"%QT_DIR%\include\QtWidgets" ^
           /I"%QT_DIR%\include\QtGui"

set QT_LIB=/LIBPATH:"%QT_DIR%\lib"

if "%1"=="release" (
    set QT_LIBS=Qt6Core.lib Qt6Widgets.lib Qt6Gui.lib
) else (
    set QT_LIBS=Qt6Cored.lib Qt6Widgetsd.lib Qt6Guid.lib
)

cl %CFLAGS% ^
   %QT_INC% ^
   /I"." ^
   /Fe:"%BUILD_DIR%\calimerge.exe" ^
   app_unity.cpp ^
   /link %QT_LIB% %QT_LIBS%

if errorlevel 1 (
    echo.
    echo BUILD FAILED
    popd
    exit /b 1
)

:: Clean up stray .obj files the compiler drops in the source dir
del /q *.obj 2>nul

popd

echo.
echo ============================================================
echo  Build OK: %BUILD_DIR%\calimerge.exe
echo ============================================================
