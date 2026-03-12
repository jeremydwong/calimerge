@echo off
:: compile.bat - Compile a single C++ file with MSVC
:: Usage: compile.bat <source.cpp> [extra libs...]
::
:: Example: compile.bat test_uvc_probe.cpp ole32.lib strmiids.lib

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul 2>&1

pushd %~dp0

set SRC=%1
shift

set LIBS=
:loop
if "%1"=="" goto done
set LIBS=%LIBS% %1
shift
goto loop
:done

echo Compiling %SRC%...
cl /EHsc /O2 /W3 %SRC% /link %LIBS%

popd
