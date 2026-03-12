@echo off
:: run_test.bat - Run a Python test script with uv, handling VIRTUAL_ENV
:: Usage: run_test.bat <script.py> [args...]
::
:: Example: run_test.bat test_recording.py
::          run_test.bat test_cameras.py

set VIRTUAL_ENV=
cd /d C:\Git\calimerge
C:\Users\wongj\.local\bin\uv run python %*
