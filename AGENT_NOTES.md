# Agent Notes — cpp/scaffold branch

This branch adds the initial `src/app/` scaffold for the C++ Qt6 GUI described in `design_cpp.md` §9 (Phase 3). The following files were created under `src/app/`:

- `AppState.h` — plain C structs porting the Python frozen dataclasses in `src/calimerge/types.py` and the sub-states in `src/calimerge/gui/state.py`; includes `CameraConfig`, `CameraIntrinsics`, `CameraExtrinsics`, `CalibratedCamera`, `CharucoConfig`, `RecordingState`, `CameraState`, `CalibrationState`, `ProcessingState`, and `AppState` with a `QMap<int, CameraState>` camera map.
- `StateManager.h` / `StateManager.cpp` — `QObject` subclass that holds one `AppState` value and emits `stateChanged`, `camerasChanged`, `recordingChanged`, `processingChanged`, `statusMessage`, and `errorOccurred` signals; skeleton only, no workers yet.
- `MainWindow.h` / `MainWindow.cpp` — `QMainWindow` with four placeholder tabs (Record, Intrinsic, Extrinsic, Process), a menu bar with File → Quit, and a status bar wired to `StateManager::statusMessage`; skeleton only, no real tab content.
- `main.cpp` — standard Qt entry point; creates `QApplication`, `StateManager`, `MainWindow`; sets app name "Calimerge" and org "Calimerge".
- `app_unity.cpp` — single compilation unit that `#include`s every `.cpp` file and the MOC output in `gen/`; commented-out includes mark where tabs/widgets/workers will be added.
- `build_win32.bat` — MSVC unity build script; calls `VsDevCmd.bat`, checks for Qt6 at `QT_DIR` (default `C:\Qt\6.9.0\msvc2022_64`) and prints a clear install URL if not found; runs `moc.exe` on all `.h` files in root, `tabs/`, `widgets/`, and `workers/` (skips missing dirs); runs `rcc.exe` on `resources.qrc` if present; compiles `app_unity.cpp` with Qt6 includes/libs; outputs to `build/app/calimerge.exe`.
- `build_macos.sh` — clang++ mirror of the Windows script; searches `QT_DIR`, Homebrew Intel/Apple Silicon paths, and `~/Qt/6.*/macos`; same MOC/RCC/unity structure; outputs to `build/app/calimerge`.
- `.gitignore` — ignores the `gen/` directory (MOC/RCC output).

**What the next agent must know before the build can be tested:**

1. Qt 6 is not installed on this machine. Install it from https://www.qt.io/download-open-source, selecting the "MSVC 2022 64-bit" component on Windows or via `brew install qt@6` on macOS, then set `QT_DIR` to the install prefix (e.g. `set QT_DIR=C:\Qt\6.9.0\msvc2022_64`) and run `src\app\build_win32.bat release` from Git Bash using `./build_win32.bat release`.
2. The build currently emits only the three root-level `Q_OBJECT` classes (StateManager, MainWindow) plus `main.cpp`; tabs, widgets, and workers are stubs to be filled in during subsequent Phase 3 steps per `design_cpp.md §6` items 12–16.
3. When adding a new class with `Q_OBJECT`: (a) put the class declaration in a `.h` file, (b) add an `#include` of its `.cpp` in `app_unity.cpp`, and (c) add a `gen/moc_<ClassName>.cpp` include below the implementation section in `app_unity.cpp`; the build script runs `moc` unconditionally on every `.h` in the relevant directory.
4. `AppState` uses `QMap` for the camera and calibration maps; that requires `#include <QMap>` to be satisfied by the Qt headers — this is fine because `AppState.h` is only ever included in TUs that already link Qt.
