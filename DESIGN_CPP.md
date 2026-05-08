# DESIGN_CPP.md — A C++ rewrite of calimerge: design, sequencing, and an honest verdict

> Status: active plan. Full rewrite including GUI confirmed. Build system is unity scripts (Casey Muratori style) — no CMake, no Makefiles. Read §8 for the updated verdict and §9 for the parallel folder structure.

---

## 1. Why rewrite — the sober version

The trigger today was real: coremltools/ultralytics + macOS `fork()` + multiprocessing, ONNX→CoreML conversion brittleness, `uv run --with` plumbing for transient deps. Every one of those was a Python-ecosystem cost. But before declaring Python the disease, separate **chronic** from **acute**:

**Acute, one-time costs (will not recur daily):**
- ONNX→CoreML conversion is run once per model-version bump (see `build_mac_models.sh:67-70`). The `--with coremltools --with onnx2torch` workaround exists *because* the user already deliberately quarantined coremltools from `uv.lock` — that quarantine is working as designed. Today it broke. It will not break again until the next bump.
- The `fork()` problem is a coremltools+ultralytics interaction at conversion time, not a runtime cost in the GUI. The shipped app doesn't fork.

**Chronic costs (will recur):**
- PySide6 interpreter startup time (~1.5–3 s cold).
- Heavyweight Python deps in the lockfile (torch, transformers, ultralytics, mediapipe) for runtime use cases that are now mostly served by C++ pipelines (`src/cuda_pipeline/`, `src/mps_pipeline/`).
- Two-language debugging (ctypes boundary at `cuda_binding.py:268`, `mps_binding.py`, `mps_offline_binding.py`, `mps_stream_binding.py` — four bindings to maintain).
- `VIRTUAL_ENV=` prefix gymnastics on Windows (CLAUDE.md:97-105). Also chronic.
- The C++ pipelines are already the live path — Python just orchestrates. The Python orchestrator is a thinning veneer.

**What C++ genuinely improves**
- Single-binary distribution (.app, .exe, .dmg, .msi) without a Python runtime.
- Cold start under 200 ms.
- One process model — no Qt-thread/Python-GIL interaction, no ctypes marshalling for hot data.
- One source of truth for keypoint shape: today `pt_common.h:34` (`PT_NUM_KEYPOINTS`) and the Python schema in `types.py` already drift; DESIGN.md §1.2 calls this out.

**What C++ makes worse**
- Calibration tuning loop. Today the user can edit `src/calimerge/calibration/extrinsic.py`, hit save, re-run. In C++ that's edit → 10–30 s rebuild → re-run.
- Notebook exploration (`notebooks/test_output.ipynb`, `notebooks/test_output_csv.ipynb`). C++ has no Jupyter analogue. xeus-cling is dead. cppyy is not it.
- The perturbation study (`src/calimerge/analysis/perturbation_study.py`, 840 lines of cv2+pandas+numpy) is exactly the kind of code that *should* be Python.
- Build matrix complexity: today macOS unity build + Windows MSVC + CUDA + CoreML are decoupled. A unified C++ binary couples them.

**The honest reframe.** The Python pain in calimerge is not "Python is slow" or "Python is the wrong language" — it's "the Python *bindings to ML toolchains* (coremltools, ultralytics, transformers) are fragile, and we already have C++ pipelines that don't need them." So the question is not "rewrite calimerge in C++" — it's **"can we delete the runtime Python ML dependencies and keep Python only for the calibration/analysis/notebook surface where it's a strict win?"**

That reframe changes the answer. See §8.

---

## 2. Surface inventory

Total Python: **~32k LOC**. Total existing C/C++: **~18.5k LOC**. The rewrite would roughly double the C++ surface, not multiply it 10×.

### Tier 1 — easy ports / mostly already C++

| Module | LOC | Current role | C++ equivalent | Difficulty | Risk |
|---|---|---|---|---|---|
| `src/calimerge/triangulation.py` | 341 | Numba-jitted DLT triangulation | `src/pt_shared/pt_triangulation.cpp` (already exists, 378 lines) | **Trivial** — call existing | None |
| `src/calimerge/tracking/cuda_binding.py` | 397 | ctypes shim into CUDA pipeline | Delete; call `pt_pipeline.cpp` directly | **Trivial** | None |
| `src/calimerge/tracking/mps_binding.py` + `mps_offline_binding.py` + `mps_stream_binding.py` | 1179 combined | ctypes shim into MPS pipeline | Delete; call `pt_offline_mps.m`/`pt_stream_mps.m` directly | **Trivial** | None |
| `src/calimerge/camera_binding.py` | 531 | ctypes shim into AVFoundation/MediaFoundation | Delete; call `calimerge_macos.mm`/`calimerge_win32.cpp` directly | **Trivial** | None |
| `src/calimerge/tracking/markers.py` | 36 | Hip indices, COM math | trivial constants | **Trivial** | None |
| `src/calimerge/tracking/track_stitch.py` | 210 | Per-track stitching after C tracker | Move into `pt_tracker.cpp` (where the comments in CLAUDE.md:25 say it should be) | Low | Behavior must match exactly — fixture in `tests/data/zelda_*` validates |
| `src/calimerge/skeleton_3d.py` | 160 | XYZPoints → (N,P,K,3) | std::vector + per-frame join with `frame_time_history.csv` | Low | None |
| `src/calimerge/clock_widget.py` | 89 | Sync verification clock | Qt6 widget, near 1:1 | Low | None |

### Tier 2 — medium ports

| Module | LOC | C++ equivalent | Difficulty | Risk |
|---|---|---|---|---|
| `src/calimerge/calibration/charuco.py` | 329 | OpenCV C++ aruco module — 1:1 API | Low | PDF generation needs a C++ PDF lib (PDFWriter, libharu) — small surface |
| `src/calimerge/calibration/intrinsic.py` | 324 | OpenCV C++ `calibrateCamera` | Low | None |
| `src/calimerge/calibration/extrinsic.py` | **1156** | Stereo pair init: OpenCV C++. Bundle adjustment: **Ceres Solver** | **Medium-High** | Bundle adjustment is the single hardest port. SciPy `least_squares` with sparse Jacobian → Ceres. Different convergence tolerances will shift RMSE in the 4th decimal. Requires regression suite. |
| `src/calimerge/types.py` | 384 | Plain structs in `pt_common.h`-style header | Low | None |
| `src/calimerge/config.py` | **1903** | toml++ + sqlite3 + std::filesystem | **Medium** | Big but mechanical. SQL schema unchanged. |
| `src/calimerge/programs.py` + `workout_types.py` + `workout_spec_db.py` | 1075 | Plain structs + nlohmann/json + sqlite3 | Medium | Migration in DESIGN.md §2 simplifies this — they should already be JSON-driven |
| `src/calimerge/keypoint_export.py` | 828 | std::ofstream + cnpy for npz | Medium | npz round-trip must match exactly; downstream `notebooks/test_output*.ipynb` consume these — that's the contract |
| `src/calimerge/tracking/pipeline.py` | 784 | Already mostly orchestration of CUDA/MPS pipelines | Medium | Drops with the bindings |
| `src/calimerge/tracking/tracker.py` | 500 | Mirror of `pt_tracker.cpp` for live MediaPipe path | Medium | None |
| `src/calimerge/video_recorder.py` | 353 | Wrap FFmpeg / AVFoundation H.264 encoder direct | Medium | Filename+CSV layout in CLAUDE.md:264-275 is fixed |

### Tier 3 — hard / reconsider scope

| Module | LOC | Issue | Recommendation |
|---|---|---|---|
| `src/calimerge/gui/workout_page.py` | **4550** | Largest single file. Heavy state machine: login → camera init → calibration check → live pose → recording → results. PySide6 surface is 50+ widgets, signals, timers. | Direct port to Qt6 C++ — same widgets exist 1:1. ~8000 LOC C++ estimated. **Significant effort, low risk per line.** |
| `src/calimerge/gui/workers.py` | **2735** | QThread workers wrapping every async op | Direct port to QThread C++ subclasses. ~4500 LOC. Mechanical. |
| `src/calimerge/gui/tabs/*` | 4127 combined | Calibration tabs (cameras, intrinsic, extrinsic, process) | Direct Qt6 C++ port |
| `src/calimerge/gui/widgets/skeleton_view.py` | 580 | Custom 3D pyqtgraph view | **Hard** — pyqtgraph has no C++ equivalent. Replace with Qt3D, OpenGL via QOpenGLWidget, or VTK. Each adds a heavy dep. |
| `src/calimerge/gui/progress_graph.py` | 220 | pyqtgraph live FPS plot | Qt Charts (lightweight) — slightly clunkier API, fine |
| `src/calimerge/gui/unified_offline_worker.py` | 753 | Drives live primitive over recorded video — designed to share code with live | Mechanical port |
| `src/calimerge/analysis/*.py` | 2700 combined | Rep counters, joint angles, balance, TUG, stretch | **Don't port. Keep Python.** See §8. |
| `src/calimerge/analysis/perturbation_study.py` | **840** | numpy/pandas Monte Carlo perturbation sensitivity | **Don't port. Keep Python.** |
| `notebooks/*.ipynb` | — | Jupyter — exploratory data analysis | **Don't port. Cannot port. Keep Python.** |
| `scripts/export_*.py` (`export_onnx.py`, `export_synthpose_onnx.py`) | — | One-time PyTorch → ONNX | **Don't port. Run once, ship the .onnx.** This is 80% of today's pain. |
| `scripts/convert_models_coreml.py` | — | ONNX → CoreML | **Don't port. Run once, ship the .mlpackage.** |
| `tests/manual/build_coreml_models.py` | — | The fragile coremltools step | **Same.** |

**Summary:** of ~32k Python LOC, roughly 22k is mechanical port to C++ (mostly GUI), 5k is medium-difficulty (calibration math, config, persistence), 2k is hard (skeleton view, bundle adjustment), and **~3k should stay Python** (analysis modules, perturbation study, model export scripts, notebooks).

---

## 3. Architecture

### Process model

**Single binary, single process, multiple threads.** No multi-process GPU isolation needed: TensorRT and CoreML both work fine in-process. The Python ctypes boundary that today separates the GUI process from inference goes away — inference becomes a thread inside the GUI binary.

The one place to consider a worker process: model conversion (ONNX → engine, ONNX → CoreML). But §8's recommendation is **don't do model conversion in the shipped binary at all** — ship pre-built artifacts. So no subprocess.

### Threading

```
main thread          : Qt event loop, all UI
camera capture       : N threads (one per AVCaptureSession / IMFSourceReader),
                       same as today's calimerge_macos.mm / calimerge_win32.cpp
inference dispatch   : 1 thread, owns TensorRT/CoreML engines
inference workers    : (TensorRT) async streams; (CoreML) MLModel.predict
recording encoder    : N threads (one per camera), H.264 encode
analysis             : on-demand thread for offline post-processing
                       (this is where Python embedded interpreter runs — see §4)
```

### Data flow

```
[cameras] ─► ring buffer ─► sync ─► [inference thread] ─► tracker ─► triangulator ─► [skeleton view]
                                                                         │
                                                                         ├──► npz/csv writer
                                                                         └──► (offline only) Python analysis layer
```

This is the same shape as the existing `pt_pipeline.cpp` + `pt_stream.cpp`. The new code is just the GUI surface and the calibration pipeline.

### Where Python *stays*

A small embedded CPython 3.11 interpreter, statically linked, used **only** for:
1. Running analysis modules (`src/calimerge/analysis/*.py`) on completed sessions.
2. Optionally re-exporting the perturbation study.
3. Optionally launching Jupyter (`jupyter lab` as a separate process the GUI just spawns).

The interpreter is invoked via pybind11 with a narrow API:

```
analysis::run(skeleton_3d_npz_path, analyzer_id, params_json) -> result_json
```

The C++ side never imports torch/transformers/ultralytics/coremltools/mediapipe. Those drop entirely. The Python side imports only numpy, scipy, pandas, matplotlib — small, stable, ABI-friendly.

This is **the key architectural choice.** It keeps the surface where Python is genuinely better (data analysis on saved data) while removing the surface where Python is genuinely worse (ML toolchain bindings, GUI startup time, two-language debug).

---

## 4. Library choices, with reasoning

### GUI: **Qt 6 / Qt Widgets** (not QML, not ImGui, not Slint)

The existing GUI is already PySide6/Qt Widgets. Direct port: `QMainWindow → QMainWindow`, `QPushButton → QPushButton`, `QThread → QThread`. Signals/slots are the same concepts. This is the lowest-risk choice by a wide margin.

Rejected:
- **QML** — would require redesigning every screen. Six month detour. No win.
- **ImGui** — wrong fit for a settings-heavy, document-style app. Excellent for tools, poor for production GUIs. Lacks accessibility, lacks IME, lacks native menus on macOS.
- **Slint** — promising, but small ecosystem and the user has zero Slint experience. Adoption cost too high for a single-developer project.
- **Native AppKit + WinUI** — three GUIs to maintain. Hard no.

Trade-off accepted: Qt 6 commercial license is $5k+/year for closed-source, or LGPL with dynamic linking. calimerge is BSD-2 (`pyproject.toml:5`), so LGPL dynamic linking is fine.

### Calibration math: **OpenCV C++ + Ceres Solver**

OpenCV C++ for everything currently using cv2 (charuco detection, calibrateCamera, stereoCalibrate, projectPoints, undistortPoints).

For bundle adjustment, replace `scipy.optimize.least_squares` (`extrinsic.py:14`) with **Ceres**, not g2o. Reasoning:
- Ceres has first-class sparse Jacobian support that matches what `lil_matrix` is doing today.
- Ceres has automatic differentiation; the current code computes the Jacobian sparsity by hand.
- Ceres is the de-facto BA solver in modern CV (COLMAP, OpenMVS use it).
- g2o is older, less actively maintained, and primarily SLAM-oriented.

**Risk:** Ceres convergence won't match scipy's Trust Region Reflective exactly. RMSE values will shift in the 3rd–4th decimal. The fixture in `tests/data/zelda_*` and the regression test rule in CLAUDE.md:25 are the safety net — re-run the harness post-port and accept the new baseline.

### Triangulation: **already done.** `src/pt_shared/pt_triangulation.cpp` is what `triangulation.py` should be. Delete the Python file once the calibration pipeline is C++.

### Pose backends: **already done.** `src/cuda_pipeline/` (TensorRT) and `src/mps_pipeline/` (CoreML) are the production paths. The PyTorch backend in `tracking/pose_detector.py` is a development convenience that should die — it's the source of half the dep weight (torch, transformers, ultralytics).

### SQLite: **direct C API, not SQLiteCpp**

The existing schema (`config.py:733-858`, `workouts.db`) is small. Direct `sqlite3.h` calls are 50% more code but zero new dependencies and the user already understands them. SQLiteCpp/sqlite_orm both add transitive complexity without earning their keep at this scale.

### TOML: **toml++ (tomlplusplus)**

Header-only, C++17, well-maintained, parses the actual TOML spec. The current Python code uses `rtoml` which is also strict-spec. Round-trip should match.

### JSON: **nlohmann/json**, not simdjson

simdjson is faster but read-only. nlohmann handles round-trip serialization, which the workout/program JSON in DESIGN.md §2.4 needs. Speed is irrelevant here — these are config files, not telemetry streams.

### HTTP / HuggingFace Hub: **drop entirely**

DESIGN.md §1.5 talks about `hf_hub_download` at runtime. **Don't.** Ship pre-converted models bundled with the binary (or in a versioned signed bundle the app downloads on first run from a static URL using the OS's HTTP API — `NSURLSession` on macOS, `WinHttp` on Windows). No libcurl, no openssl-from-source, no TLS cert chain debugging. This also eliminates the entire ONNX conversion runtime — the conversion step becomes a developer-machine build artifact, not a user-machine runtime.

This is the single biggest pain reduction in the rewrite.

### Plotting: **Qt Charts for FPS graph; QOpenGLWidget for skeleton view**

`progress_graph.py:220` (live FPS line plot) → Qt Charts QLineSeries. Sufficient.

`skeleton_view.py:580` (3D rotating pose) → custom QOpenGLWidget with hand-rolled GL or **Qt3D** module. Qt3D is heavyweight (additional 30 MB binary) but matches the Qt idiom. **Recommend QOpenGLWidget + a few hundred lines of GL** — this is a known, finite job, and the user has done equivalent work in `src/native/` already.

For the perturbation study charts (`plot_perturbation.py:337`) — that stays Python with matplotlib. Charts are output once; nobody is staring at them at 60 fps.

### Notebooks: **acknowledge the gap; keep them Python**

There is no equivalent. xeus-cling has been unmaintained since 2022. Do not pretend a C++ rewrite covers this. The embedded Python interpreter (§3) is the bridge: notebooks load `keypoints_3d.csv` / `keypoints_3d.raw.npz` directly, exactly as today. Nothing changes for notebook workflows.

---

## 5. Build system

**Unity build scripts — no CMake, no Makefiles, no build system.**

The existing lower layers (`src/native/`, `src/cuda_pipeline/`, `src/mps_pipeline/`) already use this approach successfully. It extends naturally to the calibration library and the Qt GUI. The argument that "MOC + Ceres need CMake" is wrong — both are just code generators and linker inputs that a build script handles in 10 lines.

### How Qt MOC works in a unity build

Qt's meta-object compiler is a preprocessor step, not a build system concept. A build script runs it explicitly, collects the output, and includes it in the unity compilation unit:

```bat
:: build_app_win32.bat — run from src/app/
:: Step 1: generate MOC output for every header with Q_OBJECT
set MOCBIN=C:\Qt\6.x\msvc2022_64\bin\moc.exe
for %%H in (*.h tabs\*.h widgets\*.h workers\*.h) do (
    %MOCBIN% %%H -o gen\moc_%%~nH.cpp
)

:: Step 2: run rcc for Qt resources (icons, shaders)
C:\Qt\6.x\msvc2022_64\bin\rcc.exe resources.qrc -o gen\resources.cpp

:: Step 3: unity compile
set QT=C:\Qt\6.x\msvc2022_64
cl /EHsc /std:c++17 /O2 ^
   app_unity.cpp ^
   /I"%QT%\include" /I"%QT%\include\QtCore" /I"%QT%\include\QtWidgets" ^
   /I"%QT%\include\QtOpenGL" /I"%QT%\include\QtCharts" ^
   /I"..\..\src\pt_shared" /I"..\..\src\native" ^
   /Fe"..\..\build\app\calimerge.exe" ^
   /link /LIBPATH:"%QT%\lib" ^
   Qt6Core.lib Qt6Widgets.lib Qt6OpenGLWidgets.lib Qt6Charts.lib ^
   ..\..\build\native\calimerge.lib ^
   ..\..\build\cuda\calimerge_cuda.lib
```

```cpp
// app_unity.cpp — the single compilation unit
#include "main.cpp"
#include "MainWindow.cpp"
#include "StateManager.cpp"
#include "tabs/CamerasTab.cpp"
#include "tabs/IntrinsicTab.cpp"
#include "tabs/ExtrinsicTab.cpp"
#include "tabs/ProcessTab.cpp"
#include "tabs/WorkoutPage.cpp"
#include "widgets/CameraGrid.cpp"
#include "widgets/SkeletonView.cpp"
#include "widgets/VideoPlayer.cpp"
#include "workers/CameraWorkers.cpp"
#include "workers/CalibrationWorkers.cpp"
#include "workers/OfflineWorker.cpp"
// MOC output — generated by Step 1
#include "gen/moc_MainWindow.cpp"
#include "gen/moc_StateManager.cpp"
// ... etc
#include "gen/resources.cpp"
```

This is exactly the same pattern as `build_win32.bat` running `cl test_usb_serials.cpp`. The build step that "requires CMake" is actually three lines in a `.bat` file.

### Per-component script layout

```
src/
├── native/
│   ├── build_win32.bat         # already exists
│   └── build_macos.sh          # already exists
├── pt_shared/
│   ├── build_win32.bat         # already exists (tracker, triangulation)
│   └── build_macos.sh
├── cuda_pipeline/
│   └── build_cuda_win32.bat    # already exists
├── mps_pipeline/
│   └── build_mps.sh            # already exists
├── calibration/                # NEW — OpenCV + Ceres
│   ├── build_win32.bat
│   └── build_macos.sh
└── app/                        # NEW — Qt 6 GUI
    ├── build_win32.bat
    ├── build_macos.sh
    ├── gen/                    # MOC + RCC output (gitignored)
    └── app_unity.cpp           # unity root
```

Top-level `build.sh` (already exists) dispatches to per-component scripts in dependency order. Parallelism where inputs are independent (`native` and `pt_shared` have no cross-dependency).

### Ceres without CMake

Ceres ships prebuilt on both platforms via package managers that output plain include/lib paths:
- macOS: `brew install ceres-solver` → `/usr/local/include/ceres`, `/usr/local/lib/libceres.a`
- Windows: vcpkg `x64-windows` triplet → `%VCPKG_ROOT%\installed\x64-windows\{include,lib}`

Both drop into `cl /I... /link ...` or `clang++ -I... -l...` directly. No `FindCeres.cmake` needed.

### Distribution

- macOS: `build_app_macos.sh` produces `Calimerge.app` bundle. `codesign` + `xcrun notarytool` invoked directly in the script. `hdiutil` creates `.dmg`. Models in `Contents/Resources/models/` or downloaded at first launch via `NSURLSession`.
- Windows: `build_app_win32.bat` produces `calimerge.exe`. Installer: NSIS script (a `.nsi` file, not a build system). Code-signed with `signtool.exe`.
- Linux: lower priority, not blocking.

---

## 6. Migration sequencing

The Python GUI stays live and usable at every phase. The C++ app grows in `src/app/` in parallel. Cutover happens at Phase 3 completion — until then, `run_mac.sh` / `run_win.sh` still launch the Python app.

### Phase 0 (done / days): kill the conversion fragility
1. Pre-build CoreML mlpackages on the dev machine. Commit as release artifacts.
2. `run_mac.sh` downloads pre-built mlpackages instead of running `build_mac_models.sh`.
3. `coremltools` / `onnx2torch` moved to a `dev-models/` requirements file, not in `uv.lock`.

**Status: complete.** Today's specific pain (CoreML conversion at launch time) is gone.

### Phase 1 (1–2 weeks): consolidate the C++ pipeline
4. Move `track_stitch.py` (210 lines) into `pt_tracker.cpp`. Verify against the headless reproducer.
5. Drop the runtime PyTorch pose backend (`tracking/pose_detector.py`). Remove `torch`, `transformers`, `ultralytics` from `pyproject.toml`. The TensorRT/CoreML pipelines are the live path.
6. Python GUI continues to drive everything. No user-visible change. `uv.lock` shrinks dramatically.

### Phase 2 (4–6 weeks): C++ calibration pipeline
7. Port `triangulation.py` → call `pt_triangulation.cpp` directly. Delete Python file.
8. Port `calibration/charuco.py`, `calibration/intrinsic.py` → `src/calibration/`. Build with `src/calibration/build_win32.bat` / `build_macos.sh`. Expose to Python GUI via ctypes (same pattern as camera_binding.py) during transition.
9. Port `calibration/extrinsic.py` → C++ + Ceres. **Hardest single piece.** Two-week budget. Validate against `tests/test_extrinsic_real.py`; accept new RMSE baseline.

After Phase 2: inference + calibration are 100% C++. Python is GUI + analysis only. Dependency list shrinks to `numpy, opencv-python, PySide6, scipy, pandas, matplotlib`.

### Phase 3 (8–12 weeks): C++ GUI in `src/app/` — confirmed in scope
The new app lives in `src/app/` alongside the Python source. Both are buildable simultaneously during development. See §9 for the directory structure.

10. Scaffold: `src/app/`, `app_unity.cpp`, `build_win32.bat`, `build_macos.sh`, `gen/` (gitignored). `main.cpp` opens a blank `QMainWindow`. Proves the build chain before a line of real GUI is written.
11. Port `gui/main.py` + `gui/state.py` → `MainWindow.cpp` + `StateManager.cpp` + `AppState.h`. The state struct is a mechanical port of the Python frozen dataclasses.
12. Port tabs in dependency order: `cameras_tab` → `intrinsic_tab` → `extrinsic_tab` → `process_tab` (calibration tabs are smaller and validate the camera binding before WorkoutPage).
13. Port `gui/workers.py` QThread subclasses in parallel with tabs — each tab needs its workers.
14. Port `workout_page.py` (4550 lines). Largest chunk; 3–4 weeks. The state machine (login → camera init → calibration check → live pose → recording → results) maps 1:1 to the Python logic. No redesign needed.
15. Port `unified_offline_worker.py` (753 lines) → `OfflineWorker.cpp`.
16. `skeleton_view.py` → `SkeletonView.cpp` (QOpenGLWidget). Orbit camera, axis grid, bone-line rendering, color cycling. See §7 R3 for the time budget.
17. Embed CPython 3.11 for analysis modules (`src/calimerge/analysis/*.py`). The embedded interpreter is invoked via pybind11 with the narrow API: `analysis::run(npz_path, analyzer_id, params_json) -> result_json`. The C++ side never imports torch/transformers/ultralytics.

### Phase 4 (1 week): cutover and packaging
18. `run_win.sh` / `run_mac.sh` now launch the C++ binary instead of `uv run calimerge gui`.
19. `pyproject.toml` survives as analysis-only: `uv run python notebooks/...` still works.
20. Code signing + notarization wired into `build_app_macos.sh`. NSIS installer for Windows.
21. `src/calimerge/` is not deleted — it remains as the analysis + calibration-tuning surface.

**Total:** 14–20 weeks. Phase 2's bundle adjustment is the most likely schedule risk (estimate 2 weeks, could be 4). Phase 3's WorkoutPage port is the largest single chunk but is mechanical, not algorithmically difficult.

---

## 7. Risk register (no sugar)

### R1 — Bundle adjustment regression
SciPy `least_squares` (TRF) → Ceres won't reproduce RMSE bit-exactly. **Mitigation:** freeze a regression set of current calibrations before porting; treat new RMSE as the new baseline; validate with the zelda fixture.

### R2 — GUI iteration speed regression (accepted)
Edit-save-rerun on PySide6 is 1–2 s. Edit-rebuild-rerun on Qt C++ is 10–30 s for a small change, 60+ s for a header touch. This is a real daily cost. **Accepted.** Mitigations: (a) the unity build keeps incremental compiles fast for `.cpp`-only changes — a header touch is the slow case, not the common case; (b) calibration tuning stays Python-callable via the embedded interpreter; (c) during Phase 3 development, the Python GUI stays buildable as a reference.

### R3 — Skeleton 3D view (`skeleton_view.py`)
pyqtgraph's GLViewWidget is doing real work: orbit camera, axis grid, bone-line rendering, color cycling. Re-implementing in `QOpenGLWidget` is finite but not free. **Estimate: 2 weeks. Do not use Qt3D** — it adds a 30 MB dependency for a use case that needs ~300 lines of GL. Hand-rolled QOpenGLWidget is the correct scope here, same as `src/native/` hand-rolling the camera API.

### R4 — Notebook and npz compatibility (hard constraint)
The C++ rewrite must preserve `keypoints_3d.csv` / `keypoints_3d.raw.npz` byte-compatibility. The `keypoint_export.py` and `keypoints_io.py` write paths define the contract; the C++ port must match them field-for-field including `view_transform_R`, `view_transform_t`, `model_backend`, `model_name`, `person_confidence`, `max_track_distance`, `track_patience`. Any deviation silently breaks every notebook. **Write a byte-comparison test before deleting the Python writers.**

### R5 — Build script maintenance (reduced vs. CMake)
Without CMake, each new source file that contains `Q_OBJECT` needs a MOC invocation added to the build script manually. This is low-friction (one line per file) but requires discipline. **Mitigation:** `gen/` is gitignored and regenerated on every build — if you forget the MOC line, you get a clear linker error, not a silent misbehavior. Worse failure modes come from CMake's own abstraction leaks (wrong generator, policy warnings, find_package version mismatches). The scripts are lower total risk.

### R6 — Single-developer bus factor (accepted)
~22k LOC of new C++ is harder to maintain solo than Python. **Accepted with eyes open.** The embedded Python interpreter preserves the analysis surface where Python is genuinely better. The C++ surface is the GUI and the ML pipeline — two areas where the existing C++ codebase already demonstrates the author can handle the complexity.

### R7 — Python isn't the bottleneck on hot paths (irrelevant to GUI rewrite)
True — but the GUI rewrite isn't motivated by hot-path performance. It's motivated by distribution (single binary, no Python runtime), cold start, and eliminating the ctypes binding maintenance surface. R7 was a valid argument against the rewrite when the verdict was "don't do Phase 3." That verdict has changed.

### R8 — ML toolchain fragility migrates, not disappears
TensorRT version sensitivity, CoreML ABI coupling, CUDA driver pairing — these are real and remain. The category shifts from "PyPI dep conflict at import time" to "binary won't launch on a machine with the wrong TRT version." **Mitigation:** the app does not do model conversion at runtime (Phase 0 fixed this). Pre-built engines are cached; version mismatch produces a clear error at engine-load time, not a silent wrong-answer situation.

### R9 — Qt LGPL compliance
Qt 6 under LGPL requires dynamic linking for closed-source apps, or a commercial license. calimerge is BSD-2 (`pyproject.toml:5`), so LGPL dynamic linking is fine. **Action required:** ship Qt DLLs/dylibs alongside the binary and include the LGPL notice in About. Do not statically link Qt without a commercial license. This is a one-time compliance task, not an ongoing cost.

---

## 8. Verdict

**Full rewrite: Phases 0 through 4. GUI rewrite is in scope and confirmed.**

The earlier recommendation to stop at Phase 2 was made before the build system question was resolved. CMake was the blocker — it introduced more complexity than the GUI port would remove. With unity scripts (§5), that blocker is gone. The per-line cost of the C++ GUI is the same as the C++ pipeline code already shipping.

### What the full rewrite actually achieves

- **Single binary, no Python runtime at distribution.** `.dmg` / `.exe` users install and run. No `uv`, no venv, no conda collision, no `VIRTUAL_ENV=` prefix. This is a qualitative improvement in distributability.
- **Cold start under 200 ms.** The PySide6 interpreter startup (~1.5–3 s) disappears.
- **The ctypes boundary is gone.** `cuda_binding.py` (397 lines), `mps_offline_binding.py` + `mps_binding.py` + `mps_stream_binding.py` (1179 combined), `camera_binding.py` (531) — four files whose entire purpose is to paper over the Python/C++ split. They become dead code.
- **One keypoint-count definition.** `pt_common.h:34` and `types.py` currently can drift (and have). One header, one truth.
- **The analysis surface stays Python.** The embedded CPython interpreter (§3) is not a compromise — it is the correct architecture. `analysis/*.py`, the notebooks, and the perturbation study are exactly the kind of code that should be Python. Keeping them Python is not a failure of the rewrite; it's the design.

### What the rewrite does not fix

- ML toolchain fragility (R8). TensorRT/CoreML version coupling survives the rewrite.
- The analysis iteration loop. That stays Python by design.
- Notebook byte-compatibility (R4). That must be explicitly maintained.

### The honest risk

R2 (iteration speed) and R6 (bus factor) are real. They are accepted. The mitigations are: unity builds keep incremental compiles tolerable; the Python GUI coexists through Phase 3; the embedded interpreter preserves the fast-iteration surface where it matters most (calibration math, data analysis). The author has already demonstrated they can maintain `src/cuda_pipeline/` (1330 lines) and `src/mps_pipeline/` (1174 lines) — the Qt GUI at ~8000 lines is more code but not more complex per-line.

### Confirmed plan

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Kill CoreML conversion at launch | Done |
| 1 | Drop PyTorch backend, move track_stitch to C++ | Next |
| 2 | C++ calibration pipeline (OpenCV + Ceres) | Planned |
| 3 | C++ GUI in `src/app/` — parallel, then cutover | Confirmed |
| 4 | Packaging + cutover | Follows Phase 3 |

---

## 9. Parallel folder structure (`src/app/`)

The C++ GUI lives in `src/app/` from the first commit. The Python app in `src/calimerge/` is not touched during Phase 3 development — both are buildable and runnable simultaneously. Cutover happens at Phase 4 when `run_mac.sh` / `run_win.sh` point at the C++ binary.

### Directory layout

```
src/app/
├── build_win32.bat         # MSVC unity build + MOC + RCC
├── build_macos.sh          # clang++ unity build + MOC + RCC
├── app_unity.cpp           # #includes every .cpp in the project
├── gen/                    # MOC + RCC output — gitignored
│
├── main.cpp                # main(), QApplication, MainWindow launch
├── AppState.h              # Plain struct — port of types.py frozen dataclasses
├── MainWindow.h/.cpp       # QMainWindow, tab bar, menu bar
├── StateManager.h/.cpp     # QObject coordinator — port of gui/state.py
│
├── tabs/
│   ├── CamerasTab.h/.cpp       # Tab 1: record, preview, FPS graph
│   ├── IntrinsicTab.h/.cpp     # Tab 2: per-camera calibration
│   ├── ExtrinsicTab.h/.cpp     # Tab 3: multi-camera bundle adjustment
│   ├── ProcessTab.h/.cpp       # Tab 4: tracking + triangulation
│   └── WorkoutPage.h/.cpp      # Main workout state machine (largest file)
│
├── widgets/
│   ├── CameraGrid.h/.cpp       # Multi-camera display grid
│   ├── SkeletonView.h/.cpp     # QOpenGLWidget 3D pose view
│   ├── VideoPlayer.h/.cpp      # Scrubber + playback
│   └── ProgressGraph.h/.cpp    # Qt Charts live FPS line plot
│
├── workers/
│   ├── CameraWorkers.h/.cpp    # Enumerate, Preview, Recording QThread workers
│   ├── CalibrationWorkers.h/.cpp # Intrinsic, Extrinsic workers
│   └── OfflineWorker.h/.cpp    # Unified offline pipeline worker
│
└── analysis/
    └── AnalysisBridge.h/.cpp   # pybind11 bridge to src/calimerge/analysis/*.py
```

### Naming conventions

Headers use PascalCase matching the Python class names exactly (`CamerasTab.h` ↔ `cameras_tab.py`). This makes the port auditable: open both files side by side. No renames unless the Python name was itself confusing.

### What does NOT go in `src/app/`

- Calibration math (`src/calibration/`) — separate component, separate build script. `src/app/` links the compiled output.
- Camera capture (`src/native/`) — already a compiled component.
- Inference pipelines (`src/cuda_pipeline/`, `src/mps_pipeline/`) — same.
- Analysis Python scripts (`src/calimerge/analysis/`) — stay Python, invoked via `AnalysisBridge`.

### `app_unity.cpp` discipline

Every `.cpp` file in `src/app/` is `#include`d exactly once in `app_unity.cpp`. Headers are `#pragma once`. No circular includes. The build fails loudly (duplicate symbol) if a `.cpp` is included twice. This is the Casey Muratori guarantee: the build output is deterministic and the include graph is explicit.

### MOC discipline

Every class that uses `Q_OBJECT`, `Q_SIGNALS`, `Q_SLOTS`, or any Qt metaclass feature goes in a `.h` file (not inline in a `.cpp`). The build script runs `moc` on every `.h` file in the project unconditionally — the overhead is negligible and it avoids the "forgot to add to MOC list" bug. Generated output lands in `gen/moc_ClassName.cpp` and is included at the bottom of `app_unity.cpp` after all implementation files.

---

## Appendix A — Field evidence: parameter drift in the current Python orchestrator (May 8 2026)

This appendix is field evidence, not speculation, gathered while debugging
why the MPS backend produces different track output than the PyTorch
backend on the same recording. It strongly **validates the rewrite's
sequencing** (Phase 1 first) without itself being a rewrite-blocker.

**What we found.** Three independent tracker implementations, each with
different default values, and the user-supplied tracker parameters were
silently dropped on the floor for two of three backends:

- `_LiveTracker` (Python, `src/calimerge/gui/workers.py:543`) — used by the live PyTorch path. Defaults: `max_match_distance=0.5`, `patience=10`.
- `PersonTrack` (Python, `src/calimerge/tracking/tracker.py`) — full Python tracker, legacy.
- `pt_tracker` (C++, `src/pt_shared/pt_tracker.cpp`) — used by CUDA + MPS pipelines. Until the fix below, both bindings hard-coded `max_track_distance=0.15` and `track_patience=30` regardless of caller intent (`mps_stream_binding.py:288-292`, `cuda_stream_binding.py:288-289`).

The unified offline worker (`src/calimerge/gui/unified_offline_worker.py`)
accepted `max_track_distance` and `track_patience` as constructor args
but threaded them only into stitching — never into the actual trackers.
PyTorch silently used `_LiveTracker(patience=10)`; MPS/CUDA silently used
the C config defaults; the GUI sliders had no effect on either.

**The fix that landed (Phase 1 of this design's recommendation, partial).**

1. Plumbed `max_track_distance` + `track_patience` through both ctypes binding constructors and into the C config struct.
2. Plumbed the same params through the unified offline worker so all three backends now see identical numerical values.
3. Added a *canonical re-tracker* pass: after each backend produces per-frame `persons` lists, the unified worker discards each backend's track ids and re-runs a single Python `_LiveTracker` over the recording. Track id assignment is now byte-identical for all three backends; only the inputs (kps_3d) still differ.
4. Embedded `person_confidence`, `max_track_distance`, `track_patience` into the saved npz so the regression-test harness (`tests/manual/compare_backends_to_baseline.py`) refuses to compare runs whose params disagree.

**What this proved.** Once params and tracker code are equalised, the
per-track Hip-COM diff between PyTorch and MPS on the primary subject is
~35 cm (mean) / 77 cm (max). The remaining divergence is at the
**inference layer** (FP16 CoreML vs FP32 PyTorch), not the tracking
layer — exactly what the rewrite would *not* fix on its own. Notably,
the MPS pipeline emits a **static ghost track** (a 61-frame run where
all 61 Hip-COMs are byte-identical at z=-6.75 m, i.e. behind the camera)
that the canonical retracker can't merge away because the geometry
honestly differs from the real subject's track.

**Implications for the rewrite plan.**

- The drift problem is *not* a "Python is too dynamic" problem — it's a "three tracker implementations, none authoritative, parameters dropped between layers" problem. Either language would suffer this if no one had cleaned it up. The fix that landed is structural (single retracker pass), not language-driven.
- §7 R6 (single-developer bus factor): we found this drift only because of an explicit regression-test harness someone bothered to write today. In a much larger C++ codebase, the same drift can hide for longer because iteration cost suppresses exploratory diff-testing. Phase 2's *first* deliverable should be a parameter-flow-through audit of every binding/wrapper that survives the rewrite — same scrutiny we just applied to the trackers.
- §3's "embedded CPython for analysis only" is reinforced: the canonical retracker is now a 12-line Python pass over a Python list of dicts. Rewriting it as C++ to match the C tracker would *re-introduce* the original drift. Keep this pass in Python forever.
- §4's "drop the PyTorch backend post-Phase-1" still holds: once that's gone, the comparator becomes "MPS today vs MPS yesterday", not cross-backend, and the tracker-implementation-fragmentation question simplifies to one tracker.

**What's still load-bearing post-fix.** The static-ghost detection at
z=-6.75 m is a C-pipeline output bug (likely tracker extrapolation when
inference momentarily drops, or a detection on a fixed background
object). Worth tracking down before MPS replaces PyTorch as the live
default — but it's a detection/triangulation issue, not a tracker config
issue. File a separate bug.

---

## Appendix B — File:line evidence for claims

- C++ pipelines are already production: `src/cuda_pipeline/pt_pipeline.cpp` (1330), `src/mps_pipeline/pt_offline_mps.m` (650), `src/mps_pipeline/pt_stream_mps.m` (524).
- Bundle adjustment is the SciPy hotspot: `src/calimerge/calibration/extrinsic.py:14` imports `scipy.optimize.least_squares`; `:15` imports `scipy.sparse.lil_matrix`.
- Triangulation is already near-C: `src/calimerge/triangulation.py:11` imports `numba.jit`; `pt_shared/pt_triangulation.cpp` is the C++ equivalent (378 lines).
- Coremltools fragility is `--with` quarantined: `build_mac_models.sh:67-70`.
- Largest Python file is the GUI workout page: `src/calimerge/gui/workout_page.py` — 4550 lines.
- Largest Python module overall is config: `src/calimerge/config.py` — 1903 lines (TOML + SQLite + filesystem layout).
- Headless reproducer for offline pipeline regression testing: `tests/manual/run_offline_pipeline_on_test_data.py` (CLAUDE.md:24-29).
- Ctypes binding boundaries are the fragility surface: `src/calimerge/tracking/cuda_binding.py:268-271`, `src/calimerge/tracking/mps_*_binding.py` (4 files).
- The FP16 abstraction-leak that motivated DESIGN.md: `src/cuda_pipeline/pt_tensorrt.cpp:303-332`.
- Hard-coded keypoint count that DESIGN.md §1 wants to remove: `src/pt_shared/pt_common.h:34` (also in `src/cuda_pipeline/pt_common.h`).
- Models live outside the repo by design: CLAUDE.md:48-69. This is what makes "ship pre-built artifacts" feasible.
