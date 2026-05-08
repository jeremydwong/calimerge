# DESIGN_CPP.md — A C++ rewrite of calimerge: design, sequencing, and an honest verdict

> Status: design only. Not a commitment to ship. Read §8 first if you only have five minutes — the verdict is **partial rewrite, not a full one**, and the recommended first slice is small.

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

**Single CMake top-level**, sub-projects per existing component:

```
CMakeLists.txt                          # top-level
  cmake/                                # FindCeres.cmake, FindOpenCV.cmake, etc.
  src/native/CMakeLists.txt             # camera lib (already a unit)
  src/pt_shared/CMakeLists.txt          # tracker, triangulation, matching, export
  src/cuda_pipeline/CMakeLists.txt      # CUDA + TensorRT
  src/mps_pipeline/CMakeLists.txt       # CoreML + Accelerate
  src/calibration/CMakeLists.txt        # NEW — OpenCV + Ceres
  src/gui/CMakeLists.txt                # NEW — Qt 6
  src/app/CMakeLists.txt                # NEW — main(), embedded Python
```

Reasoning: the existing per-platform unity scripts (`build_macos.sh`, `build_win32.bat`, `build_cuda_win32.bat`, `build_mps.sh`) are great for the bottom layers but cannot orchestrate Qt's MOC + Ceres' transitive deps + per-platform code signing. CMake is the boring, working answer.

The unity scripts can stay as **fast inner-loop developer tools** for the lower layers — same pattern the user already loves: fast iteration on `.mm`/`.cu` files without going through CMake. The top-level CMake is for the GUI and the shipping binary.

**Distribution:**
- macOS: `.app` bundle code-signed with Developer ID, notarized, distributed as `.dmg`. Models bundled in `Contents/Resources/models/` or downloaded to `~/Library/Application Support/Calimerge/models/` on first run.
- Windows: MSIX or NSIS installer. Code-signed with EV cert (or self-signed for now). Models same idea.
- Linux: `.deb` + `.AppImage`. Lower priority — calimerge is not currently shipping Linux.

---

## 6. Migration sequencing

The project must remain usable throughout. Here's the order that keeps shipping ability intact at every step.

### Phase 0 (1–2 days): kill the conversion fragility *without* a rewrite
1. Build the CoreML mlpackages once on the user's machine. Commit them to a Git LFS bucket or a release artifact.
2. Modify `run_mac.sh` to **download** pre-built mlpackages instead of running `build_mac_models.sh`.
3. Move `coremltools` and `onnx2torch` into a separate `dev-models/` requirements file that's only used by the developer who builds artifacts.

This phase **eliminates today's pain** without touching any source. If §8's verdict ("don't full-rewrite") is taken, this is the win.

### Phase 1 (1–2 weeks): consolidate the C++ pipeline
4. Move `track_stitch.py` (210 lines) into `pt_tracker.cpp`. Verify against the headless reproducer (CLAUDE.md:25).
5. Replace the runtime PyTorch pose backend (`tracking/pose_detector.py`) with a CoreML/TensorRT-only path. Drop torch, transformers, ultralytics from `pyproject.toml`. Massive lockfile shrink.
6. The Python GUI continues to drive everything. No user-visible change.

### Phase 2 (4–6 weeks): C++ calibration pipeline
7. Port `triangulation.py` → use existing `pt_triangulation.cpp` directly. Delete Python file.
8. Port `calibration/charuco.py`, `calibration/intrinsic.py` → C++. Expose via the existing ctypes pattern.
9. Port `calibration/extrinsic.py` → C++ + Ceres. **This is the hardest single piece.** Two-week budget. Validate against `tests/test_extrinsic_real.py` fixture; accept new RMSE baseline.

After Phase 2: the inference + calibration pipelines are 100% C++. Python is GUI + analysis. The dependency list is `numpy, opencv-python, PySide6, scipy, pandas, matplotlib`. Lighter than today's by a factor of 5.

**Checkpoint.** At this point, ask: do we even need the GUI rewrite?

### Phase 3 (8–12 weeks, if the answer above is yes): GUI rewrite
10. New `src/app/main.cpp` with QApplication.
11. Port `gui/main.py` (320 lines) — trivial.
12. Port `gui/state.py` (216) → `AppState` struct + `StateManager` QObject.
13. Port tabs in order of dependency: `cameras_tab` → `intrinsic_tab` → `extrinsic_tab` → `process_tab`.
14. Port `workout_page.py` (4550 lines). Largest single chunk; spread over 3–4 weeks.
15. Port `unified_offline_worker.py` (753).
16. `skeleton_view.py` → QOpenGLWidget custom render.
17. Embed CPython for analysis modules.

### Phase 4 (2 weeks): packaging
18. CMake top-level. Code signing. Notarization. Installer.
19. Cut Python entry points; old `pyproject.toml` becomes a sibling for analysis-only scripts.

**Total:** 16–24 weeks of focused single-developer work, assuming Phase 2's bundle adjustment doesn't take 4 weeks instead of 2 (it might).

---

## 7. Risk register (no sugar)

### R1 — Bundle adjustment regression
SciPy `least_squares` (TRF) → Ceres won't reproduce RMSE bit-exactly. Anything downstream that has thresholds tuned to the old number will need re-tuning. **Mitigation:** capture a frozen regression set of the current calibrations before porting; treat new RMSE as the new baseline; run side-by-side for two weeks.

### R2 — GUI iteration speed regression
Edit-save-rerun on PySide6 is 1–2 s. Edit-rebuild-rerun on Qt C++ is 10–30 s for a small change, 60+ s for a header touch. For a single-developer hobby/research codebase, this is **a real morale tax.** It compounds: every "let me just try one thing" becomes a coffee break. **Mitigation:** keep the calibration tuning surface (the part that gets fiddled with most) Python-callable via the embedded interpreter for development; switch to fully native at ship time.

### R3 — Skeleton 3D view (`skeleton_view.py`)
pyqtgraph's GLViewWidget is doing real work here. Re-implementing in QOpenGLWidget is finite but not free — orbit camera, axis grid, bone-line rendering, point picking, color cycling. Estimate **2 weeks**. Could blow out to 4 if Qt3D is chosen instead.

### R4 — Notebook gap
The user's data analysis loop today: record → run pipeline → open `notebooks/test_output.ipynb` → poke at numpy arrays → tweak. The C++ rewrite must preserve `keypoints_3d.csv` / `keypoints_3d.raw.npz` byte-compatibility. Any change to those files breaks every notebook the user has. **This is the single most important compatibility constraint and the easiest one to forget about.**

### R5 — Build matrix coordination
Today the build paths are decoupled: macOS unity scripts know nothing about CUDA; Windows MSVC scripts know nothing about CoreML. A single CMake unifies them and forces every dev-machine to satisfy more constraints. **Mitigation:** keep the per-component scripts; add CMake as an additional, top-level target. A macOS dev never needs `nvcc`; a Windows dev never needs `xcrun`. CMake just orchestrates whatever's available.

### R6 — Single-developer bus factor
~22k LOC of new C++ is significantly harder to maintain solo than ~22k LOC of Python. Six months out, an obscure bug in `extrinsic_tab.cpp` is much more painful to track than the equivalent in `extrinsic_tab.py`. Even with multiple Claude instances helping, **the user is the integrator and reviewer of every PR**, and there is no second human reviewer. C++ amplifies the cost of mistakes.

### R7 — Python isn't the bottleneck on hot paths
The hot path (camera → inference → triangulation) is *already* C++. Python is doing orchestration. Rewriting orchestration in C++ buys ~2 ms/frame on a 33 ms budget. No user can perceive this.

### R8 — Coremltools-style fragility doesn't disappear; it migrates
The C++ path still has TensorRT version sensitivity (`pt_tensorrt.cpp:303-332`), CoreML version sensitivity, and CUDA driver coupling. Today these are masked by the bindings being tested cells. Tomorrow they're directly in the user's binary's dependency closure. The category of pain shifts from "PyPI dep conflict" to "TensorRT 10.5 vs 10.6 ABI break" — different, not gone.

---

## 8. Honest verdict

### Argument for the full rewrite
- One language, one build, one binary. Real architectural simplification.
- Cold start under 200 ms feels great.
- No more `VIRTUAL_ENV=` papercuts, no more `uv run --with`, no more conda-VS-uv collisions.
- The C++ pipelines already exist; this is "finish what's started," not "start from zero."
- Distribution becomes shippable (`.dmg` for users who don't have Python, ever).

### Argument against
- **Today's specific pain (CoreML conversion failing) is a Phase-0 fix, not a rewrite.** The mlpackages convert successfully when conversion succeeds; the runtime is fine. Pre-build them, ship them, done. That's two days of work, not six months.
- Phase 1 + Phase 2 (kill PyTorch backend, port calibration to Ceres) eliminate ~80% of the chronic Python pain at ~20% of the rewrite cost.
- The GUI rewrite (Phase 3) is the biggest chunk and the one with the worst ratio of effort to user-visible improvement. PySide6 is *fine*. The user's frustration is not with PySide6 — it's with `coremltools`.
- The analysis modules and notebooks should stay Python regardless. So you don't actually escape Python; you just reduce its footprint.
- A single developer maintaining 22k extra LOC of Qt C++ is taking on a real, ongoing cost, every week, forever.

### Recommendation: **partial rewrite. Phase 0 + Phase 1 + Phase 2. Do not do Phase 3.**

Concretely:

1. **This week (Phase 0):** Stop running `coremltools` on every model rebuild. Pre-build the mlpackages, commit/release them as artifacts, remove `--with coremltools` from the launch path. **This eliminates today's specific pain.** It is also the highest-leverage change in the entire document.

2. **This month (Phase 1):** Drop `torch`, `transformers`, `ultralytics` from runtime dependencies. The C++ TensorRT/CoreML pipelines are the live path; the PyTorch pose backend is a fallback nobody uses in production. Move `track_stitch.py` into `pt_tracker.cpp`. The `uv.lock` shrinks dramatically. macOS install gets faster. `fork()` issues go away (no more multiprocessing-related fork; coremltools is no longer in the lockfile).

3. **Next quarter (Phase 2):** Port the calibration pipeline to C++ + Ceres. This is where Python *is* the bottleneck — bundle adjustment in scipy is genuinely slow on 8-camera, 30-minute recordings, and the C++ port is a 5–10× speedup that the user will feel.

4. **Stop there.** Keep PySide6 for the GUI. Keep notebooks. Keep `analysis/`. Keep `perturbation_study.py`. They are not the problem and rewriting them is a multi-month detour with negative ROI.

The result after Phase 2: calimerge is a thin Python GUI orchestrating a fully-C++ inference + calibration pipeline. The Python dependency footprint is ~5 packages instead of ~16. Cold start is faster. The chronic costs go away. The acute cost (today's CoreML fight) is solved by Phase 0 alone.

That is the honest answer. The user's instinct is correct — Python *was* in the way today — but the answer is "remove Python from where it doesn't belong" not "remove Python." The latter is a six-month project that solves a problem the former solves in a week.

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
