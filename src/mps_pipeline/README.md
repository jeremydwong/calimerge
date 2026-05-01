# MPS Pose Tracking Pipeline (macOS Apple Silicon)

CoreML / Apple Neural Engine + GPU counterpart to the Windows CUDA TensorRT
pipeline (`src/cuda_pipeline/`).

The C ABI mirrors the CUDA path so the Python ctypes binding and the GUI
worker can swap backends just by switching the loaded library. Two entry
points are shipped:

* **Streaming** (`pt_stream_mps`) — per-frame, drives live cameras.
* **Offline / batch** (`pt_offline_mps`) — replays recorded `port_*.mp4`
  files through the same streaming core. Offline ≡ online + (decoded
  video frame source) + (optional larger CoreML batch). The matcher,
  triangulator, and tracker are byte-identical between paths.

| Layer | File |
|---|---|
| C ABI header (streaming) | `pt_stream_mps.h` |
| C ABI header (offline) | `pt_offline_mps.h` |
| Streaming pipeline | `pt_stream_mps.m` |
| Offline pipeline | `pt_offline_mps.m` |
| CoreML inference wrapper | `pt_coreml.{h,m}` |
| vImage / Accelerate preprocessing | `pt_preprocess.{h,m}` |
| Heatmap decode | `pt_heatmap.{h,c}` |
| Calibration TOML loader | `pt_calibration.{h,c}` |
| AVFoundation video decode | `pt_videodecode.{h,m}` |
| Streaming CLI test | `pt_stream_main_mps.m` |
| Offline CLI test | `pt_main_mps.m` |
| Build script | `build_mps.sh` |

Cross-frame tracking, matching, triangulation, and CSV export are reused
from `src/pt_shared/` — they're platform-independent and shipped against
both backends.

---

## Building (on macOS)

```bash
# 1. Build the dylib + both test harnesses
bash src/mps_pipeline/build_mps.sh release

# Output:
#   build/mps/libcalimerge_mps.dylib
#   build/mps/pt_stream_main_mps    (live / streaming smoke test)
#   build/mps/pt_main_mps           (offline / batch smoke test)
```

The Python bindings (`src/calimerge/tracking/mps_stream_binding.py` and
`src/calimerge/tracking/mps_offline_binding.py`) both search
`build/mps/libcalimerge_mps.dylib` first, then `src/mps_pipeline/`, then
the package directory. They share the same dylib — different ABI groups,
one shared object.

---

## CoreML model conversion

The pipeline reads `.mlpackage` files. Convert from the same ONNX models the
CUDA path uses:

```bash
# Install coremltools (one-time, macOS-only)
pip install coremltools

# Run from repo root
python tests/manual/build_coreml_models.py

# Outputs (under repo/models/coreml/):
#   yolo_v10s.mlpackage          -> ANE preferred (cpuAndNeuralEngine)
#   vitpose_synthpose.mlpackage  -> GPU preferred (cpuAndGPU)
```

CoreML compiles each `.mlpackage` to a `.mlmodelc` on first use and caches
it next to the package; the first GUI launch with `Hardware (MPS)` selected
will take a few extra seconds while that compile happens. Subsequent runs
hit the cache.

---

## Running the GUI with MPS

After both the dylib and the `.mlpackage` files are present, the GUI
backend dropdown grows a `Hardware (MPS)` entry next to `PyTorch` and
(on the Windows machine) `Hardware (CUDA)`. The selector logic lives in
`src/calimerge/gui/workout_page.py::_init_ui` and dispatches via
`_start_mps_detection`.

If the entry is missing:

1. Verify `build/mps/libcalimerge_mps.dylib` exists.
2. Verify both `.mlpackage` files are under `models/coreml/` (or
   `<data_dir>/models/coreml/`).
3. From a Python REPL:
   ```python
   from calimerge.tracking.mps_stream_binding import is_available
   print(is_available())   # expected: True
   ```
   On non-Darwin platforms `is_available()` is hard-coded to return False.

---

## Compute-unit choice (why YOLO -> ANE, VitPose -> GPU)

Empirically:

- **YOLOv10s** is a pure-conv detector. The ANE was built for this kind of
  workload — low latency, dedicated silicon, low CPU contention.
- **VitPose** is a transformer (multi-head self-attention + MLP). On the
  M-series, the ANE struggles with attention layouts and the GPU often
  ends up faster end-to-end despite the higher per-op latency.

The defaults in `convert_onnx_to_coreml.py` reflect this. Override at
conversion time if your numbers say otherwise.

---

## CLI smoke tests

### Streaming (no cameras attached)

```bash
build/mps/pt_stream_main_mps \
    --calibration recordings/<session>/calibration.toml \
    --yolo models/coreml/yolo_v10s.mlpackage \
    --vitpose models/coreml/vitpose_synthpose.mlpackage \
    --num-cameras 3 --width 640 --height 480
```

That confirms model loading, calibration parsing, and pipeline create /
destroy work without needing a live camera tree.

### Offline / batch (replays recorded videos)

```bash
build/mps/pt_main_mps \
    recordings/<session>/ \
    recordings/<session>/calibration.toml \
    --yolo models/coreml/yolo_v10s.mlpackage \
    --vitpose models/coreml/vitpose_synthpose.mlpackage \
    --batch-size 1
```

Outputs `recordings/<session>/tracking_output/output_3d_poses_tracked.csv_personN.csv`
— the same filename schema the CUDA path uses, so the GUI's
`OfflineProcessingWorker._convert_outputs` parses both backends with no
branching. Increase `--batch-size` (1, 2, 4, 8) once the smoke test passes
to overlap AVAssetReader IO with CoreML compute.

### Python entry point (matches the GUI's offline path)

```python
from pathlib import Path
from calimerge.tracking.mps_offline_binding import run_mps_pipeline, is_available

assert is_available()  # only true on Mac with libcalimerge_mps.dylib built

run_mps_pipeline(
    video_paths={0: Path("port_0.mp4"), 1: Path("port_1.mp4"), 2: Path("port_2.mp4")},
    calibration_toml=Path("calibration.toml"),
    frame_time_csv=Path("frame_time_history.csv"),
    output_path=Path("tracking_output"),
    yolo_coreml=Path("models/coreml/yolo_v10s.mlpackage"),
    vitpose_coreml=Path("models/coreml/vitpose_synthpose.mlpackage"),
    batch_size=1,
)
```

Function signature is identical to `cuda_binding.run_cuda_pipeline` modulo
the `*_onnx` -> `*_coreml` parameter rename. The GUI's
`OfflineProcessingWorker._pick_offline_backend()` picks between them
automatically — `mps` on Darwin when the dylib loads, `cuda` on
Windows/Linux when the DLL loads, `none` otherwise.

---

## Things that still need to happen on the Mac

(everything below requires actual macOS hardware; nothing here can run on
Windows.)

- [ ] `bash src/mps_pipeline/build_mps.sh release`
- [ ] `pip install coremltools` in the venv
- [ ] `python tests/manual/build_coreml_models.py`
- [ ] Sanity-check `is_available()` returns True from a Python REPL
      (test BOTH `mps_stream_binding.is_available()` and
       `mps_offline_binding.is_available()` — they share a dylib).
- [ ] Streaming smoke test: `build/mps/pt_stream_main_mps ...`
- [ ] Offline smoke test: `build/mps/pt_main_mps recordings/<session>/ ...`
- [ ] Launch the GUI and pick `Hardware (MPS)` from the backend dropdown
- [ ] Visually confirm live skeletons appear and FPS is reasonable
      (~30 fps on M2 Pro is the rough target; CUDA hits ~100 fps)
- [ ] Record a session with `Pause tracking during recording` ON and
      `Generate CSV after save` ON, then verify
      `OfflineProcessingWorker._pick_offline_backend()` returns `"mps"` and
      the per-track CSVs land under the recording dir.
