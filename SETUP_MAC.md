# macOS Setup Guide

Step-by-step for getting calimerge running on a fresh Mac (Apple Silicon).

## Prerequisites

- **Xcode Command Line Tools**: `xcode-select --install`
- **Homebrew**: https://brew.sh
- **uv** (Python package manager): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Git**: comes with Xcode CLT

## 1. Clone the repo

```bash
git clone git@github.com:jeremydwong/calimerge.git
cd calimerge
```

Or if the repo already exists:

```bash
cd ~/Git/calimerge
git pull
```

## 2. Install Python dependencies

```bash
~/.local/bin/uv sync
```

This creates a `.venv/` and installs everything from `pyproject.toml`.

## 3. Copy data directory from an existing Mac

The data directory lives at `~/Library/Application Support/Calimerge/`.
Copy the whole folder from your source Mac:

```bash
mkdir -p ~/Library/Application\ Support/Calimerge
# From source Mac (via AirDrop, USB drive, scp, etc.):
cp -R /Volumes/source/Calimerge/* ~/Library/Application\ Support/Calimerge/
```

What's in here and what you need:

| File/Dir | Required | Notes |
|---|---|---|
| `intrinsics.db` | **Yes** | Per-camera lens calibrations |
| `extrinsics.db` | **Yes** | Multi-camera spatial calibrations |
| `view_transforms.db` | **Yes** | Rotate-to-human transforms |
| `app_settings.json` | No | Will be recreated on first launch |
| `workouts.db` | No | Session history — recreated if missing |
| `workout_spec.db` | No | Exercise specs — recreated if missing |
| `models/` | See below | ML model files |

### Models sub-directory

| Dir | Size | Required for | Portable across Macs? |
|---|---|---|---|
| `models/onnx/` | ~700 MB | Offline pipeline | Yes |
| `models/coreml/` | ~360 MB | MPS (live + offline) | **Yes** — CoreML compiles to device at load time |
| `models/yolo/` | ~31 MB | ONNX export (source weights) | Yes |
| `models/vitpose/` | ~340 MB | PyTorch backend (HuggingFace snapshot) | Yes |

All model files are architecture-independent. `.mlpackage` files built on
an M1 work on M4 and vice versa — CoreML handles hardware-specific
compilation at first inference.

**If you have the models directory from another Mac, just copy it.**
If not, build them from scratch (step 4).

## 4. Build models (only if you don't have them)

Skip this step if you copied `models/` from another Mac.

```bash
bash build_mac_models.sh
```

This runs two stages:
1. **PyTorch → ONNX** (~1-2 min, downloads model weights on first run)
2. **ONNX → CoreML** (~5-15 min, CPU-intensive)

Output goes directly to `~/Library/Application Support/Calimerge/models/`.

## 5. Build native libraries

```bash
bash run_mac.sh --no-build   # just to verify uv works (ctrl-C after launch)
bash build.sh release        # camera capture library (libcalimerge.dylib)
bash src/mps_pipeline/build_mps.sh release   # MPS pose pipeline (libcalimerge_mps.dylib)
```

Or let `run_mac.sh` handle both builds automatically:

```bash
bash run_mac.sh
```

`run_mac.sh` rebuilds both dylibs when source is newer than the built
artifact, runs `uv sync`, and launches the GUI.

## 6. Set up the workout directory

On first launch, the GUI creates `~/Documents/Calimerge/` as the default
workout directory. If you want recordings from another machine, copy the
workout folder there:

```bash
cp -R /Volumes/source/Documents/Calimerge ~/Documents/Calimerge
```

Or change the workout directory via **File → Workout Directory…** in the GUI.

## 7. Launch

```bash
bash run_mac.sh
```

This is the standard launch command. It:
1. Rebuilds native dylibs if source changed
2. Runs `uv sync`
3. Launches `uv run calimerge gui`

### Other launch modes

```bash
bash run_mac.sh clock        # sync verification clock
bash run_mac.sh --rebuild    # force-rebuild native libs
bash run_mac.sh --no-build   # skip native build
```

## 8. Verify the offline pipeline (optional)

Run the regression test against the zelda test fixture:

```bash
VIRTUAL_ENV= ~/.local/bin/uv run python tests/manual/run_offline_pipeline_on_test_data.py --max-syncs 50
```

Should complete in ~7s and report `post-stitch persons: 1`.

## Troubleshooting

### `uv` not found

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Or reinstall: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### MPS backend not working

Check that CoreML models exist:

```bash
ls ~/Library/Application\ Support/Calimerge/models/coreml/
# Should show: yolo_v10s.mlpackage  vitpose_synthpose.mlpackage
```

If missing, run `bash build_mac_models.sh`.

### Camera not detected

macOS requires camera permission for the terminal app you're using.
Go to **System Settings → Privacy & Security → Camera** and enable
your terminal (Terminal.app, iTerm2, etc.).

### conda interference

If you have conda/miniconda installed and see spurious output or wrong
Python, unset `VIRTUAL_ENV` before running uv:

```bash
VIRTUAL_ENV= ~/.local/bin/uv run calimerge gui
```
