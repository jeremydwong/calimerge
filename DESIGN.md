# DESIGN.md

Two design proposals for calimerge: 

1. pluggable pose **backends** including HuggingFace models, and 
2. a **workout/program abstraction** with JSON (de)serialization.

Scope is *design*, not implementation. Concrete file:line references point to the current code.

---

## 1. BACKENDS — Pluggable Pose Estimation

### 1.1 Current state

Two parallel implementations of the same model stack (YOLOv10 → VitPose/SynthPose-52):

| Path | Entry point | Runtime | Model source |
|---|---|---|---|
| Python/PyTorch | [src/calimerge/tracking/pose_detector.py](src/calimerge/tracking/pose_detector.py) | CPU/CUDA/MPS via `transformers` | HF `.from_pretrained` (`stanfordmimi/synthpose-vitpose-base-hf`) + local YOLO `.pt` |
| C++/CUDA | [src/cuda_pipeline/](src/cuda_pipeline/), [src/pt_shared/pt_common.h](src/pt_shared/pt_common.h) | TensorRT | ONNX files, path-passed via ctypes ([cuda_binding.py](src/calimerge/tracking/cuda_binding.py)) |

Selection is driven by a **string Literal** in [src/calimerge/types.py:195](src/calimerge/types.py#L195):

```python
pose_backend: Literal["charuco", "mediapipe", "vitpose"] = "charuco"
pose_device: str = "cpu"
```

Adding a new backend today requires: extend the Literal, branch in workers, rebuild C++ (keypoint count and input shapes are compile-time constants in [pt_common.h](src/pt_shared/pt_common.h)).

### 1.2 Hard-coded couplings that must dissolve

| Assumption | Where | Breaks with |
|---|---|---|
| `PT_NUM_KEYPOINTS = 52` | [pt_common.h:34](src/pt_shared/pt_common.h#L34) | Sapiens (308), WholeBody (133), RTMPose (17) |
| YOLO 640×640, VitPose 256×192 | [pt_common.h:42,46](src/pt_shared/pt_common.h) | RTMPose (384×288), Sapiens (384×384) |
| ImageNet mean/std | [pt_common.h:53-59](src/pt_shared/pt_common.h#L53) | Models with custom normalization |
| Hip COM at indices (11, 12) | [tracking/markers.py:26](src/calimerge/tracking/markers.py#L26), tracker.py | Any non-COCO ordering |
| FP16 I/O only safe for YOLO | [cuda_pipeline/pt_tensorrt.cpp:310-327](src/cuda_pipeline/pt_tensorrt.cpp) (fix from 2b141cc) | Every new model needs per-model FP16 audit |
| Model files resolved by name heuristics | [cuda_binding.py:268-271](src/calimerge/tracking/cuda_binding.py) | Any naming change |

The FP16 bug is the diagnostic symptom: the abstraction boundary currently sits **below** where model-specific behavior lives, so model-specific knowledge leaks into kernels. That is the thing to fix.

### 1.3 Proposed abstraction: `PoseModel` descriptor + registry

A single dataclass that fully describes a model's I/O contract. Loaders and kernels read from this instead of hard-coding.

```python
@dataclass(frozen=True, slots=True)
class KeypointSchema:
    name: str                       # "coco17", "synthpose52", "sapiens308"
    names: tuple[str, ...]          # ordered keypoint names
    hip_indices: tuple[int, int] | None  # for COM/tracker; None → use bbox center
    parents: tuple[int, ...]        # skeleton edges; drives GUI overlay

@dataclass(frozen=True, slots=True)
class PoseModelSpec:
    id: str                         # stable local id, e.g. "vitpose-synthpose-52"
    hf_repo: str | None             # optional HF repo for download
    detector: "PoseModelSpec | None"  # upstream person detector (top-down)
    input_shape: tuple[int, int]    # (H, W)
    normalization: tuple[tuple[float, ...], tuple[float, ...]]  # mean, std
    schema: KeypointSchema
    onnx_file: str                  # cached path under models/
    fp16_safe_io: bool              # from model metadata or override
    preprocess: Literal["letterbox", "crop_affine"]
    postprocess: Literal["heatmap_argmax", "simcc", "regression"]
```

A process-level registry (`MODEL_REGISTRY: dict[str, PoseModelSpec]`) is seeded with built-ins and can be extended by dropping a TOML/JSON file under `models/registry/`. `ProjectConfig.pose_backend` becomes `str` (a registry key), not a Literal.

### 1.4 Loader flow (Python + CUDA unified)

```
spec = MODEL_REGISTRY[project.pose_backend]
  → if spec.hf_repo: hf_hub_download(repo, filename) → cache under ~/.calimerge/models/
  → convert_to_onnx_if_missing(spec)  # scripts/export_*.py becomes data-driven
  → build_tensorrt_engine(spec) cached by (spec.id, gpu, precision)
  → return PyTorchBackend(spec) | TensorRTBackend(spec)
```

Backends implement a narrow Protocol, not an ABC:

```python
class PoseBackend(Protocol):
    spec: PoseModelSpec
    def infer(self, frames: np.ndarray, boxes: np.ndarray | None) -> np.ndarray: ...
      # returns (N, K, 3) float32: (x, y, score) in original image coords
```

On the C++ side, `PT_NUM_KEYPOINTS` becomes a runtime field on a `PoseEngine` struct; heatmap decoders accept it as a parameter; host-side allocations use `std::pmr` or simple `operator new[]`. Struct sizes in [pt_common.h](src/pt_shared/pt_common.h) move from fixed arrays to pointer+count. This is the only invasive change, but it's mechanical.

### 1.5 HuggingFace integration

Two levels:
1. **Built-in models**: registry entries reference `hf_repo`; first run does `hf_hub_download` + ONNX export, cached.
2. **User-added models**: `calimerge models add <hf-repo-id>` CLI writes a registry entry. For models where metadata is insufficient (custom preprocessing), we require a Python adapter file next to the TOML — this is the escape hatch that keeps the core clean.

`HF_TOKEN` read from env; anonymous fallback per HF rate limits.

### 1.6 Critical limitations to call out now

- **Top-down only.** The current architecture assumes a person detector feeds pose. Bottom-up models (e.g., OpenPose, DEKR) don't fit without a second preprocess/postprocess path. Register them as `detector=None, postprocess="paf"` and accept that they need a different C++ kernel.
- **Tracker assumes hips exist.** For schemas with no hip analogue, fall back to bbox centroid — encode this in `KeypointSchema.hip_indices = None`.
- **ONNX export is not always lossless.** SimCC-head models (RTMPose) need explicit opset ≥ 17; Sapiens has dynamic shapes that trip TensorRT. The registry should carry a `precision_notes` field the loader checks before building the engine.
- **Keypoint semantics across models don't align.** Triangulation and downstream analysis (see §2) reference joints by *name*, not index. A schema-aware mapper is needed for any cross-model comparison — designed in §1.7.

### 1.7 Keypoint naming: canonical vocabulary + alias resolution

Requiring every new model to publish keypoint names exactly matching ours is brittle. Some models use `LeftKnee`, some `l_knee`, some `knee_L`, some `kneeLeft`; Sapiens invents anatomical names we've never heard of. So the contract is two-layer:

**Layer 1 — canonical vocabulary.** calimerge owns a fixed list of joint names analyzers may reference (`left_knee`, `right_elbow`, `hip_center`, `left_ankle`, `c7`, `pelvis`, …). Analyzers only speak this vocabulary. Analysis code never sees native model names.

**Layer 2 — alias resolution per schema.** `KeypointSchema` gains an explicit mapping:

```python
@dataclass(frozen=True, slots=True)
class KeypointSchema:
    name: str
    names: tuple[str, ...]                       # native model names, ordered
    canonical_by_index: tuple[str | None, ...]   # per-keypoint canonical id, or None if no role
    # hip_indices etc. are derived from canonical_by_index, not stored separately
```

`canonical_by_index[i]` tells us "native keypoint `i` plays the canonical role `left_knee`." Unmapped kpts (`None`) are still captured and triangulated — we just don't promise analyzers can find them.

**Resolution pipeline (ordered, stop at first match).** Runs once at model registration, output is committed and reviewed:

1. **Normalize**: lowercase, strip separators (`LeftKnee`, `left-knee`, `left_knee`, `L_knee` → `leftknee`).
2. **Exact match** against canonical vocabulary or a curated synonym set (`pelvis↔hip_center`, `c7↔neck_base`).
3. **Side + part heuristic**: split into `{left, right, center, none}` × `{knee, elbow, ankle, wrist, …}` token set, recombine. Catches `knee_L`, `L.knee`, `kneeLeft`.
4. **Fuzzy match** (token-sort ratio, threshold ~0.85) against canonical names — flagged as a *proposal*, not an auto-assignment, when confidence is middling.
5. **Unmapped**: native name preserved; `canonical_by_index[i] = None`.

**Curation, not magic.** Running the resolver on a new model writes `models/registry/<model-id>.aliases.json`:

```json
{
  "_source": "auto-resolved, please review",
  "aliases": {
    "0": {"native": "Nose", "canonical": "nose", "confidence": "exact"},
    "13": {"native": "LKnee", "canonical": "left_knee", "confidence": "heuristic"},
    "41": {"native": "T7_Spine", "canonical": null, "confidence": "unmapped",
           "suggestion": "spine_mid (fuzzy 0.72 — confirm manually)"}
  }
}
```

The user (or a reviewer) edits this file and commits it. Overrides in the file always beat the heuristic. Analyzers fail loudly if they need a canonical name the schema can't deliver, naming the file to edit.

**Why this shape?** It separates three concerns that often get mashed together:
- *What the model outputs* (native names, immutable, shipped by the model author)
- *What calimerge analyzers expect* (canonical vocabulary, owned by us, evolves slowly)
- *The glue* (per-model alias file, editable, version-controlled, human-in-the-loop)

### 1.8 The FP16 bug, and how a clean rebuild flow avoids it

**Where it lives.** [src/cuda_pipeline/pt_tensorrt.cpp:303-332](src/cuda_pipeline/pt_tensorrt.cpp#L303), fixed in commit 2b141cc. When FP16 is enabled on the TensorRT builder, the engine's *input I/O dtype* (the dtype the host writes into the device input buffer) can be set to `kHALF` for throughput. The CUDA preprocessing kernels that fill that buffer must match:

| Kernel | Writes | Input buffer dtype must be |
|---|---|---|
| YOLO letterbox (`pt_kernels.cu`) | `__half` per pixel | `kHALF` |
| VitPose crop-affine (`pt_kernels.cu`) | `float` per pixel | `kFLOAT` |

The original code set `kHALF` unconditionally when FP16 was on. That was fine for YOLO. For VitPose, the crop kernel kept writing FP32 bits into a buffer TRT interpreted as FP16 — so TRT read garbage, every heatmap peak came out around `-1e30`, and every detection was dropped. The fix is a runtime string-match on the ONNX filename:

```cpp
pt_extract_model_name(eng->onnx_path, name_buf, ...);
if (strstr(name_buf, "yolo")) {
    network->getInput(i)->setType(nvinfer1::DataType::kHALF);
} else {
    /* leave FP32, TRT will cast internally */
}
```

**Why it's the canary.** Two separate pieces of model-specific knowledge — *what preprocess kernel to use* and *what input dtype it writes* — are encoded as `if (strstr(path, "yolo"))`. The filename is doing the work of a metadata field. A new model whose ONNX filename doesn't contain "yolo" or "vitpose" silently gets the wrong input shape (see the same pattern at line 364, where H/W is also filename-sniffed). This is the abstraction leak §1.2 refers to.

**The clean version.** Move both facts onto the kernel, not the model:

```python
@dataclass(frozen=True, slots=True)
class PreprocessKernel:
    id: str                         # "yolo_letterbox", "vitpose_crop_affine", "simcc_input"
    writes_dtype: Literal["fp16", "fp32"]  # what the kernel writes — authoritative
    output_shape: tuple[int, int]   # (H, W) the kernel produces
```

`PoseModelSpec.preprocess` references a `PreprocessKernel` by id. The TRT engine builder reads `kernel.writes_dtype` to decide whether to set `kHALF` on the input — no filename sniffing. Kernels and models are orthogonal: a new model reuses an existing kernel (no rebuild) or ships a new one (explicit build step).

**Rebuild flow for a new model.** Two distinct caches, both keyed, both invalidated on relevant inputs:

```
1. ONNX acquisition
   key: (spec.id, hf_repo, hf_revision)
   miss → hf_hub_download + optional torch→onnx export + shape validation
   cache: ~/.calimerge/models/<spec.id>/model.onnx

2. TensorRT engine build
   key: (spec.id, onnx_sha256, gpu_sm, trt_version, precision, kernel.id)
   miss → builder + profile + FP16/FP32 setup (driven by kernel.writes_dtype)
   cache: ~/.calimerge/engines/<hash>.engine
   typical cost: 30s – 5min
```

Kernels themselves live in the C++ .dll/.dylib — part of the normal build, not rebuilt per model. A model that needs preprocessing outside the supported kernel set is an explicit "extension" (ship a .cu alongside the registry entry, rebuild the native library once). We expect this to be rare — letterbox and crop-affine cover most top-down pose models; add `simcc_input` for RTMPose, `sapiens_square` for Sapiens, and you're there.

**What this buys us.** Downloading a new registered model is: hub download, shape check, engine build, ready. Zero C++ recompile. And the next time someone introduces a kernel whose output dtype differs from its siblings, the mismatch becomes a loud error at engine-build time ("kernel says fp16, but model spec claims fp32 preprocess output"), not a silent −1e30 heatmap in production.

---

## 2. WORKOUTABSTRACTION — Programs, Exercises, Analyses

### 2.1 Current state (more exists than I expected)

calimerge already has a meaningful analysis layer — it is *not* purely capture/triangulate/export:

- **Exercise registry** — [src/calimerge/workout_types.py](src/calimerge/workout_types.py): 8 exercise types as frozen dataclasses (sit-to-stand, biceps curl, pushup, pullup, leg raise, tandem stance, TUG, stretch).
- **Analysis functions** — [src/calimerge/analysis/](src/calimerge/analysis/): 13 modules implementing rep counting, joint angles, work/power, balance duration.
- **Programs** — [src/calimerge/programs.py](src/calimerge/programs.py): Vivifrail + Calisthenics templates, hardcoded Python dicts.
- **Persistence** — [src/calimerge/config.py:733-858](src/calimerge/config.py#L733): SQLite `workouts.db` with `programs`, `program_exercises`, `sessions`, `session_results`, `users` tables; idempotent seeding.

So the ask is not "build this from zero" but **refactor for the right abstraction and add JSON (de)serialization**.

### 2.2 Critical gaps in the existing design

| Gap | Evidence | Impact |
|---|---|---|
| No XYZPoints → per-frame skeleton converter | [triangulation.py:197-263](src/calimerge/triangulation.py#L197) returns `XYZPoints`; analysis expects `(N, P, K, 3)` arrays | Analysis code can't actually consume triangulation output |
| Time is unaligned | `XYZPoints.sync_index` only; `frame_time_history.csv` exists but isn't joined in | Can't compute `∫ power dt`; per-rep durations are frame-count based |
| Analysis per-exercise is one function, not a config | Each exercise points to one hardcoded analyzer | Can't say "squat uses joint-angle analyzer, caring about knee, init=standing"; each new variant needs a new function |
| Programs are Python dicts + SQL, not JSON | [programs.py:14-112](src/calimerge/programs.py#L14) | Can't ship/import programs as files; no GUI authoring path |
| No multi-person session model | `max_persons=1` default; session table is per-user | Group-workout analysis impossible without schema change |
| Offline-only | `ProcessingWorker` loads recorded videos | No live rep feedback during capture |

### 2.3 Proposed abstraction

Three layers, each a frozen dataclass with pure-function operators. The key move is **separating the analyzer (reusable logic) from the exercise (how that logic is configured for this movement)**.

```python
@dataclass(frozen=True, slots=True)
class AnalyzerConfig:
    """Parameterized configuration of a reusable analyzer."""
    analyzer: str                   # registry key, e.g. "joint_angle_reps"
    params: dict[str, Any]          # analyzer-specific; schema-validated
    # Common fields most analyzers read:
    #   target_joints: list[str]    (schema names, not indices — see §1)
    #   init_pose: str | None       (e.g. "standing", "prone")
    #   rep_trigger: {"angle_threshold": {"joint": "knee", "below": 95}}

@dataclass(frozen=True, slots=True)
class Exercise:
    id: str                         # stable, e.g. "back-squat"
    display_name: str
    analysis: AnalyzerConfig        # THE key decoupling
    sets: int | None = None
    target_reps: int | None = None
    target_duration_s: float | None = None
    rest_s: float | None = None
    notes: str = ""

@dataclass(frozen=True, slots=True)
class Program:
    id: str
    display_name: str
    description: str
    exercises: tuple[Exercise, ...]   # ordered
    schedule: ProgramSchedule | None  # days/week, suggested cadence

@dataclass(frozen=True, slots=True)
class WorkoutSession:
    """A single recorded instance of executing a program (or ad-hoc exercises)."""
    id: str
    program_id: str | None
    user_id: str
    started_at: datetime
    recording_dir: Path
    exercise_results: tuple[ExerciseResult, ...]
```

The **analyzer registry** is the mirror of §1's model registry:

```python
ANALYZER_REGISTRY: dict[str, Analyzer] = {
    "joint_angle_reps": ...,    # back-squat, biceps curl, pushup, pullup
    "com_displacement_reps": ..., # sit-to-stand, deadlift
    "balance_duration": ...,    # tandem stance
    "path_timing": ...,         # TUG
    "static_hold": ...,         # stretches, planks
}
```

Each analyzer exposes a JSON schema for its `params`; `AnalyzerConfig.params` is validated at load time. This is what lets squat and deadlift share `joint_angle_reps` but differ on `target_joints=["knee"]` vs `["hip"]`, and lets bicep curl specify `init_pose="arm_extended"`.

### 2.4 JSON schema for programs

Programs round-trip as JSON; SQLite remains the live store but is populated from JSON on import and exported on demand. Example:

```json
{
  "id": "vivifrail-low-a",
  "display_name": "Vivifrail – Low intensity A",
  "description": "…",
  "schedule": {"days_per_week": 3, "suggested_days": ["Mon","Wed","Fri"]},
  "exercises": [
    {
      "id": "sit-to-stand",
      "display_name": "Sit-to-stand",
      "sets": 3, "target_reps": 10, "rest_s": 60,
      "analysis": {
        "analyzer": "com_displacement_reps",
        "params": {
          "axis": "z",
          "init_pose": "seated",
          "rep_threshold_m": 0.15,
          "smoothing_window_s": 0.25
        }
      }
    }
  ]
}
```

Round-trip contract: `Program.from_json(path) → Program → to_json()` produces byte-identical output (ordering normalized, no unstable dict iteration).

### 2.5 Pipeline changes required

1. **Add `Skeleton3D`** — the missing converter. `XYZPoints` stream → `(N, P, K, 3)` array + `timestamps: (N,)` from `frame_time_history.csv`, joined on sync_index. One pure function in a new `src/calimerge/skeleton_3d.py`. Name deliberately carries the `3D` so call sites like `skeleton: Skeleton3D` make the dimensionality unmistakable — never confused with 2D pose output.
2. **Analyzer I/O contract** — every analyzer takes `(Skeleton3D, KeypointSchema, AnalyzerConfig) → ExerciseResult`. Schema-aware; looks up joints by *name*.
3. **Program runner** — `run_program(program, session_root) → WorkoutSession`: iterates exercises, slices video/keypoints by rep-break markers (user-driven or timestamp), calls the right analyzer, aggregates results.
4. **Persistence** — keep SQLite for session history/queries; add `json_spec TEXT` column on `programs` to round-trip the authored form.
5. **GUI program editor** (Tab 4 or new tab) — form-driven over `AnalyzerConfig.params` using each analyzer's JSON schema. Save produces the same JSON file.

### 2.6 Migration path (low-risk)

- Existing hardcoded [programs.py](src/calimerge/programs.py) gets rewritten as JSON files under `src/calimerge/data/programs/*.json`, loaded at seed time — the SQLite seeder reads JSON instead of Python dicts.
- Existing analysis modules become thin adapters that implement the `Analyzer` protocol and declare their JSON-schema params.
- `WorkoutType` stays as a **preset** layer (a named `AnalyzerConfig` with sensible defaults) so existing callsites still work.

### 2.7 Critical limitations to call out now

- **Offline analysis only.** Live rep feedback during recording is a separable, bigger project (needs streaming triangulation). Design the data model so it is not *precluded* — `Skeleton3D` can be produced incrementally — but don't commit to it in v1.
- **Single person.** Schema has `persons_axis`, but tracker identity persistence across sessions is unsolved. For multi-person programs, require explicit per-person track IDs chosen at session start.
- **Frame-to-time drift.** If sync across cameras drifts, per-rep durations get noisy. Worth auditing `frame_time_history.csv` derivation before trusting `∫ power dt`.
- **Keypoint name stability across pose backends** (see §1.6). Analyzers referencing `"knee"` must work regardless of backend. This is a point of coupling between the two designs: the `KeypointSchema.names` registry is the shared contract.
- **Breaks unit tests?** None at risk identified in research, but the `WorkoutType` → `AnalyzerConfig` migration will touch [workout_types.py](src/calimerge/workout_types.py) and any test that imports it — worth a spike before committing to the rename.

---

## 3. Shared concern: the schema-vs-indices split

Both designs rely on one thing: keypoints addressed by **name**, never by raw index, everywhere above the backend layer. Today this invariant is violated (hip indices hardcoded in tracker; analyzers read fixed array slots). Fixing this is the smallest possible change that unlocks both pluggable backends *and* cross-model analysis. It is the prerequisite for everything else.
