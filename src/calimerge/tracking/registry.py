"""
Pose-model registry.

Single source of truth for "what model does the GUI/runner mean by
``synthpose`` (or any other key)?". Built-in entries are defined here.
User-added entries are read from TOML files under
``<app_data>/models/registry/*.toml`` (or the repo's ``models/registry/``
fallback) at first registry access.

Adding a new model that fits today's C-side shape contract — 52 keypoints,
256×192 input, crop_affine preprocess, ImageNet normalization, heatmap
output — is just dropping one TOML file. Models that don't fit those
constraints load via the PyTorch backend today; the CUDA/MPS paths
require a one-time C++ refactor (DESIGN.md Phase C). The validator below
flags entries that won't run through the C side and refuses to register
them silently.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator

from ..types import KeypointSchema, PoseModelSpec
from .markers import SYNTHPOSE_MARKERS


# ──────────────────────────────────────────────────────────────────────────
# Built-in schemas
# ──────────────────────────────────────────────────────────────────────────

SYNTHPOSE_SCHEMA = KeypointSchema(
    names=tuple(SYNTHPOSE_MARKERS[i] for i in sorted(SYNTHPOSE_MARKERS))
)

# 21-keypoint MediaPipe Hands schema. Names from MediaPipe's HandLandmark enum
# (https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).
MEDIAPIPE_HANDS_SCHEMA = KeypointSchema(
    names=(
        "WRIST",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
        "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    )
)


# ──────────────────────────────────────────────────────────────────────────
# C-side shape contract (today)
# ──────────────────────────────────────────────────────────────────────────
#
# Until Phase C lands, the CUDA/MPS pipelines (pt_common.h, pt_kernels.cu,
# pt_preprocess.m) are pinned to these exact dimensions. A registry entry
# is "C-runnable" only if it matches all three:

_C_RUNNABLE_INPUT_SHAPE = (256, 192)
_C_RUNNABLE_KEYPOINT_COUNT = 52
_C_RUNNABLE_PREPROCESS = "crop_affine"


def is_c_runnable(spec: PoseModelSpec) -> bool:
    """True iff this spec can run through the CUDA/MPS C pipelines today."""
    return (
        spec.input_shape == _C_RUNNABLE_INPUT_SHAPE
        and spec.schema.K == _C_RUNNABLE_KEYPOINT_COUNT
        and spec.preprocess == _C_RUNNABLE_PREPROCESS
    )


# ──────────────────────────────────────────────────────────────────────────
# Built-in entries
# ──────────────────────────────────────────────────────────────────────────

_BUILTINS: dict[str, PoseModelSpec] = {
    "synthpose": PoseModelSpec(
        id="synthpose",
        display_name="VitPose / SynthPose (52 kp)",
        hf_repo="stanfordmimi/synthpose-vitpose-base-hf",
        input_shape=(256, 192),
        schema=SYNTHPOSE_SCHEMA,
        onnx_filename="vitpose_synthpose.onnx",
        coreml_filename="vitpose_synthpose.mlpackage",
        fp16_safe_io=False,
        preprocess="crop_affine",
        postprocess="heatmap_argmax",
        notes="Default body model. SynthPose-trained VitPose-base extending COCO-17 with 35 anatomical landmarks.",
    ),
    "mediapipe_hands": PoseModelSpec(
        id="mediapipe_hands",
        display_name="MediaPipe Hands (21 kp)",
        hf_repo=None,  # bundled .task file; no HF download
        input_shape=(224, 224),
        schema=MEDIAPIPE_HANDS_SCHEMA,
        preprocess="letterbox",
        postprocess="regression",
        notes="Apple-platform-friendly hand detector. PyTorch path only — not wired through C pipelines.",
    ),
}


# ──────────────────────────────────────────────────────────────────────────
# Loader (TOML extension)
# ──────────────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_loaded = False
_registry: dict[str, PoseModelSpec] = {}


def _registry_dirs() -> list[Path]:
    """Directories scanned for user-added registry entries, in priority order."""
    out: list[Path] = []
    try:
        from ..config import models_dir
        out.append(models_dir() / "registry")
    except Exception:
        pass
    repo_root = Path(__file__).resolve().parents[3]
    out.append(repo_root / "models" / "registry")
    return out


def _spec_from_toml(toml_path: Path) -> PoseModelSpec:
    """Parse a registry TOML file into a PoseModelSpec."""
    import rtoml
    raw = rtoml.load(toml_path)
    schema_names = tuple(raw.get("schema_names", ()))
    if not schema_names:
        raise ValueError(
            f"{toml_path}: must declare ``schema_names = [...]`` "
            "(ordered keypoint names)"
        )
    norm = raw.get("normalization", {})
    mean = tuple(norm.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(norm.get("std", (0.229, 0.224, 0.225)))
    return PoseModelSpec(
        id=str(raw["id"]),
        display_name=str(raw.get("display_name", raw["id"])),
        hf_repo=raw.get("hf_repo"),
        input_shape=tuple(raw.get("input_shape", (256, 192))),
        normalization=(mean, std),
        schema=KeypointSchema(names=schema_names),
        onnx_filename=raw.get("onnx_filename"),
        coreml_filename=raw.get("coreml_filename"),
        fp16_safe_io=bool(raw.get("fp16_safe_io", False)),
        preprocess=raw.get("preprocess", "crop_affine"),
        postprocess=raw.get("postprocess", "heatmap_argmax"),
        notes=str(raw.get("notes", "")),
    )


def _load() -> None:
    global _loaded
    with _lock:
        if _loaded:
            return
        _registry.clear()
        _registry.update(_BUILTINS)
        for d in _registry_dirs():
            if not d.exists():
                continue
            for tp in sorted(d.glob("*.toml")):
                try:
                    spec = _spec_from_toml(tp)
                    _registry[spec.id] = spec
                except Exception as e:
                    # Log but don't crash on a bad user file — built-ins
                    # remain available either way.
                    print(f"[registry] skipping {tp}: {e}")
        _loaded = True


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────


def get(model_key: str) -> PoseModelSpec:
    """Return the spec for ``model_key`` or raise ``KeyError`` listing what's known."""
    _load()
    if model_key in _registry:
        return _registry[model_key]
    # One transparent legacy alias: anything that used to read "vitpose"
    # is the SynthPose model now.
    if model_key == "vitpose" and "synthpose" in _registry:
        return _registry["synthpose"]
    raise KeyError(
        f"unknown pose model {model_key!r}; available: {sorted(_registry)}"
    )


def has(model_key: str) -> bool:
    """Cheap existence check that doesn't raise."""
    _load()
    return model_key in _registry or (
        model_key == "vitpose" and "synthpose" in _registry
    )


def all_specs() -> list[PoseModelSpec]:
    _load()
    return list(_registry.values())


def keys() -> list[str]:
    _load()
    return sorted(_registry.keys())


def iter_specs() -> Iterator[PoseModelSpec]:
    _load()
    yield from _registry.values()


def reload() -> None:
    """Force the next access to re-read TOML files. Useful in tests."""
    global _loaded
    with _lock:
        _loaded = False
        _registry.clear()
