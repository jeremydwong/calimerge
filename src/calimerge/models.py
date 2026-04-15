"""
Model registry for detection and tracking models.

Provides a central registry of all supported models with their metadata,
so that sessions can record exactly which model was used for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    name: str           # "yolo_v10s", "vitpose_base_coco", "mediapipe_hands"
    model_type: str     # "detection", "pose", "hand"
    source: str         # "ultralytics", "huggingface", "mediapipe"
    version: str        # "10s", "base", "0.8.4"
    num_keypoints: int  # 0 for detection, 17 for COCO, 21 for hand
    description: str


AVAILABLE_MODELS: dict[str, ModelInfo] = {
    "yolo_v10s": ModelInfo(
        name="yolo_v10s",
        model_type="detection",
        source="ultralytics",
        version="10s",
        num_keypoints=0,
        description="YOLOv10-small person detector",
    ),
    "yolo_v8n_pose": ModelInfo(
        name="yolo_v8n_pose",
        model_type="pose",
        source="ultralytics",
        version="8n",
        num_keypoints=17,
        description="YOLOv8-nano pose estimator (COCO 17 keypoints)",
    ),
    "vitpose_base_coco": ModelInfo(
        name="vitpose_base_coco",
        model_type="pose",
        source="huggingface",
        version="base",
        num_keypoints=17,
        description="ViTPose-base trained on COCO (17 keypoints)",
    ),
    "vitpose_large_coco": ModelInfo(
        name="vitpose_large_coco",
        model_type="pose",
        source="huggingface",
        version="large",
        num_keypoints=17,
        description="ViTPose-large trained on COCO (17 keypoints)",
    ),
    "mediapipe_hands": ModelInfo(
        name="mediapipe_hands",
        model_type="hand",
        source="mediapipe",
        version="0.8.4",
        num_keypoints=21,
        description="MediaPipe Hands — 21 hand landmarks per hand",
    ),
}


def get_model_version_string(model_name: str) -> str:
    """Return a compact version string for storage in the sessions table.

    Format: "<name>@<version>" e.g. "vitpose_base_coco@base"
    """
    info = AVAILABLE_MODELS.get(model_name)
    if info is None:
        return model_name
    return f"{info.name}@{info.version}"


def get_model_info(model_name: str) -> ModelInfo | None:
    """Look up a model by name. Returns None if not found."""
    return AVAILABLE_MODELS.get(model_name)
