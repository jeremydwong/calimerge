"""
Person detection and pose estimation.

Uses YOLO v10s for person detection and VitPose-Base (SynthPose) for
52-keypoint pose estimation. Models are auto-downloaded from HuggingFace.

Adapted from posetrack/pose_detector.py.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


# HuggingFace model identifiers
YOLO_MODEL_ID = "jameslahm/yolov10s.pt"
VITPOSE_MODEL_ID = "usyd-community/vitpose-base-simple"


def setup_device(device_name: str = "auto") -> str:
    """Auto-detect the best available compute device."""
    if device_name == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    if device_name == "mps" and (
        not torch.backends.mps.is_available() or not torch.backends.mps.is_built()
    ):
        return "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_name


def load_models(
    device: str = "cpu",
    log_fn=None,
):
    """
    Load person detection and pose estimation models.

    Models are auto-downloaded from HuggingFace Hub on first use
    and cached locally for subsequent runs.

    Returns:
        (person_model, pose_processor, pose_model)
    """
    def log(msg):
        if log_fn:
            log_fn(msg)

    log(f"Loading models to device: {device}")

    # --- Person Detection: YOLO v10s ---
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download

    log("Loading YOLO person detection model...")
    model_filename = YOLO_MODEL_ID.split("/")[-1]
    model_path = hf_hub_download(repo_id=YOLO_MODEL_ID, filename=model_filename)
    person_model = YOLO(model_path)
    person_model.to(device)

    # --- Pose Estimation: VitPose-Base (SynthPose) ---
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    log("Loading VitPose pose estimation model...")
    pose_processor = AutoProcessor.from_pretrained(VITPOSE_MODEL_ID)
    pose_model = VitPoseForPoseEstimation.from_pretrained(
        VITPOSE_MODEL_ID, device_map=device
    )

    # Set to eval mode
    person_model.eval()
    pose_model.eval()

    log("Models loaded successfully.")
    return person_model, pose_processor, pose_model


def detect_persons(
    image: Image.Image,
    person_model,
    device: str,
    confidence_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect persons in a PIL image using YOLO.

    Returns:
        (boxes_voc, boxes_coco, scores) where:
        - boxes_voc: (N, 4) in x1,y1,x2,y2 format
        - boxes_coco: (N, 4) in x1,y1,w,h format
        - scores: (N,) confidence scores
    """
    results = person_model(
        image, conf=confidence_threshold, half=True, classes=[0], verbose=False
    )

    if len(results[0].boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    boxes = results[0].boxes
    boxes_voc = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()

    # VOC (x1,y1,x2,y2) -> COCO (x1,y1,w,h)
    boxes_coco = boxes_voc.copy()
    boxes_coco[:, 2] = boxes_coco[:, 2] - boxes_coco[:, 0]
    boxes_coco[:, 3] = boxes_coco[:, 3] - boxes_coco[:, 1]

    return boxes_voc, boxes_coco, scores


def detect_persons_batch(
    images: list[Image.Image],
    person_model,
    device: str,
    confidence_threshold: float = 0.3,
    batch_size: int = 8,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Batch person detection across multiple images.

    Returns list of (boxes_voc, boxes_coco, scores) per image.
    """
    batch_results = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        results = person_model(
            batch_images,
            conf=confidence_threshold,
            half=True,
            classes=[0],
            batch=len(batch_images),
            verbose=False,
        )

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_voc = result.boxes.xyxy.cpu().numpy()

                boxes_coco = boxes_voc.copy()
                boxes_coco[:, 2] = boxes_coco[:, 2] - boxes_coco[:, 0]
                boxes_coco[:, 3] = boxes_coco[:, 3] - boxes_coco[:, 1]

                scores = result.boxes.conf.cpu().numpy()
            else:
                boxes_voc = np.empty((0, 4), dtype=np.float32)
                boxes_coco = np.empty((0, 4), dtype=np.float32)
                scores = np.empty((0,), dtype=np.float32)

            batch_results.append((boxes_voc, boxes_coco, scores))

    return batch_results


def estimate_poses(
    image: Image.Image,
    person_boxes_coco: np.ndarray,
    pose_processor,
    pose_model,
    device: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Estimate poses for detected persons in a single image.

    Returns:
        (all_keypoints, all_keypoint_scores) where each is a list
        of arrays, one per detected person.
    """
    if person_boxes_coco.size == 0:
        return [], []

    boxes_list = person_boxes_coco.astype(np.float32).tolist()

    inputs = pose_processor(image, boxes=[boxes_list], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = pose_model(**inputs)

    pose_results = pose_processor.post_process_pose_estimation(
        outputs, boxes=[boxes_list]
    )

    if not pose_results:
        return [], []

    image_results = pose_results[0]

    all_keypoints = []
    all_scores = []

    if not isinstance(image_results, list):
        return [], []

    for person_result in image_results:
        if not isinstance(person_result, dict):
            continue

        keypoints = person_result.get("keypoints")
        scores = person_result.get("scores")

        if keypoints is None or scores is None:
            continue

        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        if isinstance(keypoints, np.ndarray) and keypoints.ndim == 2 and keypoints.shape[1] >= 2:
            all_keypoints.append(keypoints)
            all_scores.append(scores)

    return all_keypoints, all_scores


def estimate_poses_batch(
    images_with_boxes: list[tuple[Image.Image, np.ndarray]],
    pose_processor,
    pose_model,
    device: str,
    batch_size: int = 8,
) -> list[tuple[list[np.ndarray], list[np.ndarray]]]:
    """
    Batch pose estimation across multiple images.

    Args:
        images_with_boxes: list of (image, person_boxes_coco) tuples.

    Returns:
        List of (all_keypoints, all_scores) per image.
    """
    batch_results = []

    for i in range(0, len(images_with_boxes), batch_size):
        batch_data = images_with_boxes[i : i + batch_size]

        batch_images = []
        batch_boxes_lists = []

        for image, boxes_coco in batch_data:
            batch_images.append(image)
            if boxes_coco.size == 0:
                batch_boxes_lists.append([[0, 0, 1, 1]])  # dummy box
            else:
                batch_boxes_lists.append(boxes_coco.astype(np.float32).tolist())

        inputs = pose_processor(
            batch_images, boxes=batch_boxes_lists, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = pose_model(**inputs)

        pose_results = pose_processor.post_process_pose_estimation(
            outputs, boxes=batch_boxes_lists
        )

        for j, (image, boxes_coco) in enumerate(batch_data):
            if boxes_coco.size == 0:
                batch_results.append(([], []))
                continue

            if j >= len(pose_results):
                batch_results.append(([], []))
                continue

            image_results = pose_results[j]

            all_keypoints = []
            all_scores = []

            if not isinstance(image_results, list):
                batch_results.append(([], []))
                continue

            for person_result in image_results:
                if not isinstance(person_result, dict):
                    continue

                keypoints = person_result.get("keypoints")
                scores = person_result.get("scores")

                if keypoints is None or scores is None:
                    continue

                if isinstance(keypoints, torch.Tensor):
                    keypoints = keypoints.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()

                if (
                    isinstance(keypoints, np.ndarray)
                    and keypoints.ndim == 2
                    and keypoints.shape[1] >= 2
                ):
                    all_keypoints.append(keypoints)
                    all_scores.append(scores)

            batch_results.append((all_keypoints, all_scores))

    return batch_results
