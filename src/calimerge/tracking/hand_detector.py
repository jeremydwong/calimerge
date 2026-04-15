"""
Hand landmark detection using MediaPipe Hands (tasks API, v0.10+).

Thin wrapper that takes a BGR frame and returns hand landmark positions
in pixel coordinates.

MediaPipe Hands outputs 21 landmarks per hand:
  0: WRIST
  1-4: THUMB (CMC, MCP, IP, TIP)
  5-8: INDEX (MCP, PIP, DIP, TIP)
  9-12: MIDDLE (MCP, PIP, DIP, TIP)
  13-16: RING (MCP, PIP, DIP, TIP)
  17-20: PINKY (MCP, PIP, DIP, TIP)

Key landmarks for squeeze detection:
  4: THUMB_TIP
  8: INDEX_FINGER_TIP
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    """Landmarks for a single detected hand."""
    landmarks: np.ndarray   # (21, 2) pixel coords
    handedness: str          # "Left" or "Right"
    score: float


# Module-level detector cache (reused across calls)
_detector = None
_detector_max_hands = 0


def _get_detector(max_hands: int = 2, min_confidence: float = 0.5):
    """Get or create a cached HandLandmarker detector."""
    global _detector, _detector_max_hands

    if _detector is not None and _detector_max_hands == max_hands:
        return _detector

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    # Download the hand landmarker model if needed
    model_path = Path(__file__).parent / "hand_landmarker.task"
    if not model_path.exists():
        import urllib.request
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )
        print(f"[hand_detector] Downloading hand_landmarker.task...")
        urllib.request.urlretrieve(url, str(model_path))
        print(f"[hand_detector] Downloaded to {model_path}")

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(model_path),
        ),
        num_hands=max_hands,
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    _detector = mp_vision.HandLandmarker.create_from_options(options)
    _detector_max_hands = max_hands
    return _detector


def detect_hands(
    frame: np.ndarray,
    max_hands: int = 2,
    min_detection_confidence: float = 0.5,
) -> list:
    """
    Detect hand landmarks in a BGR frame.

    Returns a list of lists, where each inner list contains 21 (x, y, z)
    tuples in normalized coordinates (0-1 range relative to frame size).

    For backwards compatibility with the worker code that expects this format.
    """
    import cv2
    import mediapipe as mp

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detector = _get_detector(max_hands, min_detection_confidence)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    hands_out = []
    if result.hand_landmarks:
        for hand_lms in result.hand_landmarks:
            # Each landmark has x, y, z in normalized coords
            lms = [(lm.x, lm.y, lm.z) for lm in hand_lms]
            hands_out.append(lms)

    return hands_out


def detect_hands_full(
    frame: np.ndarray,
    max_hands: int = 2,
    min_detection_confidence: float = 0.5,
) -> list[HandLandmarks]:
    """
    Detect hand landmarks and return structured HandLandmarks objects.
    """
    import cv2
    import mediapipe as mp

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detector = _get_detector(max_hands, min_detection_confidence)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    detected: list[HandLandmarks] = []
    if result.hand_landmarks:
        for i, hand_lms in enumerate(result.hand_landmarks):
            coords = np.array(
                [(lm.x * w, lm.y * h) for lm in hand_lms],
                dtype=np.float64,
            )

            handedness = "Unknown"
            score = 0.0
            if result.handedness and i < len(result.handedness):
                cat = result.handedness[i][0]
                handedness = cat.category_name
                score = cat.score

            detected.append(HandLandmarks(
                landmarks=coords,
                handedness=handedness,
                score=score,
            ))

    return detected


def get_thumb_index_distance(hand) -> float:
    """Return the pixel distance between thumb tip (4) and index tip (8).

    Accepts either a HandLandmarks object or a list of (x, y, z) tuples
    (normalized coords — returns normalized distance in that case).
    """
    if isinstance(hand, HandLandmarks):
        thumb_tip = hand.landmarks[4]
        index_tip = hand.landmarks[8]
        return float(np.linalg.norm(thumb_tip - index_tip))
    elif isinstance(hand, list) and len(hand) >= 9:
        tx, ty = hand[4][0], hand[4][1]
        ix, iy = hand[8][0], hand[8][1]
        return float(np.sqrt((tx - ix)**2 + (ty - iy)**2))
    return 0.0
