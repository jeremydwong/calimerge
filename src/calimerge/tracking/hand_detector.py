"""
Hand landmark detection using MediaPipe Hands.

Thin wrapper around mediapipe.solutions.hands that takes a BGR frame
and returns hand landmark positions in pixel coordinates.

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

import numpy as np


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    """Landmarks for a single detected hand.

    Attributes
    ----------
    landmarks : (21, 2) array
        Pixel coordinates (x, y) for each of the 21 hand landmarks.
    handedness : str
        "Left" or "Right".
    score : float
        Detection confidence score.
    """
    landmarks: np.ndarray   # (21, 2) pixel coords
    handedness: str
    score: float


def detect_hands(
    frame: np.ndarray,
    max_hands: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> list[HandLandmarks]:
    """
    Detect hand landmarks in a BGR frame using MediaPipe Hands.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (H, W, 3), as produced by OpenCV.
    max_hands : int
        Maximum number of hands to detect.
    min_detection_confidence : float
        Minimum confidence for the initial hand detection.
    min_tracking_confidence : float
        Minimum confidence for landmark tracking.

    Returns
    -------
    list[HandLandmarks]
        One entry per detected hand, with pixel-space landmarks.
    """
    import cv2
    import mediapipe as mp

    h, w = frame.shape[:2]

    # MediaPipe expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hands_module = mp.solutions.hands
    with hands_module.Hands(
        static_image_mode=True,
        max_num_hands=max_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as hands:
        results = hands.process(rgb)

    if results.multi_hand_landmarks is None:
        return []

    detected: list[HandLandmarks] = []
    for i, hand_lms in enumerate(results.multi_hand_landmarks):
        # Convert normalized landmarks to pixel coordinates
        coords = np.array(
            [(lm.x * w, lm.y * h) for lm in hand_lms.landmark],
            dtype=np.float64,
        )  # (21, 2)

        # Handedness
        if results.multi_handedness and i < len(results.multi_handedness):
            classification = results.multi_handedness[i].classification[0]
            handedness = classification.label  # "Left" or "Right"
            score = classification.score
        else:
            handedness = "Unknown"
            score = 0.0

        detected.append(HandLandmarks(
            landmarks=coords,
            handedness=handedness,
            score=score,
        ))

    return detected


def get_thumb_index_distance(hand: HandLandmarks) -> float:
    """Return the pixel distance between thumb tip (4) and index tip (8).

    Parameters
    ----------
    hand : HandLandmarks
        A detected hand with 21 landmarks.

    Returns
    -------
    float
        Euclidean distance in pixels.
    """
    thumb_tip = hand.landmarks[4]
    index_tip = hand.landmarks[8]
    return float(np.linalg.norm(thumb_tip - index_tip))
