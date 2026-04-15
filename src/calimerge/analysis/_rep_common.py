"""
Shared helpers for rep-based workout analyses (sit-to-stand, biceps, pushups).
"""

from __future__ import annotations

import numpy as np


def smooth(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average smoothing, preserving length."""
    if len(signal) < window:
        return signal
    kernel = np.ones(window) / window
    padded = np.pad(signal, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(signal)]


def elbow_angle_deg(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    """Compute the internal elbow angle (degrees) from three 3D keypoints.

    0°   = forearm folded onto upper arm (fully flexed)
    180° = arm straight (fully extended)

    Returns NaN if any input contains NaN or the vectors are degenerate.
    """
    if shoulder is None or elbow is None or wrist is None:
        return float("nan")
    s = np.asarray(shoulder, dtype=float)
    e = np.asarray(elbow, dtype=float)
    w = np.asarray(wrist, dtype=float)
    if np.isnan(s).any() or np.isnan(e).any() or np.isnan(w).any():
        return float("nan")
    v1 = s - e
    v2 = w - e
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def average_elbow_angle(persons_kps: list) -> float:
    """Compute the average of left and right elbow angles for person 0.

    COCO-17 indices: 5=L_shoulder, 6=R_shoulder, 7=L_elbow, 8=R_elbow,
                     9=L_wrist, 10=R_wrist.

    Returns NaN if neither elbow produces a valid angle.
    """
    if not persons_kps or persons_kps[0] is None:
        return float("nan")
    kps = persons_kps[0]
    if len(kps) < 11:
        return float("nan")
    left = elbow_angle_deg(kps[5], kps[7], kps[9])
    right = elbow_angle_deg(kps[6], kps[8], kps[10])

    vals = [v for v in (left, right) if not np.isnan(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def find_upward_crossings(signal: np.ndarray, threshold: float,
                          min_gap: int = 1) -> list[int]:
    """Return indices where signal crosses UPWARD through threshold.

    A crossing is detected at index i when signal[i-1] <= threshold < signal[i].
    Consecutive crossings closer than min_gap samples are merged.
    """
    above = signal > threshold
    crossings: list[int] = []
    for i in range(1, len(signal)):
        if above[i] and not above[i - 1]:
            if not crossings or (i - crossings[-1]) >= min_gap:
                crossings.append(i)
    return crossings


def find_downward_crossings(signal: np.ndarray, threshold: float,
                            min_gap: int = 1) -> list[int]:
    """Return indices where signal crosses DOWNWARD through threshold.

    A crossing is detected at index i when signal[i-1] >= threshold > signal[i].
    Consecutive crossings closer than min_gap samples are merged.
    """
    below = signal < threshold
    crossings: list[int] = []
    for i in range(1, len(signal)):
        if below[i] and not below[i - 1]:
            if not crossings or (i - crossings[-1]) >= min_gap:
                crossings.append(i)
    return crossings


def interpolate_nans(signal: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN entries in a 1D signal. Leaves constant-NaN as-is."""
    signal = np.asarray(signal, dtype=float).copy()
    mask = np.isnan(signal)
    if not np.any(mask):
        return signal
    if np.all(mask):
        return signal
    idx = np.arange(len(signal))
    signal[mask] = np.interp(idx[mask], idx[~mask], signal[~mask])
    return signal
