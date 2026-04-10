"""
Hand squeeze analysis — rep counting from thumb-to-index fingertip distance.

For older adults doing grip/squeeze exercises with a ball or similar prop.

Algorithm:
  1. Compute the Euclidean distance between thumb tip (landmark 4) and
     index fingertip (landmark 8) at each frame.
  2. Smooth the distance signal.
  3. Count DOWNWARD crossings of a "closed" threshold — each time the
     distance drops below the threshold counts as one squeeze rep.
  4. Report rep count, average squeeze distance, and squeeze speed.

Note: this uses 2D pixel-space distances normalized by hand size, so the
output distance is dimensionless (fraction of hand span). For a first
approximation, the threshold is specified in centimeters assuming a
typical hand-to-camera distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._rep_common import smooth, find_downward_crossings, interpolate_nans


@dataclass(frozen=True)
class HandSqueezeResult:
    rep_count: int
    total_time_seconds: float
    per_rep_times: list[float]          # time between successive squeezes
    rep_trigger_times: list[float]      # time at each squeeze crossing
    rep_min_distances: list[float]      # closest distance reached in each rep
    avg_squeeze_distance_m: float       # mean min distance across reps
    avg_squeeze_speed_mps: float | None # average squeeze speed (distance / time)
    squeeze_threshold_m: float          # threshold used

    def to_dict(self) -> dict:
        return {
            "rep_count": self.rep_count,
            "total_time_seconds": self.total_time_seconds,
            "per_rep_times": self.per_rep_times,
            "rep_trigger_times": self.rep_trigger_times,
            "rep_min_distances": self.rep_min_distances,
            "avg_squeeze_distance_m": self.avg_squeeze_distance_m,
            "avg_squeeze_speed_mps": self.avg_squeeze_speed_mps,
            "squeeze_threshold_m": self.squeeze_threshold_m,
        }


def compute_thumb_index_distance(
    thumb_tip: np.ndarray,
    index_tip: np.ndarray,
) -> np.ndarray:
    """Compute per-frame Euclidean distance between thumb tip and index tip.

    Parameters
    ----------
    thumb_tip : (N, 2) or (N, 3) array
        Thumb tip positions over time.
    index_tip : (N, 2) or (N, 3) array
        Index fingertip positions over time.

    Returns
    -------
    distances : (N,) array
        Euclidean distance at each frame.
    """
    thumb = np.asarray(thumb_tip, dtype=float)
    index = np.asarray(index_tip, dtype=float)
    diff = thumb - index
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def analyze_hand_squeeze(
    distances: np.ndarray,
    timestamps: np.ndarray,
    squeeze_threshold_m: float = 0.03,
    smooth_window: int = 5,
    min_gap_frames: int = 10,
) -> HandSqueezeResult:
    """
    Count squeeze reps from a thumb-to-index distance time series.

    Parameters
    ----------
    distances : (N,) array
        Distance between thumb tip and index fingertip (meters or
        normalized units). NaNs are interpolated.
    timestamps : (N,) array
        Seconds aligned with distances.
    squeeze_threshold_m : float
        Distance below which the hand is considered "squeezed".
        A downward crossing through this threshold counts as one rep.
    smooth_window : int
        Moving-average window for smoothing.
    min_gap_frames : int
        Minimum frames between successive reps (anti-noise).

    Returns
    -------
    HandSqueezeResult
    """
    empty = HandSqueezeResult(
        rep_count=0,
        total_time_seconds=0.0,
        per_rep_times=[],
        rep_trigger_times=[],
        rep_min_distances=[],
        avg_squeeze_distance_m=0.0,
        avg_squeeze_speed_mps=None,
        squeeze_threshold_m=squeeze_threshold_m,
    )

    if len(distances) < 3:
        return empty

    d = interpolate_nans(np.asarray(distances, dtype=float))
    if np.all(np.isnan(d)):
        return empty

    smoothed = smooth(d, smooth_window)
    crossings = find_downward_crossings(smoothed, squeeze_threshold_m,
                                        min_gap=min_gap_frames)

    if len(crossings) == 0:
        return empty

    rep_count = len(crossings)

    # Time between successive reps
    per_rep_times: list[float] = []
    for i in range(1, len(crossings)):
        dt = float(timestamps[crossings[i]] - timestamps[crossings[i - 1]])
        per_rep_times.append(dt)

    # Find minimum distance within each rep window
    rep_trigger_times: list[float] = []
    rep_min_distances: list[float] = []
    for i, start_idx in enumerate(crossings):
        end_idx = crossings[i + 1] if i + 1 < len(crossings) else len(smoothed)
        segment = smoothed[start_idx:end_idx]
        if len(segment) == 0:
            continue
        rep_trigger_times.append(float(timestamps[start_idx]))
        rep_min_distances.append(float(np.min(segment)))

    # Total time from first to last rep
    total_time = 0.0
    if len(crossings) >= 2:
        total_time = float(timestamps[crossings[-1]] - timestamps[crossings[0]])

    avg_squeeze = 0.0
    if rep_min_distances:
        avg_squeeze = float(np.mean(rep_min_distances))

    # Average squeeze speed: how fast the distance drops per rep
    avg_speed = None
    if per_rep_times:
        # Speed = threshold distance / average time per rep
        mean_dt = float(np.mean(per_rep_times))
        if mean_dt > 0:
            avg_speed = squeeze_threshold_m / mean_dt

    return HandSqueezeResult(
        rep_count=rep_count,
        total_time_seconds=total_time,
        per_rep_times=per_rep_times,
        rep_trigger_times=rep_trigger_times,
        rep_min_distances=rep_min_distances,
        avg_squeeze_distance_m=avg_squeeze,
        avg_squeeze_speed_mps=avg_speed,
        squeeze_threshold_m=squeeze_threshold_m,
    )
