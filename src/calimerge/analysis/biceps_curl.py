"""
Biceps curl analysis — rep counting from elbow angle trajectory.

Algorithm:
  1. Smooth the elbow angle (degrees) time series.
  2. Count upward crossings of an "extended" threshold (default 150°) —
     each upward crossing is one completed curl (flexed → back to extended).
  3. Within each rep window, find the minimum angle (deepest flex)
     and the maximum angle (fullest extension). The peak-to-valley range
     is a quality metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._rep_common import smooth, find_upward_crossings, interpolate_nans


@dataclass(frozen=True)
class BicepsResult:
    rep_count: int
    total_time_seconds: float
    per_rep_times: list[float]
    rep_min_angles: list[float]    # deepest flexion per rep (deg)
    rep_max_angles: list[float]    # max extension per rep (deg)
    rep_peak_times: list[float]    # time of the extension peak (where the rep "lands")
    avg_range_deg: float           # mean (max - min) across reps
    extended_threshold_deg: float

    def to_dict(self) -> dict:
        return {
            "rep_count": self.rep_count,
            "total_time_seconds": self.total_time_seconds,
            "per_rep_times": self.per_rep_times,
            "rep_min_angles": self.rep_min_angles,
            "rep_max_angles": self.rep_max_angles,
            "rep_peak_times": self.rep_peak_times,
            "avg_range_deg": self.avg_range_deg,
            "extended_threshold_deg": self.extended_threshold_deg,
        }


def analyze_biceps_curl(
    elbow_angles: np.ndarray,
    timestamps: np.ndarray,
    extended_threshold_deg: float = 150.0,
    smooth_window: int = 5,
    min_gap_frames: int = 10,
) -> BicepsResult:
    """
    Count biceps curl reps from an elbow-angle time series.

    Parameters
    ----------
    elbow_angles : (N,) array
        Elbow angle over time in degrees (180 = extended, 0 = flexed).
        NaNs are interpolated.
    timestamps : (N,) array
        Seconds aligned with elbow_angles.
    extended_threshold_deg : float
        Angle above which the arm is considered extended. Default 150°.
    smooth_window : int
        Moving-average window for smoothing.
    min_gap_frames : int
        Minimum frames between successive reps.
    """
    empty = BicepsResult(
        rep_count=0, total_time_seconds=0.0, per_rep_times=[],
        rep_min_angles=[], rep_max_angles=[], rep_peak_times=[],
        avg_range_deg=0.0, extended_threshold_deg=extended_threshold_deg,
    )

    if len(elbow_angles) < 3:
        return empty

    angles = interpolate_nans(np.asarray(elbow_angles, dtype=float))
    if np.all(np.isnan(angles)):
        return empty

    smoothed = smooth(angles, smooth_window)
    crossings = find_upward_crossings(smoothed, extended_threshold_deg,
                                       min_gap=min_gap_frames)

    if len(crossings) == 0:
        return empty

    rep_count = len(crossings)

    per_rep_times = []
    for i in range(1, len(crossings)):
        dt = float(timestamps[crossings[i]] - timestamps[crossings[i - 1]])
        per_rep_times.append(dt)

    rep_min_angles = []
    rep_max_angles = []
    rep_peak_times = []
    for i, start_idx in enumerate(crossings):
        if i + 1 < len(crossings):
            end_idx = crossings[i + 1]
        else:
            end_idx = len(smoothed)
        segment = smoothed[start_idx:end_idx]
        if len(segment) == 0:
            continue
        rep_min_angles.append(float(np.min(segment)))
        rep_max_angles.append(float(np.max(segment)))
        rep_peak_times.append(float(timestamps[start_idx]))

    total_time = 0.0
    if len(crossings) >= 2:
        total_time = float(timestamps[crossings[-1]] - timestamps[crossings[0]])

    avg_range = 0.0
    if rep_min_angles and rep_max_angles:
        avg_range = float(np.mean(
            [mx - mn for mx, mn in zip(rep_max_angles, rep_min_angles)]
        ))

    return BicepsResult(
        rep_count=rep_count,
        total_time_seconds=total_time,
        per_rep_times=per_rep_times,
        rep_min_angles=rep_min_angles,
        rep_max_angles=rep_max_angles,
        rep_peak_times=rep_peak_times,
        avg_range_deg=avg_range,
        extended_threshold_deg=extended_threshold_deg,
    )
