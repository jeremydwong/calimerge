"""
Pullup analysis — rep counting from head (nose) height.

Algorithm:
  1. Smooth the head Z time series (in body-centred meters).
  2. Count UPWARD crossings of a user-chosen threshold — each time the head
     rises through the threshold is one completed pullup (chin over bar).
  3. Within each rep window, report the maximum head height (fullest
     extension at the top) and the minimum (hanging position) so the range
     of motion can be displayed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._rep_common import smooth, find_upward_crossings, interpolate_nans


@dataclass(frozen=True)
class PullupResult:
    rep_count: int
    total_time_seconds: float
    per_rep_times: list[float]
    rep_peak_times: list[float]          # time of each rep's maximum
    rep_peak_heights: list[float]
    rep_min_heights: list[float]
    avg_range_m: float
    top_threshold_m: float

    def to_dict(self) -> dict:
        return {
            "rep_count": self.rep_count,
            "total_time_seconds": self.total_time_seconds,
            "per_rep_times": self.per_rep_times,
            "rep_peak_times": self.rep_peak_times,
            "rep_peak_heights": self.rep_peak_heights,
            "rep_min_heights": self.rep_min_heights,
            "avg_range_m": self.avg_range_m,
            "top_threshold_m": self.top_threshold_m,
        }


def analyze_pullup(
    head_z: np.ndarray,
    timestamps: np.ndarray,
    top_threshold_m: float = 1.80,
    smooth_window: int = 5,
    min_gap_frames: int = 10,
) -> PullupResult:
    """
    Count pullup reps from a head midpoint (nose) height time series.

    Parameters
    ----------
    head_z : (N,) array
        Head vertical position in body-frame meters (0 = floor).
        NaNs are interpolated.
    timestamps : (N,) array
        Seconds aligned with head_z.
    top_threshold_m : float
        Height above which the subject has pulled themselves up over the bar.
        An upward crossing through this threshold counts as one rep.
    smooth_window : int
        Moving-average window for smoothing.
    min_gap_frames : int
        Minimum frames between successive reps.
    """
    empty = PullupResult(
        rep_count=0, total_time_seconds=0.0, per_rep_times=[],
        rep_peak_times=[], rep_peak_heights=[], rep_min_heights=[],
        avg_range_m=0.0, top_threshold_m=top_threshold_m,
    )

    if len(head_z) < 3:
        return empty

    z = interpolate_nans(np.asarray(head_z, dtype=float))
    if np.all(np.isnan(z)):
        return empty

    smoothed = smooth(z, smooth_window)
    crossings = find_upward_crossings(smoothed, top_threshold_m,
                                       min_gap=min_gap_frames)

    if len(crossings) == 0:
        return empty

    rep_count = len(crossings)

    per_rep_times = []
    for i in range(1, len(crossings)):
        dt = float(timestamps[crossings[i]] - timestamps[crossings[i - 1]])
        per_rep_times.append(dt)

    rep_peak_times = []
    rep_peak_heights = []
    rep_min_heights = []
    for i, start_idx in enumerate(crossings):
        if i + 1 < len(crossings):
            end_idx = crossings[i + 1]
        else:
            end_idx = len(smoothed)
        segment = smoothed[start_idx:end_idx]
        if len(segment) == 0:
            continue
        peak_offset = int(np.argmax(segment))
        peak_idx = start_idx + peak_offset
        rep_peak_times.append(float(timestamps[peak_idx]))
        rep_peak_heights.append(float(smoothed[peak_idx]))
        rep_min_heights.append(float(np.min(segment)))

    total_time = 0.0
    if len(crossings) >= 2:
        total_time = float(timestamps[crossings[-1]] - timestamps[crossings[0]])

    avg_range = 0.0
    if rep_peak_heights and rep_min_heights:
        avg_range = float(np.mean(
            [mx - mn for mx, mn in zip(rep_peak_heights, rep_min_heights)]
        ))

    return PullupResult(
        rep_count=rep_count,
        total_time_seconds=total_time,
        per_rep_times=per_rep_times,
        rep_peak_times=rep_peak_times,
        rep_peak_heights=rep_peak_heights,
        rep_min_heights=rep_min_heights,
        avg_range_m=avg_range,
        top_threshold_m=top_threshold_m,
    )
