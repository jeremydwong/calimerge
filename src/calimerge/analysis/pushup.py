"""
Pushup analysis — rep counting from shoulder height.

Algorithm:
  1. Smooth the shoulder midpoint Z signal (in body-centred coordinates, so
     Z is height above the floor).
  2. Count DOWNWARD crossings of a user-chosen threshold — each time the
     shoulder drops below the threshold is one pushup (chest reached the
     bottom).
  3. Within each rep window (between successive downward crossings) report
     the minimum shoulder height reached and the maximum, so the range of
     motion can be displayed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._rep_common import smooth, find_downward_crossings, interpolate_nans


@dataclass(frozen=True)
class PushupResult:
    rep_count: int
    total_time_seconds: float
    per_rep_times: list[float]           # time between successive downward crossings
    rep_min_times: list[float]           # time at which each rep's minimum occurs
    rep_min_heights: list[float]         # bottom of each rep (m)
    rep_max_heights: list[float]         # top of each rep (m)
    rep_trigger_times: list[float]       # time at which each rep was triggered (cross)
    avg_range_m: float                   # mean (max - min) per rep
    top_threshold_m: float               # threshold used (shoulder height)

    def to_dict(self) -> dict:
        return {
            "rep_count": self.rep_count,
            "total_time_seconds": self.total_time_seconds,
            "per_rep_times": self.per_rep_times,
            "rep_min_times": self.rep_min_times,
            "rep_min_heights": self.rep_min_heights,
            "rep_max_heights": self.rep_max_heights,
            "rep_trigger_times": self.rep_trigger_times,
            "avg_range_m": self.avg_range_m,
            "top_threshold_m": self.top_threshold_m,
        }


def analyze_pushup(
    shoulder_z: np.ndarray,
    timestamps: np.ndarray,
    top_threshold_m: float = 0.30,
    smooth_window: int = 5,
    min_gap_frames: int = 10,
) -> PushupResult:
    """
    Count pushup reps from a shoulder midpoint height time series.

    Parameters
    ----------
    shoulder_z : (N,) array
        Shoulder midpoint vertical position in body-frame meters
        (0 = floor). NaNs are interpolated.
    timestamps : (N,) array
        Seconds aligned with shoulder_z.
    top_threshold_m : float
        Height below which the subject is considered to be at the bottom
        of a pushup. A downward crossing through this threshold counts
        as one rep.
    smooth_window : int
        Moving-average window for smoothing.
    min_gap_frames : int
        Minimum frames between successive reps (anti-noise).
    """
    empty = PushupResult(
        rep_count=0, total_time_seconds=0.0, per_rep_times=[],
        rep_min_times=[], rep_min_heights=[], rep_max_heights=[],
        rep_trigger_times=[],
        avg_range_m=0.0, top_threshold_m=top_threshold_m,
    )

    if len(shoulder_z) < 3:
        return empty

    z = interpolate_nans(np.asarray(shoulder_z, dtype=float))
    if np.all(np.isnan(z)):
        return empty

    smoothed = smooth(z, smooth_window)
    crossings = find_downward_crossings(smoothed, top_threshold_m,
                                        min_gap=min_gap_frames)

    if len(crossings) == 0:
        return empty

    rep_count = len(crossings)

    per_rep_times = []
    for i in range(1, len(crossings)):
        dt = float(timestamps[crossings[i]] - timestamps[crossings[i - 1]])
        per_rep_times.append(dt)

    rep_min_times = []
    rep_min_heights = []
    rep_max_heights = []
    rep_trigger_times = []
    for i, start_idx in enumerate(crossings):
        if i + 1 < len(crossings):
            end_idx = crossings[i + 1]
        else:
            end_idx = len(smoothed)
        segment = smoothed[start_idx:end_idx]
        if len(segment) == 0:
            continue
        min_offset = int(np.argmin(segment))
        min_idx = start_idx + min_offset
        rep_min_times.append(float(timestamps[min_idx]))
        rep_min_heights.append(float(np.min(segment)))
        rep_max_heights.append(float(np.max(segment)))
        rep_trigger_times.append(float(timestamps[start_idx]))

    total_time = 0.0
    if len(crossings) >= 2:
        total_time = float(timestamps[crossings[-1]] - timestamps[crossings[0]])

    avg_range = 0.0
    if rep_min_heights and rep_max_heights:
        avg_range = float(np.mean(
            [mx - mn for mx, mn in zip(rep_max_heights, rep_min_heights)]
        ))

    return PushupResult(
        rep_count=rep_count,
        total_time_seconds=total_time,
        per_rep_times=per_rep_times,
        rep_min_times=rep_min_times,
        rep_min_heights=rep_min_heights,
        rep_max_heights=rep_max_heights,
        rep_trigger_times=rep_trigger_times,
        avg_range_m=avg_range,
        top_threshold_m=top_threshold_m,
    )
