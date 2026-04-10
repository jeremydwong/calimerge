"""
Leg raise analysis — rep counting from knee height.

Algorithm:
  1. Signal = max(L_knee_z, R_knee_z) in body-frame meters, so whichever
     leg is currently raised drives the trace.
  2. Smooth and count upward crossings of a user-chosen threshold.
  3. Each upward crossing = one rep.
  4. Within each rep window report the peak knee height.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._rep_common import smooth, find_upward_crossings, interpolate_nans


@dataclass(frozen=True)
class LegRaiseResult:
    rep_count: int
    total_time_seconds: float
    per_rep_times: list[float]
    rep_peak_times: list[float]
    rep_peak_heights: list[float]
    rep_min_heights: list[float]
    avg_range_m: float
    lift_threshold_m: float

    def to_dict(self) -> dict:
        return {
            "rep_count": self.rep_count,
            "total_time_seconds": self.total_time_seconds,
            "per_rep_times": self.per_rep_times,
            "rep_peak_times": self.rep_peak_times,
            "rep_peak_heights": self.rep_peak_heights,
            "rep_min_heights": self.rep_min_heights,
            "avg_range_m": self.avg_range_m,
            "lift_threshold_m": self.lift_threshold_m,
        }


def analyze_leg_raise(
    knee_z_max: np.ndarray,
    timestamps: np.ndarray,
    lift_threshold_m: float = 0.60,
    smooth_window: int = 5,
    min_gap_frames: int = 10,
) -> LegRaiseResult:
    empty = LegRaiseResult(
        rep_count=0, total_time_seconds=0.0, per_rep_times=[],
        rep_peak_times=[], rep_peak_heights=[], rep_min_heights=[],
        avg_range_m=0.0, lift_threshold_m=lift_threshold_m,
    )

    if len(knee_z_max) < 3:
        return empty

    z = interpolate_nans(np.asarray(knee_z_max, dtype=float))
    if np.all(np.isnan(z)):
        return empty

    smoothed = smooth(z, smooth_window)
    crossings = find_upward_crossings(smoothed, lift_threshold_m,
                                       min_gap=min_gap_frames)
    if not crossings:
        return empty

    rep_count = len(crossings)
    per_rep_times = [
        float(timestamps[crossings[i]] - timestamps[crossings[i - 1]])
        for i in range(1, len(crossings))
    ]

    rep_peak_times = []
    rep_peak_heights = []
    rep_min_heights = []
    for i, start_idx in enumerate(crossings):
        end_idx = crossings[i + 1] if i + 1 < len(crossings) else len(smoothed)
        segment = smoothed[start_idx:end_idx]
        if len(segment) == 0:
            continue
        peak_offset = int(np.argmax(segment))
        rep_peak_times.append(float(timestamps[start_idx + peak_offset]))
        rep_peak_heights.append(float(segment[peak_offset]))
        rep_min_heights.append(float(np.min(segment)))

    total_time = 0.0
    if len(crossings) >= 2:
        total_time = float(timestamps[crossings[-1]] - timestamps[crossings[0]])

    avg_range = 0.0
    if rep_peak_heights and rep_min_heights:
        avg_range = float(np.mean(
            [mx - mn for mx, mn in zip(rep_peak_heights, rep_min_heights)]
        ))

    return LegRaiseResult(
        rep_count=rep_count,
        total_time_seconds=total_time,
        per_rep_times=per_rep_times,
        rep_peak_times=rep_peak_times,
        rep_peak_heights=rep_peak_heights,
        rep_min_heights=rep_min_heights,
        avg_range_m=avg_range,
        lift_threshold_m=lift_threshold_m,
    )
