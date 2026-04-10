"""
Sit-to-stand analysis — rep counting and work/power estimation.

Algorithm: threshold-based rep counting.
- User specifies a "seated" hip height threshold (default 0.65 m).
- A rep is counted each time the hip crosses UPWARD through the threshold
  (sit → stand transition).
- Time starts on the first upward crossing.
- Requires the input hip_z to be in body-centred coordinates (Z = up from floor).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SitToStandResult:
    """Results from sit-to-stand analysis."""
    rep_count: int
    total_time_seconds: float
    per_rep_times: list[float]           # time between successive upward crossings
    rep_peak_times: list[float]          # times of the max hip Z within each rep
    rep_peak_heights: list[float]        # max hip Z within each rep
    com_displacement_m: float            # avg peak - threshold
    work_per_rep_joules: float | None    # requires mass
    avg_power_watts: float | None        # requires mass
    seated_threshold_m: float            # threshold used

    def to_dict(self) -> dict:
        return {
            "rep_count": self.rep_count,
            "total_time_seconds": self.total_time_seconds,
            "per_rep_times": self.per_rep_times,
            "rep_peak_times": self.rep_peak_times,
            "rep_peak_heights": self.rep_peak_heights,
            "com_displacement_m": self.com_displacement_m,
            "work_per_rep_joules": self.work_per_rep_joules,
            "avg_power_watts": self.avg_power_watts,
            "seated_threshold_m": self.seated_threshold_m,
        }


def _smooth(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(signal) < window:
        return signal
    kernel = np.ones(window) / window
    padded = np.pad(signal, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(signal)]


def analyze_sit_to_stand(
    hip_z: np.ndarray,
    timestamps: np.ndarray,
    mass_kg: float | None = None,
    seated_threshold_m: float = 0.65,
    smooth_window: int = 5,
) -> SitToStandResult:
    """
    Analyze sit-to-stand repetitions from hip midpoint Z trajectory.

    Algorithm:
      1. Smooth hip_z to reduce noise.
      2. Compute sign of (hip_z - threshold):
           -1 = seated (below threshold)
           +1 = standing (above threshold)
      3. Count sign-change events where sign flips from -1 to +1 (upward crossing).
         Each such event is one rep (the user stood up from seated).
      4. Time the trial from the first upward crossing.
      5. Between successive upward crossings, find the peak hip Z
         (the highest point reached during the standing portion of that rep).

    Parameters
    ----------
    hip_z : (N,) array
        Vertical position of hip midpoint over time (meters, in body frame).
    timestamps : (N,) array
        Corresponding timestamps in seconds.
    mass_kg : optional
        Subject mass for work/power calculation.
    seated_threshold_m : float
        Hip height below which the subject is considered seated. Default 0.65 m.
    smooth_window : int
        Moving average window for smoothing.

    Returns
    -------
    SitToStandResult with rep count, timing, and optionally work/power.
    """
    empty = SitToStandResult(
        rep_count=0, total_time_seconds=0.0, per_rep_times=[],
        rep_peak_times=[], rep_peak_heights=[],
        com_displacement_m=0.0, work_per_rep_joules=None, avg_power_watts=None,
        seated_threshold_m=seated_threshold_m,
    )

    if len(hip_z) < 3:
        return empty

    smoothed = _smooth(hip_z, smooth_window)

    # Sign relative to threshold: -1 = seated, +1 = standing
    above = smoothed > seated_threshold_m
    # Upward crossings: was below, now above
    # np.diff gives True where state changed; combine with direction
    crossings_up = []
    for i in range(1, len(above)):
        if above[i] and not above[i - 1]:
            crossings_up.append(i)

    if len(crossings_up) == 0:
        return empty

    # Time starts at the first upward crossing (first stand-up)
    t0 = timestamps[crossings_up[0]]

    # Each upward crossing is one rep
    rep_count = len(crossings_up)

    # Per-rep timing: time between successive crossings
    # The first rep has no "previous" — we report N-1 inter-crossing times
    per_rep_times = []
    for i in range(1, len(crossings_up)):
        dt = timestamps[crossings_up[i]] - timestamps[crossings_up[i - 1]]
        per_rep_times.append(float(dt))

    # For each rep window (between successive crossings), find:
    #   - the peak hip Z (standing position)
    #   - the minimum hip Z in that window (actual seated position)
    # The per-rep displacement is peak - min, independent of the user-chosen threshold.
    rep_peak_times = []
    rep_peak_heights = []
    rep_min_heights = []
    rep_displacements = []
    for i, start_idx in enumerate(crossings_up):
        if i + 1 < len(crossings_up):
            end_idx = crossings_up[i + 1]
        else:
            end_idx = len(smoothed)
        segment = smoothed[start_idx:end_idx]
        if len(segment) == 0:
            continue
        peak_offset = int(np.argmax(segment))
        peak_idx = start_idx + peak_offset
        rep_peak_times.append(float(timestamps[peak_idx]))
        rep_peak_heights.append(float(smoothed[peak_idx]))

        min_val = float(np.min(segment))
        rep_min_heights.append(min_val)
        rep_displacements.append(float(smoothed[peak_idx]) - min_val)

    # Total trial time: from first crossing to last peak
    if rep_peak_times:
        total_time = rep_peak_times[-1] - t0
    else:
        total_time = 0.0

    # Average displacement (peak - valley) per rep — independent of threshold
    if rep_displacements:
        avg_displacement = float(np.mean(rep_displacements))
    else:
        avg_displacement = 0.0

    # Work and power (if mass available)
    work_per_rep = None
    avg_power = None
    if mass_kg is not None and mass_kg > 0 and avg_displacement > 0:
        g = 9.81
        work_per_rep = mass_kg * g * avg_displacement
        if per_rep_times:
            avg_rep_time = float(np.mean(per_rep_times))
            if avg_rep_time > 0:
                avg_power = work_per_rep / avg_rep_time

    return SitToStandResult(
        rep_count=rep_count,
        total_time_seconds=float(total_time),
        per_rep_times=per_rep_times,
        rep_peak_times=rep_peak_times,
        rep_peak_heights=rep_peak_heights,
        com_displacement_m=avg_displacement,
        work_per_rep_joules=float(work_per_rep) if work_per_rep is not None else None,
        avg_power_watts=float(avg_power) if avg_power is not None else None,
        seated_threshold_m=seated_threshold_m,
    )
