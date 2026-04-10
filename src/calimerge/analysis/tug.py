"""
Timed Up and Go (TUG) analysis.

Algorithm:
  1. Reject initial frames before the subject is seated (hip_z < seated_threshold).
     Only data after the first "valid seated" frame is considered.
  2. Compute head velocity magnitude (speed) over time from a head keypoint.
  3. Smooth the speed signal.
  4. Start of TUG = first frame AFTER valid seated where head_speed > speed_threshold.
     (The moment they start moving to stand up.)
  5. End of TUG = last frame where head_speed > speed_threshold, or equivalently
     the last frame before head speed drops below threshold for good.
     (The moment they stop moving after sitting back down.)
  6. Duration = t_end - t_start.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TugResult:
    """Results from a TUG analysis."""
    duration_seconds: float
    start_time: float
    end_time: float
    max_head_speed: float
    start_valid: bool                    # True if a valid seated period was found
    seated_threshold_m: float
    speed_threshold_mps: float
    # Per-sample arrays for plotting
    times: np.ndarray
    head_speed: np.ndarray               # smoothed head speed over time
    hip_z: np.ndarray                    # hip midpoint Z over time (aligned with times)

    def to_dict(self) -> dict:
        return {
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "max_head_speed": self.max_head_speed,
            "start_valid": self.start_valid,
            "seated_threshold_m": self.seated_threshold_m,
            "speed_threshold_mps": self.speed_threshold_mps,
        }


def _smooth(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(signal) < window:
        return signal
    kernel = np.ones(window) / window
    padded = np.pad(signal, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(signal)]


def _compute_head_speed(head_xyz: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Compute magnitude of head velocity from a (N, 3) position array.

    Returns a (N,) array; first sample uses forward difference.
    NaNs are propagated — missing frames produce NaN speed samples.
    """
    n = len(head_xyz)
    if n < 2:
        return np.zeros(n)
    speed = np.zeros(n)
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            speed[i] = 0.0
            continue
        dp = head_xyz[i] - head_xyz[i - 1]
        if np.isnan(dp).any():
            speed[i] = np.nan
            continue
        speed[i] = float(np.linalg.norm(dp) / dt)
    speed[0] = speed[1] if n > 1 else 0.0
    return speed


def analyze_tug(
    hip_z: np.ndarray,
    head_xyz: np.ndarray,
    timestamps: np.ndarray,
    seated_threshold_m: float = 0.65,
    speed_threshold_mps: float = 0.3,
    smooth_window: int = 7,
) -> TugResult:
    """
    Analyze a TUG recording.

    Parameters
    ----------
    hip_z : (N,) array
        Hip midpoint vertical position in the body frame (meters).
    head_xyz : (N, 3) array
        Head 3D position over time in the body frame. Used to compute speed.
    timestamps : (N,) array
        Seconds since recording start, aligned with hip_z and head_xyz.
    seated_threshold_m : float
        Hip height below which the subject is considered seated.
        Used only to reject initial "garbage" frames before the subject is ready.
    speed_threshold_mps : float
        Head speed threshold for detecting the start and end of movement.
    smooth_window : int
        Moving-average window for both hip and speed signals.

    Returns
    -------
    TugResult with duration, start/end times, and the smoothed traces.
    """
    n = len(hip_z)
    empty = TugResult(
        duration_seconds=0.0,
        start_time=0.0,
        end_time=0.0,
        max_head_speed=0.0,
        start_valid=False,
        seated_threshold_m=seated_threshold_m,
        speed_threshold_mps=speed_threshold_mps,
        times=np.asarray(timestamps),
        head_speed=np.zeros(n),
        hip_z=np.asarray(hip_z),
    )

    if n < 3 or len(head_xyz) != n:
        return empty

    hip_smoothed = _smooth(hip_z, smooth_window)

    # 1. Find first frame where hip is clearly seated (reject garbage)
    seated_mask = hip_smoothed < seated_threshold_m
    if not np.any(seated_mask):
        return empty
    first_seated = int(np.argmax(seated_mask))  # first True index

    # 2. Compute head speed
    raw_speed = _compute_head_speed(np.asarray(head_xyz), timestamps)
    # Interpolate NaNs
    speed = raw_speed.copy()
    nan_mask = np.isnan(speed)
    if np.any(nan_mask) and not np.all(nan_mask):
        idx = np.arange(n)
        speed[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], speed[~nan_mask])
    elif np.all(nan_mask):
        return empty
    speed = _smooth(speed, smooth_window)

    # 3. Start of TUG: first frame at or after first_seated where speed exceeds threshold
    above = speed[first_seated:] > speed_threshold_mps
    if not np.any(above):
        return TugResult(
            duration_seconds=0.0,
            start_time=float(timestamps[first_seated]),
            end_time=float(timestamps[first_seated]),
            max_head_speed=float(np.nanmax(speed) if n > 0 else 0.0),
            start_valid=False,
            seated_threshold_m=seated_threshold_m,
            speed_threshold_mps=speed_threshold_mps,
            times=np.asarray(timestamps),
            head_speed=speed,
            hip_z=hip_smoothed,
        )

    start_offset = int(np.argmax(above))
    start_idx = first_seated + start_offset

    # 4. End of TUG: last frame where speed is above threshold (walking back, sitting down)
    # Search from the end backwards for the last index above threshold.
    above_all = speed > speed_threshold_mps
    if not np.any(above_all[start_idx:]):
        end_idx = start_idx
    else:
        # Last True in the array from start_idx onward
        end_idx = start_idx + int(n - start_idx - 1 - np.argmax(above_all[start_idx:][::-1]))

    start_time = float(timestamps[start_idx])
    end_time = float(timestamps[end_idx])
    duration = max(0.0, end_time - start_time)

    return TugResult(
        duration_seconds=duration,
        start_time=start_time,
        end_time=end_time,
        max_head_speed=float(np.nanmax(speed)),
        start_valid=True,
        seated_threshold_m=seated_threshold_m,
        speed_threshold_mps=speed_threshold_mps,
        times=np.asarray(timestamps),
        head_speed=speed,
        hip_z=hip_smoothed,
    )
