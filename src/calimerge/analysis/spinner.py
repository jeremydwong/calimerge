"""Spinner — single-trial head-motion tracker.

There's no rep counting and no threshold. The subject does whatever the
trial is (e.g. spin in place) and we just summarise how the head moved:

- duration_seconds: trial length from first to last head sample
- path_length_m:    total distance the head travelled (sum of frame-to-
                    frame distances), a useful "how much motion" number
- max_speed_mps:    peak instantaneous head speed (smoothed)
- mean_speed_mps:   path_length / duration
- range_x_m / range_y_m / range_z_m:
                    bounding-box extents (max-min on each axis)

We work on the **nose** keypoint only because it's the most reliable head
position from VitPose-Base / SynthPose. The other head landmarks (eyes,
ears) are written to the npz alongside, so the notebook can plot them
without re-running detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpinnerResult:
    duration_seconds: float
    path_length_m: float
    max_speed_mps: float
    mean_speed_mps: float
    range_x_m: float
    range_y_m: float
    range_z_m: float

    # For plotting: the (smoothed) head trace plus instantaneous speed.
    times: np.ndarray
    head_xyz: np.ndarray              # (N, 3) nose position over time
    speed_mps: np.ndarray             # (N,) instantaneous speed, NaN at first sample

    def to_dict(self) -> dict:
        return {
            "duration_seconds": self.duration_seconds,
            "path_length_m": self.path_length_m,
            "max_speed_mps": self.max_speed_mps,
            "mean_speed_mps": self.mean_speed_mps,
            "range_x_m": self.range_x_m,
            "range_y_m": self.range_y_m,
            "range_z_m": self.range_z_m,
        }


def analyze_spinner(
    head_xyz: np.ndarray,
    timestamps: np.ndarray,
    smooth_window: int = 5,
) -> SpinnerResult:
    """Summarise head motion across a single trial.

    Parameters
    ----------
    head_xyz : (N, 3) array
        Per-frame nose position. NaN rows (frames with no detection) are
        skipped for path-length / speed but the input arrays are
        preserved for plotting.
    timestamps : (N,) array
        Monotonically-increasing seconds since recording start.
    smooth_window : int
        Boxcar window length for speed smoothing — knocks down per-
        frame jitter without changing the path length integral.

    Returns
    -------
    SpinnerResult — dataclass; ``to_dict()`` for downstream consumers.
    """
    times = np.asarray(timestamps, dtype=float)
    head = np.asarray(head_xyz, dtype=float).reshape(-1, 3)
    n = len(times)

    if n < 2 or len(head) < 2:
        return SpinnerResult(
            duration_seconds=0.0, path_length_m=0.0,
            max_speed_mps=0.0, mean_speed_mps=0.0,
            range_x_m=0.0, range_y_m=0.0, range_z_m=0.0,
            times=times, head_xyz=head,
            speed_mps=np.full(n, np.nan),
        )

    duration = float(times[-1] - times[0])

    # Path length and speed — only over rows where the nose was actually
    # detected. Skipped frames don't contribute distance, and the speed
    # at a "first valid frame after a gap" uses the gap's elapsed time
    # so we don't divide by ~0 on consecutive valid samples after a long
    # NaN run.
    valid = ~np.isnan(head).any(axis=1)
    valid_idx = np.where(valid)[0]

    speeds = np.full(n, np.nan)
    path_length = 0.0
    if len(valid_idx) >= 2:
        for prev_i, cur_i in zip(valid_idx[:-1], valid_idx[1:]):
            dx = head[cur_i] - head[prev_i]
            dt = times[cur_i] - times[prev_i]
            d = float(np.linalg.norm(dx))
            path_length += d
            if dt > 0:
                speeds[cur_i] = d / dt

        # Smooth the speed to suppress per-frame VitPose jitter without
        # touching the path length integral.
        speeds_finite = np.where(np.isfinite(speeds), speeds, np.nan)
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            # Convolve only over finite values; pad NaNs as zero with a
            # mask, then divide by the smoothed mask to restore weights.
            mask = np.isfinite(speeds_finite).astype(float)
            filled = np.where(np.isfinite(speeds_finite), speeds_finite, 0.0)
            conv = np.convolve(filled, kernel, mode="same")
            mw = np.convolve(mask, kernel, mode="same")
            with np.errstate(invalid="ignore", divide="ignore"):
                speeds = np.where(mw > 0, conv / mw, np.nan)

    max_speed = float(np.nanmax(speeds)) if np.any(np.isfinite(speeds)) else 0.0
    mean_speed = (path_length / duration) if duration > 0 else 0.0

    if np.any(valid):
        rng = np.nanmax(head[valid], axis=0) - np.nanmin(head[valid], axis=0)
        range_x, range_y, range_z = (float(rng[0]), float(rng[1]), float(rng[2]))
    else:
        range_x = range_y = range_z = 0.0

    return SpinnerResult(
        duration_seconds=duration,
        path_length_m=path_length,
        max_speed_mps=max_speed,
        mean_speed_mps=mean_speed,
        range_x_m=range_x,
        range_y_m=range_y,
        range_z_m=range_z,
        times=times,
        head_xyz=head,
        speed_mps=speeds,
    )
