"""
Tandem stance (feet in line) analysis — balance hold duration.

Algorithm:
  1. Signal = hip midpoint horizontal position (X, Y) in body-frame meters.
  2. Compute rolling horizontal sway (std dev of the XY position over a
     1-second window).
  3. A "stable" frame is one where sway is below a user-chosen threshold.
  4. Find the longest continuous run of stable frames → that's the hold.
  5. Report the hold duration (the main outcome metric).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._rep_common import smooth, interpolate_nans


@dataclass(frozen=True)
class TandemStanceResult:
    hold_seconds: float                 # longest stable hold
    total_seconds: float                # total recording length
    stability_fraction: float           # fraction of frames below sway threshold
    max_sway_m: float                   # largest sway value seen
    sway_threshold_m: float
    # For plotting:
    times: np.ndarray
    sway: np.ndarray
    hold_start: float | None
    hold_end: float | None

    def to_dict(self) -> dict:
        return {
            "hold_seconds": self.hold_seconds,
            "total_seconds": self.total_seconds,
            "stability_fraction": self.stability_fraction,
            "max_sway_m": self.max_sway_m,
            "sway_threshold_m": self.sway_threshold_m,
            "hold_start": self.hold_start,
            "hold_end": self.hold_end,
        }


def _rolling_std_2d(xy: np.ndarray, window: int) -> np.ndarray:
    """Rolling std deviation magnitude of a (N, 2) signal over `window` frames."""
    n = len(xy)
    out = np.zeros(n)
    if n < 2:
        return out
    half = max(1, window // 2)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = xy[lo:hi]
        if len(seg) < 2:
            out[i] = 0.0
            continue
        mean = np.mean(seg, axis=0)
        diff = seg - mean
        out[i] = float(np.mean(np.linalg.norm(diff, axis=1)))
    return out


def analyze_tandem_stance(
    hip_xy: np.ndarray,
    timestamps: np.ndarray,
    sway_threshold_m: float = 0.05,
    window_seconds: float = 1.0,
) -> TandemStanceResult:
    """
    Analyze a tandem stance hold.

    Parameters
    ----------
    hip_xy : (N, 2) array
        Hip midpoint horizontal position over time.
    timestamps : (N,) array
        Seconds aligned with hip_xy.
    sway_threshold_m : float
        Allowed mean-distance-from-centre per window for the subject to be
        considered "stable". Default 5 cm.
    window_seconds : float
        Window length for the rolling stability metric (default 1 second).
    """
    n = len(hip_xy)
    empty_times = np.asarray(timestamps)
    empty_sway = np.zeros(n)
    empty = TandemStanceResult(
        hold_seconds=0.0,
        total_seconds=float(empty_times[-1] - empty_times[0]) if n > 1 else 0.0,
        stability_fraction=0.0,
        max_sway_m=0.0,
        sway_threshold_m=sway_threshold_m,
        times=empty_times,
        sway=empty_sway,
        hold_start=None,
        hold_end=None,
    )

    if n < 3:
        return empty

    xy = np.asarray(hip_xy, dtype=float).copy()
    # Interpolate any NaNs in each column
    for col in range(xy.shape[1]):
        c = xy[:, col]
        mask = np.isnan(c)
        if np.any(mask) and not np.all(mask):
            idx = np.arange(n)
            c[mask] = np.interp(idx[mask], idx[~mask], c[~mask])
            xy[:, col] = c
    if np.all(np.isnan(xy)):
        return empty

    # Estimate frame rate to convert the window length to samples
    dts = np.diff(timestamps)
    dt = float(np.median(dts)) if len(dts) > 0 else (1.0 / 30.0)
    window_frames = max(2, int(round(window_seconds / max(dt, 1e-3))))

    sway = _rolling_std_2d(xy, window_frames)
    max_sway = float(np.max(sway))

    stable = sway < sway_threshold_m
    stability_frac = float(np.mean(stable)) if n > 0 else 0.0

    # Longest continuous run of stable frames
    best_len = 0
    best_start = best_end = None
    cur_len = 0
    cur_start = None
    for i, s in enumerate(stable):
        if s:
            if cur_start is None:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_start, best_end = cur_start, i - 1
            cur_len = 0
            cur_start = None
    if cur_len > best_len:
        best_len = cur_len
        best_start, best_end = cur_start, n - 1

    hold_seconds = 0.0
    hold_start_t = None
    hold_end_t = None
    if best_start is not None and best_end is not None and best_len >= 2:
        hold_start_t = float(timestamps[best_start])
        hold_end_t = float(timestamps[best_end])
        hold_seconds = hold_end_t - hold_start_t

    total_seconds = float(timestamps[-1] - timestamps[0])

    return TandemStanceResult(
        hold_seconds=hold_seconds,
        total_seconds=total_seconds,
        stability_fraction=stability_frac,
        max_sway_m=max_sway,
        sway_threshold_m=sway_threshold_m,
        times=np.asarray(timestamps),
        sway=sway,
        hold_start=hold_start_t,
        hold_end=hold_end_t,
    )
