"""
Stretch analysis — simple timed hold.

No rep counting. We report the duration the subject was active in the
recording (total captured time) and the hip vertical stability as a
quality metric. A good stretch is held steady; excessive movement lowers
the "steadiness" score but does not invalidate the recording.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._rep_common import smooth


@dataclass(frozen=True)
class StretchResult:
    hold_seconds: float               # total recording duration
    steadiness: float                 # 0-1: how steady the pose was
    max_range_m: float                # max hip-z range observed

    # For plotting
    times: np.ndarray
    hip_z: np.ndarray

    def to_dict(self) -> dict:
        return {
            "hold_seconds": self.hold_seconds,
            "steadiness": self.steadiness,
            "max_range_m": self.max_range_m,
        }


def analyze_stretch(
    hip_z: np.ndarray,
    timestamps: np.ndarray,
) -> StretchResult:
    """
    Simple stretch analysis.

    Reports how long the stretch was held and a steadiness score based on
    the hip Z signal's variance. Low variance = steady hold = higher score.
    """
    n = len(hip_z)
    if n < 2:
        return StretchResult(
            hold_seconds=0.0, steadiness=0.0, max_range_m=0.0,
            times=np.asarray(timestamps), hip_z=np.asarray(hip_z),
        )

    hip = np.asarray(hip_z, dtype=float)
    mask = ~np.isnan(hip)
    if not np.any(mask):
        return StretchResult(
            hold_seconds=0.0, steadiness=0.0, max_range_m=0.0,
            times=np.asarray(timestamps), hip_z=hip,
        )

    smoothed = smooth(hip, window=5)
    z_range = float(np.nanmax(smoothed) - np.nanmin(smoothed))

    # Steadiness: 1 - normalized std (clipped to [0, 1])
    std = float(np.nanstd(smoothed))
    # 5 cm std = score 0; 0 cm std = score 1 (linear interpolation)
    steadiness = max(0.0, 1.0 - (std / 0.05))
    steadiness = min(1.0, steadiness)

    hold_seconds = float(timestamps[-1] - timestamps[0])

    return StretchResult(
        hold_seconds=hold_seconds,
        steadiness=steadiness,
        max_range_m=z_range,
        times=np.asarray(timestamps),
        hip_z=smoothed,
    )
