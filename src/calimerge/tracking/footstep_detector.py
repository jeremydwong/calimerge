"""
Footstep placement detection.

Detects discrete "placement" events from a stream of 3D ankle positions
in body coordinates (z = body-up, z=0 is the floor after the user has
zeroed origin at L_Ankle).

Algorithm
---------
A placement is emitted when the foot transitions from a "moving"
state (recent vertical speed above MOVING_VZ_THRESHOLD sustained for
MOVING_FRAME_COUNT frames) into a "rested-near-floor" state
(|vz| < REST_VZ_THRESHOLD AND |z| < FLOOR_Z_THRESHOLD).

Two placements occurring within DEBOUNCE_SECONDS are coalesced into one
(only the first is kept). This rejects the high-frequency wobble that
happens when the foot first contacts the floor.

The detector is purposefully a small, plain Python class so it can be
tested without any GUI / Qt / OpenCV dependencies.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

# ── Tuneable thresholds ──────────────────────────────────────────────
#
# Vertical speed above which the foot counts as "moving". Tuned for a
# walking/squatting cadence — when the foot is in the air during a step
# its vertical speed easily exceeds 0.05 m/s. (m/s)
MOVING_VZ_THRESHOLD = 0.05

# How many consecutive frames must show |vz| > MOVING_VZ_THRESHOLD
# before we're convinced the foot is in motion. Prevents triggering
# on a single noisy spike when the foot is otherwise still.
MOVING_FRAME_COUNT = 3

# Vertical speed below which the foot is considered "at rest". Must be
# noticeably tighter than MOVING_VZ_THRESHOLD so the moving→rest
# transition is unambiguous. (m/s)
REST_VZ_THRESHOLD = 0.02

# Maximum height above the zeroed floor at which a rest event still
# counts as a placement. 10 cm absorbs ankle-marker noise + foot
# thickness without admitting a foot held in the air. (m)
FLOOR_Z_THRESHOLD = 0.10

# Placements within this window are merged into one. Suppresses
# intra-impact micro-bounces. (s)
DEBOUNCE_SECONDS = 0.20


@dataclass(frozen=True)
class FootPlacement:
    """A single foot-placement event in body coordinates."""

    t: float        # seconds since detector start
    x: float
    y: float
    z: float


class FootstepDetector:
    """Streaming detector for one foot.

    Feed it (timestamp, position) samples in order; it emits a
    `FootPlacement` whenever the foot finishes a movement and rests
    near the floor.

    Body-coordinate convention (post "Zero at L_Ankle"):
        x = foot-to-foot, y = forward, z = body-up (floor at z=0).
    """

    def __init__(
        self,
        moving_vz_threshold: float = MOVING_VZ_THRESHOLD,
        moving_frame_count: int = MOVING_FRAME_COUNT,
        rest_vz_threshold: float = REST_VZ_THRESHOLD,
        floor_z_threshold: float = FLOOR_Z_THRESHOLD,
        debounce_seconds: float = DEBOUNCE_SECONDS,
    ):
        self._move_vz = moving_vz_threshold
        self._move_n = moving_frame_count
        self._rest_vz = rest_vz_threshold
        self._floor_z = floor_z_threshold
        self._debounce = debounce_seconds

        # State
        self._prev: Optional[Tuple[float, float, float, float]] = None  # (t, x, y, z)
        self._moving_streak: int = 0
        self._is_moving: bool = False
        self._last_placement_t: float = -math.inf

    def reset(self) -> None:
        """Clear all detector state."""
        self._prev = None
        self._moving_streak = 0
        self._is_moving = False
        self._last_placement_t = -math.inf

    def update(
        self, t: float, x: float, y: float, z: float
    ) -> Optional[FootPlacement]:
        """Submit one (timestamp, position) sample.

        Returns a FootPlacement on the frame the placement is detected,
        else None.
        """
        if any(math.isnan(v) for v in (t, x, y, z)):
            return None

        if self._prev is None:
            self._prev = (t, x, y, z)
            return None

        pt, px, py, pz = self._prev
        dt = t - pt
        if dt <= 0:
            # Duplicate / out-of-order sample: keep state; replace prev.
            self._prev = (t, x, y, z)
            return None

        vz = (z - pz) / dt
        abs_vz = abs(vz)

        # Track the moving streak.
        if abs_vz > self._move_vz:
            self._moving_streak += 1
            if self._moving_streak >= self._move_n:
                self._is_moving = True
        else:
            self._moving_streak = 0

        placement: Optional[FootPlacement] = None

        # Detect rest-near-floor after a sustained motion.
        if (
            self._is_moving
            and abs_vz < self._rest_vz
            and abs(z) < self._floor_z
        ):
            if (t - self._last_placement_t) >= self._debounce:
                placement = FootPlacement(t=t, x=x, y=y, z=z)
                self._last_placement_t = t
            # Either way, the motion phase is over.
            self._is_moving = False
            self._moving_streak = 0

        self._prev = (t, x, y, z)
        return placement


class TwoFootTracker:
    """Convenience: independent left/right detectors + bounded history.

    The skeleton view holds the rolling deques; this class is mainly
    for tests and any future non-GUI consumer.
    """

    def __init__(self, history: int = 10):
        self.left = FootstepDetector()
        self.right = FootstepDetector()
        self.left_history: Deque[FootPlacement] = deque(maxlen=history)
        self.right_history: Deque[FootPlacement] = deque(maxlen=history)

    def update_left(self, t: float, x: float, y: float, z: float) -> Optional[FootPlacement]:
        ev = self.left.update(t, x, y, z)
        if ev is not None:
            self.left_history.append(ev)
        return ev

    def update_right(self, t: float, x: float, y: float, z: float) -> Optional[FootPlacement]:
        ev = self.right.update(t, x, y, z)
        if ev is not None:
            self.right_history.append(ev)
        return ev

    def reset(self) -> None:
        self.left.reset()
        self.right.reset()
        self.left_history.clear()
        self.right_history.clear()
