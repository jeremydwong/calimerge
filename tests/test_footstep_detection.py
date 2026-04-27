"""Tests for calimerge.tracking.footstep_detector."""

from __future__ import annotations

import math

import pytest

from calimerge.tracking.footstep_detector import (
    DEBOUNCE_SECONDS,
    FootPlacement,
    FootstepDetector,
    TwoFootTracker,
)


def _feed(detector: FootstepDetector, samples):
    """Feed (t, x, y, z) tuples and return the list of emitted placements."""
    out = []
    for (t, x, y, z) in samples:
        ev = detector.update(t, x, y, z)
        if ev is not None:
            out.append(ev)
    return out


def _step_trajectory(
    fps: float = 60.0,
    duration: float = 1.0,
    apex_height: float = 0.20,
    rest_frames: int = 30,
    apex_frac: float = 0.5,
    x: float = 0.10,
    y: float = 0.0,
):
    """Generate (t, x, y, z) samples for a single foot step.

    Foot starts at z=0, arcs up to z=apex_height, comes back to z=0,
    then stays at z=0 for `rest_frames` frames.
    """
    n_air = int(fps * duration)
    samples = []
    apex_idx = int(n_air * apex_frac)
    for i in range(n_air):
        t = i / fps
        if i <= apex_idx:
            frac = i / max(1, apex_idx)
        else:
            frac = 1.0 - (i - apex_idx) / max(1, n_air - apex_idx)
        z = apex_height * frac
        samples.append((t, x, y, z))

    # Static rest after landing
    t0 = n_air / fps
    for i in range(rest_frames):
        t = t0 + i / fps
        samples.append((t, x, y, 0.0))
    return samples


class TestSingleStep:
    def test_one_placement_emitted_at_landing(self):
        det = FootstepDetector()
        samples = _step_trajectory()
        events = _feed(det, samples)

        assert len(events) == 1, f"expected 1 placement, got {len(events)}"
        ev = events[0]
        assert isinstance(ev, FootPlacement)
        # Should land at roughly z=0
        assert abs(ev.z) < 0.10
        # Landing time should be after the apex (i.e. > half the air phase).
        # Air duration = 1.0s, apex at 0.5s. The placement fires once the
        # foot has come fully to rest on the floor, which happens shortly
        # after the air-phase ends.
        assert ev.t > 0.4
        # And not arbitrarily later — well within the rest window.
        assert ev.t < 1.2

    def test_landing_position_xy_preserved(self):
        det = FootstepDetector()
        samples = _step_trajectory(x=0.42, y=-0.13)
        events = _feed(det, samples)
        assert len(events) == 1
        assert events[0].x == pytest.approx(0.42)
        assert events[0].y == pytest.approx(-0.13)


class TestDebounce:
    def test_rapid_double_tap_emits_one(self):
        """Two placements within the debounce window should be merged."""
        det = FootstepDetector()
        # First step
        samples = _step_trajectory(duration=0.5, rest_frames=2)
        # Tiny lift + immediate rest, well within DEBOUNCE_SECONDS
        # Trajectory ends at t = 0.5 + 2/60 ≈ 0.533s.
        t_resume = samples[-1][0] + 1 / 60.0
        # Quick lift to 0.05m and back down inside debounce window
        for i, dz in enumerate([0.02, 0.04, 0.05, 0.04, 0.02, 0.0, 0.0]):
            t = t_resume + i / 60.0
            samples.append((t, 0.10, 0.0, dz))

        # Whole sequence end-to-end is ~0.65s; debounce window is 0.20s.
        # First placement should land near t≈0.5s; second wobble is at
        # t≈0.6s, inside the debounce window → suppressed.
        events = _feed(det, samples)
        assert len(events) == 1, (
            f"debounce failed: emitted {len(events)} events at "
            f"{[e.t for e in events]}"
        )

    def test_two_distinct_steps_emit_two(self):
        """Steps separated by more than debounce should both fire."""
        det = FootstepDetector()
        s1 = _step_trajectory(duration=0.6, rest_frames=30)
        # Offset second step well past the debounce window
        t_offset = s1[-1][0] + 0.5
        s2 = [(t + t_offset, x, y, z) for (t, x, y, z) in
              _step_trajectory(duration=0.6, rest_frames=30)]
        events = _feed(det, s1 + s2)
        assert len(events) == 2
        gap = events[1].t - events[0].t
        assert gap > DEBOUNCE_SECONDS


class TestStillFoot:
    def test_static_foot_emits_nothing(self):
        """A foot that never moves should never trigger a placement."""
        det = FootstepDetector()
        samples = [(i / 60.0, 0.10, 0.0, 0.0) for i in range(120)]
        events = _feed(det, samples)
        assert events == []

    def test_static_foot_in_air_emits_nothing(self):
        det = FootstepDetector()
        # Held high — never near floor
        samples = [(i / 60.0, 0.10, 0.0, 0.5) for i in range(120)]
        events = _feed(det, samples)
        assert events == []


class TestNeverRests:
    def test_perpetually_moving_foot_emits_nothing(self):
        """A foot that moves continuously and never rests should not emit."""
        det = FootstepDetector()
        # Sinusoidal vertical motion: |vz| stays well above rest threshold.
        # Amplitude 0.3m, period 1.0s → peak vz = 2π * 0.3 ≈ 1.88 m/s.
        samples = []
        for i in range(240):
            t = i / 60.0
            z = 0.3 * (1 + math.sin(2 * math.pi * t)) / 2 + 0.2
            samples.append((t, 0.0, 0.0, z))
        events = _feed(det, samples)
        assert events == []

    def test_moving_foot_far_from_floor_emits_nothing(self):
        """Foot rests but high above floor → no placement."""
        det = FootstepDetector()
        # Lift to 1.0m, then hold steady up there
        n_air = 30
        samples = []
        for i in range(n_air):
            t = i / 60.0
            z = 1.0 * (i / max(1, n_air - 1))
            samples.append((t, 0.0, 0.0, z))
        # Now hold at z=1.0
        for i in range(60):
            t = (n_air + i) / 60.0
            samples.append((t, 0.0, 0.0, 1.0))
        events = _feed(det, samples)
        assert events == []


class TestNaNHandling:
    def test_nan_samples_are_ignored(self):
        det = FootstepDetector()
        samples = _step_trajectory()
        # Inject some NaNs
        samples_with_nan = []
        for i, s in enumerate(samples):
            if i % 10 == 5:
                samples_with_nan.append((s[0], float("nan"), s[2], s[3]))
            else:
                samples_with_nan.append(s)
        events = _feed(det, samples_with_nan)
        # Should still detect exactly one placement
        assert len(events) == 1


class TestTwoFootTracker:
    def test_independent_history_per_foot(self):
        tracker = TwoFootTracker(history=10)
        for (t, x, y, z) in _step_trajectory(x=-0.10):
            tracker.update_left(t, x, y, z)
        for (t, x, y, z) in _step_trajectory(x=0.10):
            tracker.update_right(t, x, y, z)
        assert len(tracker.left_history) == 1
        assert len(tracker.right_history) == 1
        assert tracker.left_history[0].x == pytest.approx(-0.10)
        assert tracker.right_history[0].x == pytest.approx(0.10)

    def test_history_bounded_to_maxlen(self):
        tracker = TwoFootTracker(history=10)
        # Run 15 well-spaced steps.
        t0 = 0.0
        for i in range(15):
            for (t, x, y, z) in _step_trajectory(duration=0.5, rest_frames=30):
                tracker.update_left(t + t0, x, y, z)
            t0 += 1.0
        assert len(tracker.left_history) == 10

    def test_reset_clears_state(self):
        tracker = TwoFootTracker()
        for (t, x, y, z) in _step_trajectory():
            tracker.update_left(t, x, y, z)
        assert len(tracker.left_history) == 1
        tracker.reset()
        assert len(tracker.left_history) == 0
        # And after reset, should still detect a fresh step.
        for (t, x, y, z) in _step_trajectory():
            tracker.update_left(t, x, y, z)
        assert len(tracker.left_history) == 1
