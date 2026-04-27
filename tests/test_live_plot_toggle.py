"""Tests for the Live-plot kill switch in the workout page.

Locks the contract: when the checkbox is unchecked, the heavy per-frame UI
paints (camera grid + 3D skeleton view) are skipped, but the data plumbing
(_last_annotated cache, recording buffer fill, primary-person tracking) is
unaffected. Recording must still capture data even with live plotting off.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytestqt")


@pytest.fixture
def page(qtbot):
    """Build a WorkoutPage with a stub StateManager."""
    from calimerge.gui.workout_page import WorkoutPage
    from calimerge.gui.state import StateManager

    sm = StateManager()
    p = WorkoutPage(sm)
    qtbot.addWidget(p)
    return p


def test_live_plot_checkbox_exists_and_defaults_on(page):
    assert hasattr(page, "live_plot_checkbox")
    assert page.live_plot_checkbox.isChecked() is True
    assert page._live_plot_enabled() is True


def test_unchecking_skips_skeleton_paint(page, monkeypatch):
    """When live plotting is off, _on_keypoints_3d must NOT call
    skeleton_view.update_keypoints(). The data plumbing (primary index)
    still updates."""
    calls: list = []
    monkeypatch.setattr(
        page.skeleton_view, "update_keypoints", lambda persons: calls.append(persons)
    )

    page.live_plot_checkbox.setChecked(False)
    fake_persons = [[None] * 52]
    page._on_keypoints_3d(fake_persons, primary_index=0)

    assert calls == [], "skeleton_view paint fired despite live-plot off"
    assert page._primary_person_index == 0


def test_checked_paints_skeleton(page, monkeypatch):
    """And the inverse: with the box checked, paints fire."""
    calls: list = []
    monkeypatch.setattr(
        page.skeleton_view, "update_keypoints", lambda persons: calls.append(persons)
    )

    page.live_plot_checkbox.setChecked(True)
    fake_persons = [[None] * 52]
    page._on_keypoints_3d(fake_persons, primary_index=0)

    assert len(calls) == 1


def test_unchecking_skips_camera_grid_but_caches_frame(page, monkeypatch):
    """Detection-ready must skip the grid update when off, but the
    _last_annotated cache should still fill so other readers (post-recording
    re-render, screenshots, etc.) see the latest frame."""
    import numpy as np

    grid_calls: list = []
    monkeypatch.setattr(
        page.camera_grid, "update_frame",
        lambda port, frame: grid_calls.append(port),
    )

    page.live_plot_checkbox.setChecked(False)
    fake = np.zeros((10, 10, 3), dtype=np.uint8)
    page._on_detection_ready(port=0, annotated_frame=fake)

    assert grid_calls == [], "camera grid painted with live-plot off"
    assert 0 in page._last_annotated, "frame cache must still fill"


def test_re_enabling_resumes_paints(page, monkeypatch):
    """Toggle off, send a frame (skipped), toggle on, send another (drawn).
    No re-init step required."""
    import numpy as np

    grid_calls: list = []
    monkeypatch.setattr(
        page.camera_grid, "update_frame",
        lambda port, frame: grid_calls.append(port),
    )

    fake = np.zeros((10, 10, 3), dtype=np.uint8)

    page.live_plot_checkbox.setChecked(False)
    page._on_detection_ready(port=0, annotated_frame=fake)
    assert grid_calls == []

    page.live_plot_checkbox.setChecked(True)
    page._on_detection_ready(port=0, annotated_frame=fake)
    assert grid_calls == [0]


def test_recording_buffer_fills_when_live_plot_off(page, monkeypatch):
    """Critical scientific-data invariant: turning off live plotting must
    NOT stop the recording buffer from collecting keypoints. The whole point
    of the kill switch is: free CPU for detection + still save the data."""
    page._is_recording = True
    page._recording_start_time = 0.0
    page._recording_keypoints = []

    page.live_plot_checkbox.setChecked(False)

    # Pretend three frames come in.
    import numpy as np
    fake_persons = [[np.array([0.0, 0.0, 1.0])] * 52]
    page._on_keypoints_3d(fake_persons, primary_index=0)
    page._on_keypoints_3d(fake_persons, primary_index=0)
    page._on_keypoints_3d(fake_persons, primary_index=0)

    assert len(page._recording_keypoints) == 3, (
        "recording buffer must fill regardless of the live-plot toggle"
    )
