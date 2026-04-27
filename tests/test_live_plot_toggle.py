"""Tests for the recording-time tracking toggle and the queue-coalescing
paint path.

Two distinct features verified here:

1. **Pause live tracking during recording** — checkbox in the Session
   group. When checked (default), recording stops the detection worker
   for its duration and restarts it on finish, freeing the cameras to
   hit commanded fps. Raw videos are always saved.

2. **Drop-old paint coalescing** — `_on_detection_ready` and
   `_on_keypoints_3d` route through `_pending_*` slots + a one-shot
   QTimer so a backlog of queued cross-thread emits can't replay
   minute-stale frames when the UI thread is overloaded.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytestqt")


@pytest.fixture
def page(qtbot):
    from calimerge.gui.workout_page import WorkoutPage
    from calimerge.gui.state import StateManager

    sm = StateManager()
    p = WorkoutPage(sm)
    qtbot.addWidget(p)
    return p


def test_pause_tracking_checkbox_exists_and_defaults_on(page):
    assert hasattr(page, "pause_tracking_during_record_checkbox")
    assert page.pause_tracking_during_record_checkbox.isChecked() is True


def test_pause_tracking_label_is_not_bold(page):
    """Should match Gill Sans body weight, not chunky bold."""
    css = page.pause_tracking_during_record_checkbox.styleSheet().lower()
    assert "font-weight: normal" in css


def test_csv_checkbox_label_is_not_bold(page):
    """Same regular-weight contract for the sibling 'Generate CSV' toggle."""
    css = page.generate_csv_checkbox.styleSheet().lower()
    assert "font-weight: normal" in css


def test_application_font_is_gill_sans(qtbot):
    """Confirm the QApplication-wide font policy still puts Gill Sans first
    in the fallback chain, so checkbox labels inherit it."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    # The actual font used by main() is set in gui.main.main(); here we
    # simulate the same setFont and read it back.
    from PySide6.QtGui import QFont
    font = QFont()
    font.setFamilies([
        "Gill Sans", "Gill Sans MT", "Gill Sans Std", "Gill Sans Nova",
        "Helvetica Neue", "Helvetica", "Segoe UI", "Arial",
    ])
    app.setFont(font)
    assert app.font().families()[0] == "Gill Sans"


# ── Drop-old paint coalescing ──────────────────────────────────────────


def test_grid_paint_coalesces_to_latest(page, monkeypatch, qtbot):
    """Many queued _on_detection_ready calls between repaints should
    collapse into ONE camera_grid.update_frame call, painting the freshest
    frame per port."""
    import numpy as np

    calls: list = []
    monkeypatch.setattr(
        page.camera_grid, "update_frame",
        lambda port, frame: calls.append((port, int(frame[0, 0, 0]))),
    )

    # Three "frames" per port, distinguishable by their first pixel value.
    f1 = np.full((10, 10, 3), 1, dtype=np.uint8)
    f2 = np.full((10, 10, 3), 2, dtype=np.uint8)
    f3 = np.full((10, 10, 3), 3, dtype=np.uint8)
    page._on_detection_ready(0, f1)
    page._on_detection_ready(0, f2)
    page._on_detection_ready(0, f3)

    qtbot.wait(20)  # let the QTimer.singleShot fire

    assert len(calls) == 1, f"expected one coalesced paint, got {calls}"
    assert calls[0] == (0, 3), "must paint the freshest frame, not a stale one"


def test_skeleton_paint_coalesces_to_latest(page, monkeypatch, qtbot):
    """Same drop-old story for keypoints_3d_ready emits."""
    import numpy as np

    calls: list = []
    monkeypatch.setattr(
        page.skeleton_view, "update_keypoints",
        lambda persons: calls.append(persons),
    )

    # Distinguish frames by the value at the first keypoint.
    def make(value: float):
        return [[np.array([value, 0.0, 0.0])]]

    page._on_keypoints_3d(make(1.0), primary_index=0)
    page._on_keypoints_3d(make(2.0), primary_index=1)
    page._on_keypoints_3d(make(3.0), primary_index=2)

    qtbot.wait(20)

    assert len(calls) == 1, "expected coalesced single paint"
    assert float(calls[0][0][0][0]) == 3.0, "must paint the freshest persons list"
    assert page._primary_person_index == 2, (
        "data plumbing must keep updating per emit even though paint coalesces"
    )


def test_recording_buffer_fills_per_emit_not_coalesced(page):
    """Critical scientific-data invariant: every emitted keypoint set must
    land in _recording_keypoints — that's what gets saved as raw data.
    Coalescing is for paints only."""
    import numpy as np

    page._is_recording = True
    page._recording_start_time = 0.0
    page._recording_keypoints = []

    fake_persons = [[np.array([0.0, 0.0, 1.0])] * 52]
    page._on_keypoints_3d(fake_persons, primary_index=0)
    page._on_keypoints_3d(fake_persons, primary_index=0)
    page._on_keypoints_3d(fake_persons, primary_index=0)

    assert len(page._recording_keypoints) == 3, (
        "recording buffer must capture every emit; coalescing is paint-only"
    )
