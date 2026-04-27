"""Imports + schema test for the offline-processing wiring.

Locks: OfflineProcessingWorker exists, RecordingWorker takes a `retain_frames`
flag, and pose_batch_size lives in the project-settings defaults.
"""

from __future__ import annotations


def test_offline_processing_worker_importable():
    from calimerge.gui.workers import OfflineProcessingWorker
    assert hasattr(OfflineProcessingWorker, "run")


def test_recording_worker_accepts_retain_frames():
    from calimerge.gui.workers import RecordingWorker
    import inspect
    sig = inspect.signature(RecordingWorker.__init__)
    assert "retain_frames" in sig.parameters
    assert sig.parameters["retain_frames"].default is False


def test_pose_batch_size_in_project_settings_defaults():
    from calimerge.config import _PROJECT_SETTINGS_DEFAULTS
    assert "pose_batch_size" in _PROJECT_SETTINGS_DEFAULTS
    val = _PROJECT_SETTINGS_DEFAULTS["pose_batch_size"]
    assert isinstance(val, int) and val >= 1


def test_workout_page_offline_progress_widgets_exist():
    """The footer progress strip must be wired so _start_offline_processing
    has somewhere to render."""
    import pytest
    pytest.importorskip("pytestqt")
    from PySide6.QtWidgets import QApplication
    if QApplication.instance() is None:
        QApplication([])
    from calimerge.gui.workout_page import WorkoutPage
    from calimerge.gui.state import StateManager
    page = WorkoutPage(StateManager())
    assert hasattr(page, "offline_progress_container")
    assert hasattr(page, "offline_progress_bar")
    assert hasattr(page, "offline_status_label")
    assert page.offline_progress_container.isVisible() is False, (
        "offline strip should start hidden"
    )
