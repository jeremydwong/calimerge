"""
Tests for calimerge.gui.state.
"""

from pathlib import Path

import numpy as np
import pytest

from calimerge.gui.state import (
    AppState,
    CameraState,
    RecordingState,
    CalibrationState,
    ProcessingState,
    StateManager,
)


class TestCameraState:
    def test_defaults(self):
        # Create a mock camera info
        class MockCameraInfo:
            serial_number = "ABC123"
            display_name = "Test Camera"

        state = CameraState(info=MockCameraInfo())
        assert state.enabled is True
        assert state.is_open is False
        assert state.last_frame is None


class TestRecordingState:
    def test_defaults(self):
        state = RecordingState()
        assert state.is_recording is False
        assert state.output_path is None
        assert state.current_frame == 0
        assert state.total_frames == 0
        assert state.fps == 30
        assert state.duration == 10.0


class TestCalibrationState:
    def test_defaults(self):
        state = CalibrationState()
        assert state.intrinsic_video_paths == {}
        assert state.intrinsics == {}
        assert state.extrinsic_video_paths == {}
        assert state.calibrated_cameras == {}
        assert state.extrinsic_error is None


class TestProcessingState:
    def test_defaults(self):
        state = ProcessingState()
        assert state.input_videos == {}
        assert state.is_processing is False
        assert state.current_step == ""
        assert state.progress == 0.0


class TestAppState:
    def test_defaults(self):
        state = AppState()
        assert state.project_path is None
        assert state.project_config is None
        assert state.cameras == {}
        assert state.is_previewing is False
        assert state.current_tab == 0
        assert state.status_message == ""

    def test_immutability(self):
        state = AppState()
        # Frozen dataclass should raise on attribute assignment
        with pytest.raises(AttributeError):
            state.current_tab = 1

    def test_nested_states(self):
        state = AppState()
        assert isinstance(state.recording, RecordingState)
        assert isinstance(state.calibration, CalibrationState)
        assert isinstance(state.processing, ProcessingState)


class TestStateManager:
    """Tests for StateManager - no Qt event loop needed for basic tests."""

    def test_initial_state(self):
        manager = StateManager()
        assert manager.state is not None
        assert isinstance(manager.state, AppState)

    def test_update_state(self):
        """update_state should create new state with updated values."""
        manager = StateManager()
        manager.update_state(current_tab=2, status_message="Testing")

        assert manager.state.current_tab == 2
        assert manager.state.status_message == "Testing"

    def test_set_status(self):
        manager = StateManager()
        manager.set_status("Hello")

        assert manager.state.status_message == "Hello"

    def test_update_recording(self):
        manager = StateManager()
        manager.update_recording(is_recording=True, fps=24)

        assert manager.state.recording.is_recording is True
        assert manager.state.recording.fps == 24
        # Other fields should remain default
        assert manager.state.recording.duration == 10.0

    def test_update_calibration(self):
        manager = StateManager()
        manager.update_calibration(extrinsic_error=0.25)

        assert manager.state.calibration.extrinsic_error == 0.25

    def test_update_processing(self):
        manager = StateManager()
        manager.update_processing(is_processing=True, current_step="Tracking")

        assert manager.state.processing.is_processing is True
        assert manager.state.processing.current_step == "Tracking"

    def test_set_cameras(self):
        class MockCameraInfo:
            serial_number = "CAM001"
            display_name = "Camera 1"

        manager = StateManager()
        cameras = {
            0: CameraState(info=MockCameraInfo(), enabled=True),
        }
        manager.set_cameras(cameras)

        assert 0 in manager.state.cameras
        assert manager.state.cameras[0].enabled is True

    def test_update_camera(self):
        class MockCameraInfo:
            serial_number = "CAM001"
            display_name = "Camera 1"

        manager = StateManager()
        cameras = {
            0: CameraState(info=MockCameraInfo(), enabled=True),
        }
        manager.set_cameras(cameras)

        manager.update_camera(0, enabled=False, is_open=True)

        assert manager.state.cameras[0].enabled is False
        assert manager.state.cameras[0].is_open is True

    def test_update_nonexistent_camera_does_nothing(self):
        manager = StateManager()
        # Should not raise
        manager.update_camera(999, enabled=False)
        # State should be unchanged
        assert 999 not in manager.state.cameras

    def test_report_error(self):
        manager = StateManager()
        manager.report_error("Something went wrong")

        assert "Error" in manager.state.status_message

    def test_state_immutability_preserved(self):
        """Each update should create a new state object."""
        manager = StateManager()
        state1 = manager.state
        manager.update_state(current_tab=1)
        state2 = manager.state

        # Should be different objects
        assert state1 is not state2
        # Original should be unchanged
        assert state1.current_tab == 0
        assert state2.current_tab == 1
