"""
Application state management.

AppState is a frozen dataclass holding all application state.
StateManager is a thin QObject that:
- Spawns workers for async operations
- Updates AppState immutably
- Emits signals on state changes
- Does NOT contain business logic
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    import numpy as np
    from ..types import (
        CameraConfig,
        CameraIntrinsics,
        CalibratedCamera,
        CharucoConfig,
        ProjectConfig,
    )
    from ..camera_binding import CameraInfo


@dataclass(frozen=True)
class CameraState:
    """State for a single camera."""

    info: "CameraInfo"
    enabled: bool = True
    is_open: bool = False
    last_frame: "np.ndarray | None" = None


@dataclass(frozen=True)
class RecordingState:
    """State for recording session."""

    is_recording: bool = False
    output_path: Path | None = None
    current_frame: int = 0
    total_frames: int = 0
    fps: int = 30
    duration: float = 10.0


@dataclass(frozen=True)
class CalibrationState:
    """State for calibration workflow."""

    # Intrinsic
    intrinsic_video_paths: dict[int, Path] = field(default_factory=dict)
    intrinsics: dict[int, "CameraIntrinsics"] = field(default_factory=dict)
    intrinsic_progress: dict[int, float] = field(default_factory=dict)

    # Extrinsic
    extrinsic_video_paths: dict[int, Path] = field(default_factory=dict)
    calibrated_cameras: dict[int, "CalibratedCamera"] = field(default_factory=dict)
    extrinsic_error: float | None = None


@dataclass(frozen=True)
class ProcessingState:
    """State for processing workflow."""

    input_videos: dict[int, Path] = field(default_factory=dict)
    is_processing: bool = False
    current_step: str = ""
    progress: float = 0.0


@dataclass(frozen=True)
class AppState:
    """
    Complete application state.

    Immutable - all updates create new instances via dataclasses.replace().
    """

    # Project
    project_path: Path | None = None
    project_config: "ProjectConfig | None" = None
    charuco_config: "CharucoConfig | None" = None

    # Cameras
    cameras: dict[int, CameraState] = field(default_factory=dict)
    is_previewing: bool = False

    # Recording
    recording: RecordingState = field(default_factory=RecordingState)

    # Calibration
    calibration: CalibrationState = field(default_factory=CalibrationState)

    # Processing
    processing: ProcessingState = field(default_factory=ProcessingState)

    # UI
    current_tab: int = 0
    status_message: str = ""


class StateManager(QObject):
    """
    Thin coordinator between UI and workers.

    Responsibilities:
    - Hold current AppState
    - Emit signals when state changes
    - Spawn workers for async operations
    - Update state from worker results

    Does NOT contain business logic - that lives in pure functions.
    """

    # State change signals
    state_changed = Signal(AppState)
    cameras_changed = Signal(dict)  # dict[int, CameraState]
    recording_changed = Signal(RecordingState)
    calibration_changed = Signal(CalibrationState)
    processing_changed = Signal(ProcessingState)

    # Status signals
    status_message = Signal(str)
    error_occurred = Signal(str)

    # Frame signals (high frequency, separate from state)
    frame_received = Signal(int, object)  # port, np.ndarray

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._state = AppState()

    @property
    def state(self) -> AppState:
        """Current application state."""
        return self._state

    def update_state(self, **kwargs) -> None:
        """
        Update state with new values.

        Usage:
            manager.update_state(current_tab=1, status_message="Ready")
        """
        self._state = replace(self._state, **kwargs)
        self.state_changed.emit(self._state)

    def set_cameras(self, cameras: dict[int, CameraState]) -> None:
        """Update camera state."""
        self._state = replace(self._state, cameras=cameras)
        self.cameras_changed.emit(cameras)
        self.state_changed.emit(self._state)

    def update_camera(self, port: int, **kwargs) -> None:
        """Update a single camera's state."""
        if port not in self._state.cameras:
            return
        old_cam = self._state.cameras[port]
        new_cam = replace(old_cam, **kwargs)
        new_cameras = {**self._state.cameras, port: new_cam}
        self.set_cameras(new_cameras)

    def set_recording(self, recording: RecordingState) -> None:
        """Update recording state."""
        self._state = replace(self._state, recording=recording)
        self.recording_changed.emit(recording)
        self.state_changed.emit(self._state)

    def update_recording(self, **kwargs) -> None:
        """Update recording state with new values."""
        new_recording = replace(self._state.recording, **kwargs)
        self.set_recording(new_recording)

    def set_calibration(self, calibration: CalibrationState) -> None:
        """Update calibration state."""
        self._state = replace(self._state, calibration=calibration)
        self.calibration_changed.emit(calibration)
        self.state_changed.emit(self._state)

    def update_calibration(self, **kwargs) -> None:
        """Update calibration state with new values."""
        new_calibration = replace(self._state.calibration, **kwargs)
        self.set_calibration(new_calibration)

    def set_processing(self, processing: ProcessingState) -> None:
        """Update processing state."""
        self._state = replace(self._state, processing=processing)
        self.processing_changed.emit(processing)
        self.state_changed.emit(self._state)

    def update_processing(self, **kwargs) -> None:
        """Update processing state with new values."""
        new_processing = replace(self._state.processing, **kwargs)
        self.set_processing(new_processing)

    def set_status(self, message: str) -> None:
        """Set status bar message."""
        self._state = replace(self._state, status_message=message)
        self.status_message.emit(message)

    def report_error(self, message: str) -> None:
        """Report an error."""
        self.error_occurred.emit(message)
        self.set_status(f"Error: {message}")
