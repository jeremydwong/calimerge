"""
Cameras tab - detection, preview, configuration.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QCheckBox,
    QSplitter,
)
from PySide6.QtCore import Qt, Signal

from ..widgets.camera_grid import CameraGrid
from ..state import StateManager, CameraState
from ..workers import CameraEnumerateWorker, CameraPreviewWorker


class CamerasTab(QWidget):
    """
    Camera detection and preview tab.

    Allows users to:
    - Enumerate connected cameras
    - Enable/disable individual cameras
    - Preview live feeds
    - View camera information
    """

    status_message = Signal(str)

    def __init__(self, state_manager: StateManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.state_manager = state_manager

        self.enumerate_worker: CameraEnumerateWorker | None = None
        self.preview_worker: CameraPreviewWorker | None = None
        self.opened_cameras: list = []

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()

        self.refresh_button = QPushButton("Refresh Cameras")
        self.refresh_button.clicked.connect(self.refresh_cameras)
        toolbar.addWidget(self.refresh_button)

        self.preview_button = QPushButton("Start Preview")
        self.preview_button.setCheckable(True)
        self.preview_button.clicked.connect(self.toggle_preview)
        self.preview_button.setEnabled(False)
        toolbar.addWidget(self.preview_button)

        toolbar.addStretch()

        self.camera_count_label = QLabel("No cameras detected")
        toolbar.addWidget(self.camera_count_label)

        layout.addLayout(toolbar)

        # Main content: splitter with camera list and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: camera table
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        camera_group = QGroupBox("Detected Cameras")
        camera_layout = QVBoxLayout(camera_group)

        self.camera_table = QTableWidget()
        self.camera_table.setColumnCount(4)
        self.camera_table.setHorizontalHeaderLabels(
            ["Port", "Name", "Serial", "Enabled"]
        )
        self.camera_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.camera_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        camera_layout.addWidget(self.camera_table)

        left_layout.addWidget(camera_group)
        splitter.addWidget(left_panel)

        # Right: camera preview grid
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        preview_group = QGroupBox("Live Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.camera_grid = CameraGrid()
        preview_layout.addWidget(self.camera_grid)

        right_layout.addWidget(preview_group)
        splitter.addWidget(right_panel)

        splitter.setSizes([300, 500])
        layout.addWidget(splitter)

    def _connect_signals(self):
        self.state_manager.cameras_changed.connect(self._on_cameras_changed)
        self.state_manager.frame_received.connect(self._on_frame_received)

    def refresh_cameras(self):
        """Start camera enumeration."""
        self.refresh_button.setEnabled(False)
        self.status_message.emit("Enumerating cameras...")

        self.enumerate_worker = CameraEnumerateWorker()
        self.enumerate_worker.cameras_found.connect(self._on_cameras_found)
        self.enumerate_worker.error.connect(self._on_enumerate_error)
        self.enumerate_worker.finished.connect(
            lambda: self.refresh_button.setEnabled(True)
        )
        self.enumerate_worker.start()

    def _on_cameras_found(self, cameras: list):
        """Handle camera enumeration results."""
        from ..state import CameraState

        camera_states = {}
        for i, cam in enumerate(cameras):
            camera_states[i] = CameraState(info=cam, enabled=True, is_open=False)

        self.state_manager.set_cameras(camera_states)
        self._update_camera_table(camera_states)

        count = len(cameras)
        self.camera_count_label.setText(f"{count} camera(s) detected")
        self.preview_button.setEnabled(count > 0)
        self.status_message.emit(f"Found {count} camera(s)")

    def _on_enumerate_error(self, error: str):
        """Handle enumeration error."""
        self.status_message.emit(f"Error: {error}")
        self.state_manager.report_error(error)

    def _update_camera_table(self, cameras: dict[int, CameraState]):
        """Update the camera table."""
        self.camera_table.setRowCount(len(cameras))

        for row, (port, cam_state) in enumerate(sorted(cameras.items())):
            info = cam_state.info

            # Port
            port_item = QTableWidgetItem(str(port))
            port_item.setFlags(port_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 0, port_item)

            # Name
            name_item = QTableWidgetItem(info.display_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 1, name_item)

            # Serial
            serial_item = QTableWidgetItem(info.serial_number)
            serial_item.setFlags(serial_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 2, serial_item)

            # Enabled checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(cam_state.enabled)
            checkbox.stateChanged.connect(
                lambda state, p=port: self._on_enabled_changed(p, state)
            )
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.camera_table.setCellWidget(row, 3, checkbox_widget)

    def _on_enabled_changed(self, port: int, state: int):
        """Handle camera enable/disable toggle."""
        enabled = state == Qt.CheckState.Checked.value
        self.state_manager.update_camera(port, enabled=enabled)

    def toggle_preview(self):
        """Toggle live preview on/off."""
        if self.preview_button.isChecked():
            self.start_preview()
        else:
            self.stop_preview()

    def start_preview(self):
        """Start live preview of enabled cameras."""
        from ..camera_binding import open_camera, close_camera

        cameras = self.state_manager.state.cameras

        # Open enabled cameras
        self.opened_cameras = []
        camera_info = {}

        for port, cam_state in cameras.items():
            if not cam_state.enabled:
                continue
            try:
                open_camera(cam_state.info)
                self.opened_cameras.append(cam_state.info)
                camera_info[port] = cam_state.info.display_name
                self.state_manager.update_camera(port, is_open=True)
            except Exception as e:
                self.status_message.emit(f"Failed to open camera {port}: {e}")

        if not self.opened_cameras:
            self.preview_button.setChecked(False)
            self.status_message.emit("No cameras opened")
            return

        # Set up grid
        self.camera_grid.set_cameras(camera_info)

        # Start preview worker
        self.preview_worker = CameraPreviewWorker(self.opened_cameras, fps=30)
        self.preview_worker.frame_captured.connect(self._on_frame_received)
        self.preview_worker.error.connect(self._on_preview_error)
        self.preview_worker.start()

        self.preview_button.setText("Stop Preview")
        self.state_manager.update_state(is_previewing=True)
        self.status_message.emit("Preview started")

    def stop_preview(self):
        """Stop live preview."""
        from ..camera_binding import close_camera

        if self.preview_worker:
            self.preview_worker.stop()
            self.preview_worker.wait()
            self.preview_worker = None

        for cam in self.opened_cameras:
            try:
                close_camera(cam)
            except Exception:
                pass

        for port in self.state_manager.state.cameras:
            self.state_manager.update_camera(port, is_open=False)

        self.opened_cameras = []
        self.camera_grid.clear_all()

        self.preview_button.setText("Start Preview")
        self.preview_button.setChecked(False)
        self.state_manager.update_state(is_previewing=False)
        self.status_message.emit("Preview stopped")

    def _on_frame_received(self, port: int, frame):
        """Handle incoming frame from preview."""
        self.camera_grid.update_frame(port, frame)

    def _on_preview_error(self, error: str):
        """Handle preview error."""
        self.stop_preview()
        self.status_message.emit(f"Preview error: {error}")

    def _on_cameras_changed(self, cameras: dict):
        """Handle cameras state change."""
        self._update_camera_table(cameras)

    def showEvent(self, event):
        """Auto-enumerate on first show."""
        super().showEvent(event)
        if not self.state_manager.state.cameras:
            self.refresh_cameras()

    def hideEvent(self, event):
        """Stop preview when tab is hidden."""
        super().hideEvent(event)
        if self.preview_worker:
            self.stop_preview()
