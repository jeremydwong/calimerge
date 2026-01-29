"""
Intrinsic calibration tab - per-camera lens calibration.
"""

from __future__ import annotations

from pathlib import Path

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
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QProgressBar,
    QSplitter,
    QComboBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from ..state import StateManager
from ..workers import IntrinsicCalibrationWorker
from ..widgets.video_player import VideoPlayer


class IntrinsicTab(QWidget):
    """
    Intrinsic calibration tab.

    Allows users to:
    - Configure ChArUco board parameters
    - Load calibration videos per camera
    - Run intrinsic calibration
    - View and save results
    """

    status_message = Signal(str)

    def __init__(self, state_manager: StateManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.state_manager = state_manager

        self.calibration_workers: dict[int, IntrinsicCalibrationWorker] = {}
        self.video_paths: dict[int, Path] = {}

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ChArUco settings
        charuco_group = QGroupBox("ChArUco Board Settings")
        charuco_layout = QHBoxLayout(charuco_group)

        charuco_layout.addWidget(QLabel("Columns:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(3, 20)
        self.cols_spin.setValue(7)
        charuco_layout.addWidget(self.cols_spin)

        charuco_layout.addWidget(QLabel("Rows:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(3, 20)
        self.rows_spin.setValue(5)
        charuco_layout.addWidget(self.rows_spin)

        charuco_layout.addWidget(QLabel("Square (cm):"))
        self.square_spin = QDoubleSpinBox()
        self.square_spin.setRange(0.5, 20.0)
        self.square_spin.setValue(3.0)  # 3cm default for intrinsic
        self.square_spin.setSingleStep(0.5)
        self.square_spin.setDecimals(1)
        charuco_layout.addWidget(self.square_spin)

        charuco_layout.addWidget(QLabel("Dictionary:"))
        self.dict_combo = QComboBox()
        self.dict_combo.addItems([
            "DICT_4X4_50",
            "DICT_4X4_100",
            "DICT_5X5_50",
            "DICT_5X5_100",
            "DICT_6X6_50",
        ])
        charuco_layout.addWidget(self.dict_combo)

        charuco_layout.addStretch()
        layout.addWidget(charuco_group)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: camera list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Camera table
        cameras_group = QGroupBox("Cameras")
        cameras_layout = QVBoxLayout(cameras_group)

        self.camera_table = QTableWidget()
        self.camera_table.setColumnCount(4)
        self.camera_table.setHorizontalHeaderLabels(
            ["Camera", "Video", "Status", "Error"]
        )
        self.camera_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.camera_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.camera_table.itemSelectionChanged.connect(self._on_camera_selected)
        cameras_layout.addWidget(self.camera_table)

        # Camera controls
        camera_controls = QHBoxLayout()
        self.load_video_button = QPushButton("Load Video...")
        self.load_video_button.clicked.connect(self._load_video)
        camera_controls.addWidget(self.load_video_button)

        self.calibrate_button = QPushButton("Calibrate Selected")
        self.calibrate_button.clicked.connect(self._calibrate_selected)
        camera_controls.addWidget(self.calibrate_button)

        self.calibrate_all_button = QPushButton("Calibrate All")
        self.calibrate_all_button.clicked.connect(self._calibrate_all)
        camera_controls.addWidget(self.calibrate_all_button)

        cameras_layout.addLayout(camera_controls)
        left_layout.addWidget(cameras_group)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Results
        results_group = QGroupBox("Calibration Results")
        results_layout = QVBoxLayout(results_group)
        self.results_label = QLabel("Select a camera to view results")
        self.results_label.setFont(QFont("Courier", 10))
        self.results_label.setWordWrap(True)
        results_layout.addWidget(self.results_label)

        self.save_button = QPushButton("Save to Database")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save_intrinsics)
        results_layout.addWidget(self.save_button)

        left_layout.addWidget(results_group)

        splitter.addWidget(left_panel)

        # Right: video preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        preview_group = QGroupBox("Video Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.video_player = VideoPlayer()
        preview_layout.addWidget(self.video_player)
        right_layout.addWidget(preview_group)

        splitter.addWidget(right_panel)
        splitter.setSizes([400, 400])

        layout.addWidget(splitter)

    def _connect_signals(self):
        self.state_manager.cameras_changed.connect(self._on_cameras_changed)
        self.state_manager.calibration_changed.connect(self._on_calibration_changed)

    def _on_cameras_changed(self, cameras: dict):
        """Update camera table when cameras change."""
        self.camera_table.setRowCount(len(cameras))

        for row, (port, cam_state) in enumerate(sorted(cameras.items())):
            # Camera name
            name_item = QTableWidgetItem(cam_state.info.display_name)
            name_item.setData(Qt.ItemDataRole.UserRole, port)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 0, name_item)

            # Video path
            video_path = self.video_paths.get(port)
            video_text = video_path.name if video_path else "Not loaded"
            video_item = QTableWidgetItem(video_text)
            video_item.setFlags(video_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 1, video_item)

            # Status
            cal_state = self.state_manager.state.calibration
            if port in cal_state.intrinsics:
                status = "Calibrated"
            elif port in cal_state.intrinsic_progress:
                progress = cal_state.intrinsic_progress[port]
                status = f"Processing {progress:.0%}"
            else:
                status = "Not calibrated"
            status_item = QTableWidgetItem(status)
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 2, status_item)

            # Error
            error_text = ""
            if port in cal_state.intrinsics:
                error_text = f"{cal_state.intrinsics[port].error:.4f}"
            error_item = QTableWidgetItem(error_text)
            error_item.setFlags(error_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 3, error_item)

    def _on_calibration_changed(self, calibration):
        """Refresh table on calibration change."""
        cameras = self.state_manager.state.cameras
        if cameras:
            self._on_cameras_changed(cameras)

    def _get_selected_port(self) -> int | None:
        """Get port of selected camera."""
        items = self.camera_table.selectedItems()
        if not items:
            return None
        row = items[0].row()
        item = self.camera_table.item(row, 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _on_camera_selected(self):
        """Handle camera selection."""
        port = self._get_selected_port()
        if port is None:
            return

        # Load video if available
        if port in self.video_paths:
            self.video_player.load_video(self.video_paths[port])
        else:
            self.video_player.unload()

        # Show calibration results if available
        cal_state = self.state_manager.state.calibration
        if port in cal_state.intrinsics:
            intrinsics = cal_state.intrinsics[port]
            self.results_label.setText(
                f"Camera: {intrinsics.serial_number}\n"
                f"Resolution: {intrinsics.resolution[0]}x{intrinsics.resolution[1]}\n"
                f"Error: {intrinsics.error:.4f}\n"
                f"Grid count: {intrinsics.grid_count}\n\n"
                f"fx: {intrinsics.matrix[0, 0]:.2f}\n"
                f"fy: {intrinsics.matrix[1, 1]:.2f}\n"
                f"cx: {intrinsics.matrix[0, 2]:.2f}\n"
                f"cy: {intrinsics.matrix[1, 2]:.2f}"
            )
            self.save_button.setEnabled(True)
        else:
            self.results_label.setText("Not calibrated")
            self.save_button.setEnabled(False)

    def _load_video(self):
        """Load calibration video for selected camera."""
        port = self._get_selected_port()
        if port is None:
            self.status_message.emit("Select a camera first")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Calibration Video", "", "Videos (*.mp4 *.avi *.mov)"
        )
        if path:
            self.video_paths[port] = Path(path)
            self.video_player.load_video(Path(path))
            self._on_cameras_changed(self.state_manager.state.cameras)
            self.status_message.emit(f"Loaded video for camera {port}")

    def _get_charuco_config(self):
        """Get current ChArUco configuration."""
        from ...types import CharucoConfig

        return CharucoConfig(
            columns=self.cols_spin.value(),
            rows=self.rows_spin.value(),
            square_size_cm=self.square_spin.value(),
            dictionary=self.dict_combo.currentText(),
        )

    def _calibrate_selected(self):
        """Calibrate selected camera."""
        port = self._get_selected_port()
        if port is None:
            self.status_message.emit("Select a camera first")
            return

        if port not in self.video_paths:
            self.status_message.emit("Load a video first")
            return

        self._run_calibration(port)

    def _calibrate_all(self):
        """Calibrate all cameras with videos."""
        for port in self.video_paths:
            if port not in self.calibration_workers:
                self._run_calibration(port)

    def _run_calibration(self, port: int):
        """Run calibration for a camera."""
        cam_state = self.state_manager.state.cameras.get(port)
        if not cam_state:
            return

        video_path = self.video_paths[port]
        charuco_config = self._get_charuco_config()

        worker = IntrinsicCalibrationWorker(
            video_path=video_path,
            serial_number=cam_state.info.serial_number,
            charuco_config=charuco_config,
        )
        worker.log_message.connect(lambda msg: self.status_message.emit(msg))
        worker.progress_update.connect(
            lambda cur, tot, p=port: self._on_calibration_progress(p, cur, tot)
        )
        worker.calibration_finished.connect(
            lambda result, p=port: self._on_calibration_finished(p, result)
        )
        worker.error.connect(lambda err, p=port: self._on_calibration_error(p, err))
        worker.finished.connect(lambda p=port: self._on_worker_finished(p))

        self.calibration_workers[port] = worker
        worker.start()

        self.status_message.emit(f"Calibrating camera {port}...")

    def _on_calibration_progress(self, port: int, current: int, total: int):
        """Update calibration progress."""
        progress = current / total if total > 0 else 0
        new_progress = {
            **self.state_manager.state.calibration.intrinsic_progress,
            port: progress,
        }
        self.state_manager.update_calibration(intrinsic_progress=new_progress)

    def _on_calibration_finished(self, port: int, intrinsics):
        """Handle calibration completion - auto-saves to database."""
        new_intrinsics = {
            **self.state_manager.state.calibration.intrinsics,
            port: intrinsics,
        }
        self.state_manager.update_calibration(intrinsics=new_intrinsics)

        # Auto-save to database
        try:
            from ...config import save_intrinsics, get_default_intrinsics_db

            db_path = get_default_intrinsics_db()
            save_intrinsics(intrinsics, db_path)
            self.status_message.emit(
                f"Camera {port} calibrated (error: {intrinsics.error:.4f}), saved to database"
            )
        except Exception as e:
            self.status_message.emit(
                f"Camera {port} calibrated (error: {intrinsics.error:.4f}), "
                f"but failed to save: {e}"
            )

        # Refresh selection
        if self._get_selected_port() == port:
            self._on_camera_selected()

    def _on_calibration_error(self, port: int, error: str):
        """Handle calibration error."""
        self.status_message.emit(f"Camera {port} calibration failed: {error}")

    def _on_worker_finished(self, port: int):
        """Clean up worker."""
        if port in self.calibration_workers:
            del self.calibration_workers[port]

    def _save_intrinsics(self):
        """Save intrinsics to database."""
        port = self._get_selected_port()
        if port is None:
            return

        cal_state = self.state_manager.state.calibration
        if port not in cal_state.intrinsics:
            return

        intrinsics = cal_state.intrinsics[port]

        try:
            from ...config import save_intrinsics, get_default_intrinsics_db

            db_path = get_default_intrinsics_db()
            save_intrinsics(intrinsics, db_path)
            self.status_message.emit(f"Saved intrinsics to {db_path}")
        except Exception as e:
            self.status_message.emit(f"Failed to save: {e}")
