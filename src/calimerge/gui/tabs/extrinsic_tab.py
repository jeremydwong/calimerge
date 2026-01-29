"""
Extrinsic calibration tab - multi-camera spatial calibration.
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
    QFileDialog,
    QProgressBar,
    QSplitter,
    QTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from ..state import StateManager
from ..workers import ExtrinsicCalibrationWorker
from ..widgets.video_player import VideoPlayer


class ExtrinsicTab(QWidget):
    """
    Extrinsic calibration tab.

    Allows users to:
    - Load synchronized calibration videos
    - Verify intrinsics are available
    - Run bundle adjustment
    - View and export camera positions
    """

    status_message = Signal(str)

    def __init__(self, state_manager: StateManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.state_manager = state_manager

        self.calibration_worker: ExtrinsicCalibrationWorker | None = None
        self.video_paths: dict[int, Path] = {}

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel(
            "Load synchronized videos of ChArUco board captured from multiple cameras. "
            "Intrinsic calibration must be completed first for all cameras."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(instructions)

        # ChArUco settings for extrinsic (typically larger board)
        charuco_group = QGroupBox("ChArUco Board (Extrinsic)")
        charuco_layout = QHBoxLayout(charuco_group)

        charuco_layout.addWidget(QLabel("Columns:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(3, 20)
        self.cols_spin.setValue(4)  # Smaller grid for larger squares
        charuco_layout.addWidget(self.cols_spin)

        charuco_layout.addWidget(QLabel("Rows:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(3, 20)
        self.rows_spin.setValue(3)
        charuco_layout.addWidget(self.rows_spin)

        charuco_layout.addWidget(QLabel("Square (cm):"))
        self.square_spin = QDoubleSpinBox()
        self.square_spin.setRange(1.0, 30.0)
        self.square_spin.setValue(5.0)  # 5cm default for extrinsic
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

        # Main splitter
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
            ["Camera", "Intrinsics", "Video", "Status"]
        )
        self.camera_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.camera_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.camera_table.itemSelectionChanged.connect(self._on_camera_selected)
        cameras_layout.addWidget(self.camera_table)

        # Video controls
        video_controls = QHBoxLayout()

        self.load_folder_button = QPushButton("Load Video Folder...")
        self.load_folder_button.clicked.connect(self._load_video_folder)
        video_controls.addWidget(self.load_folder_button)

        self.load_single_button = QPushButton("Load Single Video...")
        self.load_single_button.clicked.connect(self._load_single_video)
        video_controls.addWidget(self.load_single_button)

        cameras_layout.addLayout(video_controls)
        left_layout.addWidget(cameras_group)

        # Calibration controls
        calibration_group = QGroupBox("Calibration")
        calibration_layout = QVBoxLayout(calibration_group)

        self.calibrate_button = QPushButton("Run Extrinsic Calibration")
        self.calibrate_button.setMinimumHeight(40)
        self.calibrate_button.clicked.connect(self._run_calibration)
        calibration_layout.addWidget(self.calibrate_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        calibration_layout.addWidget(self.progress_bar)

        left_layout.addWidget(calibration_group)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 10))
        self.results_text.setMaximumHeight(200)
        self.results_text.setPlaceholderText("Calibration results will appear here...")
        results_layout.addWidget(self.results_text)

        export_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Camera Rig")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_rig)
        export_layout.addWidget(self.export_button)
        export_layout.addStretch()

        results_layout.addLayout(export_layout)
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
        """Update camera table."""
        cal_state = self.state_manager.state.calibration

        self.camera_table.setRowCount(len(cameras))

        for row, (port, cam_state) in enumerate(sorted(cameras.items())):
            # Camera name
            name_item = QTableWidgetItem(cam_state.info.display_name)
            name_item.setData(Qt.ItemDataRole.UserRole, port)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 0, name_item)

            # Intrinsics status
            has_intrinsics = port in cal_state.intrinsics
            intrinsics_text = "Ready" if has_intrinsics else "Missing"
            intrinsics_item = QTableWidgetItem(intrinsics_text)
            intrinsics_item.setFlags(intrinsics_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if not has_intrinsics:
                intrinsics_item.setForeground(Qt.GlobalColor.red)
            self.camera_table.setItem(row, 1, intrinsics_item)

            # Video path
            video_path = self.video_paths.get(port)
            video_text = video_path.name if video_path else "Not loaded"
            video_item = QTableWidgetItem(video_text)
            video_item.setFlags(video_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 2, video_item)

            # Calibration status
            if port in cal_state.calibrated_cameras:
                status = "Calibrated"
            else:
                status = "Not calibrated"
            status_item = QTableWidgetItem(status)
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 3, status_item)

    def _on_calibration_changed(self, calibration):
        """Refresh table on calibration change."""
        cameras = self.state_manager.state.cameras
        if cameras:
            self._on_cameras_changed(cameras)

        # Update results
        if calibration.calibrated_cameras:
            self._show_results(calibration.calibrated_cameras, calibration.extrinsic_error)
            self.export_button.setEnabled(True)
        else:
            self.export_button.setEnabled(False)

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

        if port in self.video_paths:
            self.video_player.load_video(self.video_paths[port])
        else:
            self.video_player.unload()

    def _load_video_folder(self):
        """Load videos from folder matching port_X.mp4 pattern."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Video Folder"
        )
        if not folder:
            return

        folder_path = Path(folder)
        cameras = self.state_manager.state.cameras
        loaded = 0

        for port in cameras:
            # Try port_X.mp4 pattern
            video_path = folder_path / f"port_{port}.mp4"
            if video_path.exists():
                self.video_paths[port] = video_path
                loaded += 1

        self._on_cameras_changed(cameras)
        self.status_message.emit(f"Loaded {loaded} videos from folder")

    def _load_single_video(self):
        """Load video for selected camera."""
        port = self._get_selected_port()
        if port is None:
            self.status_message.emit("Select a camera first")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.avi *.mov)"
        )
        if path:
            self.video_paths[port] = Path(path)
            self.video_player.load_video(Path(path))
            self._on_cameras_changed(self.state_manager.state.cameras)
            self.status_message.emit(f"Loaded video for camera {port}")

    def _get_charuco_config(self):
        """Get current ChArUco configuration for extrinsic calibration."""
        from ...types import CharucoConfig

        return CharucoConfig(
            columns=self.cols_spin.value(),
            rows=self.rows_spin.value(),
            square_size_cm=self.square_spin.value(),
            dictionary=self.dict_combo.currentText(),
        )

    def _run_calibration(self):
        """Run extrinsic calibration."""
        cameras = self.state_manager.state.cameras
        cal_state = self.state_manager.state.calibration

        # Check all cameras have intrinsics
        missing_intrinsics = [
            port for port in cameras if port not in cal_state.intrinsics
        ]
        if missing_intrinsics:
            self.status_message.emit(
                f"Missing intrinsics for cameras: {missing_intrinsics}"
            )
            return

        # Check all cameras have videos
        missing_videos = [port for port in cameras if port not in self.video_paths]
        if missing_videos:
            self.status_message.emit(f"Missing videos for cameras: {missing_videos}")
            return

        # Get charuco config from UI
        charuco_config = self._get_charuco_config()

        self.calibrate_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.calibration_worker = ExtrinsicCalibrationWorker(
            video_paths=self.video_paths.copy(),
            intrinsics=cal_state.intrinsics.copy(),
            charuco_config=charuco_config,
        )
        self.calibration_worker.log_message.connect(
            lambda msg: self.results_text.append(msg)
        )
        self.calibration_worker.progress_update.connect(
            lambda p: self.progress_bar.setValue(int(p * 100))
        )
        self.calibration_worker.calibration_finished.connect(self._on_calibration_done)
        self.calibration_worker.error.connect(self._on_calibration_error)
        self.calibration_worker.finished.connect(self._on_worker_finished)
        self.calibration_worker.start()

        self.status_message.emit("Running extrinsic calibration...")

    def _on_calibration_done(self, cameras: dict, error: float):
        """Handle calibration completion."""
        self.state_manager.update_calibration(
            calibrated_cameras=cameras, extrinsic_error=error
        )
        self.status_message.emit(f"Extrinsic calibration complete, error: {error:.4f}")

    def _on_calibration_error(self, error: str):
        """Handle calibration error."""
        self.status_message.emit(f"Calibration failed: {error}")
        self.results_text.append(f"ERROR: {error}")

    def _on_worker_finished(self):
        """Clean up after worker."""
        self.calibrate_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.calibration_worker = None

    def _show_results(self, cameras: dict, error: float | None):
        """Display calibration results."""
        self.results_text.clear()
        self.results_text.append("=== Extrinsic Calibration Results ===\n")

        if error is not None:
            self.results_text.append(f"Reprojection error: {error:.4f}\n")

        for port, cam in sorted(cameras.items()):
            self.results_text.append(f"\nCamera {port}:")
            self.results_text.append(f"  Translation: [{cam.extrinsics.translation[0]:.3f}, "
                                     f"{cam.extrinsics.translation[1]:.3f}, "
                                     f"{cam.extrinsics.translation[2]:.3f}]")

    def _export_rig(self):
        """Export camera rig to file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Camera Rig", "camera_rig.toml", "TOML (*.toml)"
        )
        if not path:
            return

        cal_state = self.state_manager.state.calibration
        if not cal_state.calibrated_cameras:
            return

        try:
            # Simple TOML export
            import rtoml

            rig_data = {}
            for port, cam in cal_state.calibrated_cameras.items():
                rig_data[f"camera_{port}"] = {
                    "serial_number": cam.serial_number,
                    "translation": cam.extrinsics.translation.tolist(),
                    "rotation": cam.extrinsics.rotation.flatten().tolist(),
                }

            with open(path, "w") as f:
                rtoml.dump(rig_data, f)

            self.status_message.emit(f"Exported camera rig to {path}")
        except Exception as e:
            self.status_message.emit(f"Export failed: {e}")
