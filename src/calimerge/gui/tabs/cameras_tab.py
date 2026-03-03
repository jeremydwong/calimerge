"""
Record tab - camera detection, preview, configuration, and recording.

Merges camera detection/preview with recording functionality.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from datetime import datetime
import time

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
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QProgressBar,
    QTextEdit,
    QFileDialog,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPainter, QColor, QPen

from ..widgets.camera_grid import CameraGrid
from ..state import StateManager, CameraState
from ..workers import CameraEnumerateWorker, CameraPreviewWorker, RecordingWorker


# ── Camera Colors ──

CAMERA_COLORS = [
    QColor(80, 200, 120),    # 0: green
    QColor(100, 160, 255),   # 1: blue
    QColor(255, 180, 80),    # 2: orange
    QColor(220, 100, 220),   # 3: purple
    QColor(255, 100, 100),   # 4: red
    QColor(100, 220, 220),   # 5: cyan
    QColor(255, 220, 80),    # 6: yellow
    QColor(180, 140, 255),   # 7: lavender
]


def camera_color(port: int) -> QColor:
    """Get color for camera port."""
    return CAMERA_COLORS[port % len(CAMERA_COLORS)]


# ── FPS Graph Widget ──

class FpsGraphWidget(QWidget):
    """Rolling FPS graph with one line per camera."""

    def __init__(self, buffer_size: int = 120, parent: QWidget | None = None):
        super().__init__(parent)
        self.buffer_size = buffer_size
        self._series: dict[int, deque[float]] = {}
        self._names: dict[int, str] = {}
        self._target_fps: int = 30
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_target_fps(self, fps: int):
        """Set target FPS - Y-axis fixed at 0 to 1.5*target."""
        self._target_fps = max(1, fps)
        self.update()

    def set_cameras(self, camera_info: dict[int, str]):
        """Reset series for new camera set."""
        self._series.clear()
        self._names.clear()
        for port, name in camera_info.items():
            self._series[port] = deque(maxlen=self.buffer_size)
            self._names[port] = name
        self.update()

    def push_fps(self, port: int, fps: float):
        if port in self._series:
            self._series[port].append(fps)
            self.update()

    def clear_all(self):
        for s in self._series.values():
            s.clear()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        margin_left = 30
        margin_bottom = 20
        margin_right = 4
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_bottom - 4

        # Background
        painter.fillRect(0, 0, w, h, QColor(26, 26, 26))

        if plot_w < 10 or plot_h < 10:
            painter.end()
            return

        # Fixed Y-axis: 0 to 1.5 * target FPS
        y_max = int(self._target_fps * 1.5)
        time_span_s = self.buffer_size / max(self._target_fps, 1)

        # Y-axis grid lines
        painter.setPen(QPen(QColor(55, 55, 55), 1, Qt.PenStyle.DotLine))
        for target in (self._target_fps // 2, self._target_fps):
            if target > 0 and target <= y_max:
                y = 2 + plot_h - (target / y_max) * plot_h
                painter.drawLine(margin_left, int(y), w - margin_right, int(y))
                painter.setPen(QColor(90, 90, 90))
                painter.drawText(2, int(y) + 4, f"{target}")
                painter.setPen(QPen(QColor(55, 55, 55), 1, Qt.PenStyle.DotLine))

        # X-axis labels
        painter.setPen(QColor(90, 90, 90))
        for sec in range(0, int(time_span_s) + 1):
            if sec % 2 == 0 or time_span_s <= 4:
                x = margin_left + plot_w - (sec / time_span_s) * plot_w
                painter.drawText(int(x) - 8, h - 4, f"{sec}s")

        if not self._series:
            painter.end()
            return

        # Draw each camera's line
        step_x = plot_w / max(self.buffer_size - 1, 1)
        for port in sorted(self._series):
            vals = list(self._series[port])
            n = len(vals)
            if n < 2:
                continue

            color = camera_color(port)
            painter.setPen(QPen(color, 1.5))

            offset = self.buffer_size - n
            prev_x = margin_left + int(offset * step_x)
            prev_y = 2 + int(plot_h - (min(vals[0], y_max) / y_max) * plot_h)
            for i in range(1, n):
                x = margin_left + int((offset + i) * step_x)
                clamped_val = min(vals[i], y_max)
                y = 2 + int(plot_h - (clamped_val / y_max) * plot_h)
                painter.drawLine(prev_x, prev_y, x, y)
                prev_x, prev_y = x, y

        # Legend
        legend_x = margin_left + 6
        legend_y = 14
        for port in sorted(self._series):
            vals = self._series[port]
            if not vals:
                continue
            color = camera_color(port)
            painter.setPen(color)
            name = self._names.get(port, f"Cam {port}")
            short = name[:12] if len(name) > 12 else name
            painter.drawText(legend_x, legend_y, f"{short}: {vals[-1]:.0f}")
            legend_y += 13

        painter.end()


# ── Main Tab ──

class CamerasTab(QWidget):
    """
    Combined camera detection, preview, configuration, and recording tab.
    """

    status_message = Signal(str)

    def __init__(self, state_manager: StateManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.state_manager = state_manager

        self.enumerate_worker: CameraEnumerateWorker | None = None
        self.preview_worker: CameraPreviewWorker | None = None
        self.recording_worker: RecordingWorker | None = None
        self.opened_cameras: list = []
        self.base_output_path = Path("recordings")
        self.output_path: Path | None = None
        self._is_recording = False
        self._updating_table = False

        # Per-camera last-frame timestamp for FPS computation
        self._last_frame_time: dict[int, float] = {}

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # === Toolbar ===
        toolbar = QHBoxLayout()

        self.refresh_button = QPushButton("⟳ Refresh")
        self.refresh_button.clicked.connect(self.refresh_cameras)
        toolbar.addWidget(self.refresh_button)

        self.preview_button = QPushButton("▶ Preview")
        self.preview_button.setCheckable(True)
        self.preview_button.clicked.connect(self.toggle_preview)
        self.preview_button.setEnabled(False)
        toolbar.addWidget(self.preview_button)

        self.record_button = QPushButton("⏺ Record")
        self.record_button.setStyleSheet("font-weight: bold; color: #ff4444;")
        self.record_button.clicked.connect(self._start_recording)
        self.record_button.setEnabled(False)
        toolbar.addWidget(self.record_button)

        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setStyleSheet("color: #888;")
        self.stop_button.clicked.connect(self._stop_all)
        self.stop_button.setEnabled(False)
        toolbar.addWidget(self.stop_button)

        toolbar.addStretch()

        self.camera_count_label = QLabel("No cameras")
        toolbar.addWidget(self.camera_count_label)

        layout.addLayout(toolbar)

        # === Project/Recording settings (top, always visible) ===
        settings_group = QGroupBox("Recording Settings")
        settings_layout = QHBoxLayout(settings_group)

        settings_layout.addWidget(QLabel("Project:"))
        self.output_label = QLabel("./recordings/")
        self.output_label.setFont(QFont("monospace", 9))
        self.output_label.setMinimumWidth(150)
        settings_layout.addWidget(self.output_label)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_output)
        settings_layout.addWidget(self.browse_button)

        settings_layout.addSpacing(20)

        settings_layout.addWidget(QLabel("Duration:"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 3600.0)
        self.duration_spin.setValue(10.0)
        self.duration_spin.setSuffix("s")
        self.duration_spin.setFixedWidth(80)
        settings_layout.addWidget(self.duration_spin)

        settings_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        self.fps_spin.setFixedWidth(60)
        self.fps_spin.valueChanged.connect(lambda v: self.fps_graph.set_target_fps(v))
        settings_layout.addWidget(self.fps_spin)

        settings_layout.addWidget(QLabel("Codec:"))
        self.codec_combo = QComboBox()
        self.codec_combo.addItem("H.264 (recommended)", "h264")
        self.codec_combo.addItem("H.265/HEVC", "hevc")
        self.codec_combo.addItem("MPEG-4", "mpeg4")
        self.codec_combo.setFixedWidth(140)
        settings_layout.addWidget(self.codec_combo)

        settings_layout.addStretch()
        layout.addWidget(settings_group)

        # === Main vertical splitter ===
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # === Top: Camera table + FPS graph (horizontal split) ===
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Camera table
        camera_group = QGroupBox("Cameras")
        camera_layout = QVBoxLayout(camera_group)

        self.camera_table = QTableWidget()
        self.camera_table.setColumnCount(6)
        self.camera_table.setHorizontalHeaderLabels(
            ["", "Port", "Name", "Resolution", "Exposure", "Enabled"]
        )
        header = self.camera_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(0, 24)  # Color
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, 40)  # Port
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)  # Name
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(3, 100)  # Resolution
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(4, 80)  # Exposure
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(5, 60)  # Enabled
        self.camera_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        camera_layout.addWidget(self.camera_table)

        top_splitter.addWidget(camera_group)

        # Right: FPS graph
        fps_group = QGroupBox("Frame Rate")
        fps_layout = QVBoxLayout(fps_group)
        self.fps_graph = FpsGraphWidget(buffer_size=120)
        fps_layout.addWidget(self.fps_graph)
        top_splitter.addWidget(fps_group)

        top_splitter.setSizes([400, 300])
        main_splitter.addWidget(top_splitter)

        # === Bottom: Preview grid ===
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.camera_grid = CameraGrid()
        self.camera_grid.setMinimumHeight(200)
        preview_layout.addWidget(self.camera_grid)
        main_splitter.addWidget(preview_group)

        main_splitter.setSizes([150, 400])
        layout.addWidget(main_splitter, stretch=1)

        # === Progress bar ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # === Log ===
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("monospace", 9))
        self.log_text.setMaximumHeight(100)
        self.log_text.setVisible(False)
        layout.addWidget(self.log_text)

    def _connect_signals(self):
        self.state_manager.cameras_changed.connect(self._on_cameras_changed)
        self.state_manager.frame_received.connect(self._on_frame_received)

    # ── Camera enumeration ──

    def refresh_cameras(self):
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
        camera_states = {}
        for cam in cameras:
            camera_states[cam.device_index] = CameraState(
                info=cam, enabled=True, is_open=False
            )

        self.state_manager.set_cameras(camera_states)
        self._update_camera_table(camera_states)

        count = len(cameras)
        self.camera_count_label.setText(f"{count} camera(s)")
        self.preview_button.setEnabled(count > 0)
        self.record_button.setEnabled(count > 0)
        self.status_message.emit(f"Found {count} camera(s)")

    def _on_enumerate_error(self, error: str):
        self.status_message.emit(f"Error: {error}")

    # ── Camera table ──

    def _update_camera_table(self, cameras: dict[int, CameraState]):
        self._updating_table = True
        self.camera_table.setRowCount(len(cameras))

        for row, (port, cam_state) in enumerate(sorted(cameras.items())):
            info = cam_state.info

            # Color indicator
            color = camera_color(port)
            color_widget = QWidget()
            color_widget.setFixedSize(16, 16)
            color_widget.setStyleSheet(
                f"background-color: {color.name()}; border-radius: 3px;"
            )
            color_container = QWidget()
            color_layout = QHBoxLayout(color_container)
            color_layout.addWidget(color_widget)
            color_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            color_layout.setContentsMargins(2, 2, 2, 2)
            self.camera_table.setCellWidget(row, 0, color_container)

            # Port
            port_item = QTableWidgetItem(str(port))
            port_item.setFlags(port_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 1, port_item)

            # Name (brand + serial for unique identification)
            name_text = f"{info.display_name} [{info.serial_number}]"
            name_item = QTableWidgetItem(name_text)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 2, name_item)

            # Resolution dropdown
            res_combo = QComboBox()
            resolutions = info.supported_resolutions
            for w, h in resolutions:
                res_combo.addItem(f"{w}x{h}", (w, h))
            # Select 640x480 by default (fast preview), fallback to current
            selected_idx = 0
            for idx in range(res_combo.count()):
                res = res_combo.itemData(idx)
                if res == (640, 480):
                    selected_idx = idx
                    break
                elif res == (info.width, info.height):
                    selected_idx = idx
            res_combo.setCurrentIndex(selected_idx)
            res_combo.currentIndexChanged.connect(
                lambda idx, p=port, cb=res_combo: self._on_resolution_changed(p, cb)
            )
            self.camera_table.setCellWidget(row, 3, res_combo)

            # Exposure spinbox
            exposure_spin = QSpinBox()
            exposure_spin.setRange(-13, 0)
            exposure_spin.setValue(info.exposure)
            exposure_spin.setToolTip("Exposure (log2 seconds, e.g. -4 = 1/16s)")
            exposure_spin.valueChanged.connect(
                lambda val, p=port: self._on_exposure_changed(p, val)
            )
            self.camera_table.setCellWidget(row, 4, exposure_spin)

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
            self.camera_table.setCellWidget(row, 5, checkbox_widget)

        self._updating_table = False

    def _on_enabled_changed(self, port: int, state: int):
        enabled = state == Qt.CheckState.Checked.value
        self.state_manager.update_camera(port, enabled=enabled)

    def _on_resolution_changed(self, port: int, combo: QComboBox):
        if self._updating_table:
            return
        res = combo.currentData()
        if res is None:
            return

        # Apply live if camera is open
        cameras = self.state_manager.state.cameras
        cam_state = cameras.get(port)
        if cam_state and cam_state.is_open:
            try:
                from ...camera_binding import set_resolution
                set_resolution(cam_state.info, res[0], res[1])
                self.status_message.emit(f"Port {port}: resolution changed to {res[0]}x{res[1]}")
            except Exception as e:
                self.status_message.emit(f"Port {port}: resolution change failed: {e}")
        else:
            self.status_message.emit(f"Port {port}: resolution set to {res[0]}x{res[1]} (will apply on preview)")

    def _on_exposure_changed(self, port: int, value: int):
        if self._updating_table:
            return

        cameras = self.state_manager.state.cameras
        cam_state = cameras.get(port)
        if cam_state and cam_state.is_open:
            try:
                from ...camera_binding import set_exposure
                set_exposure(cam_state.info, value)
                self.status_message.emit(f"Port {port}: exposure set to {value}")
            except Exception as e:
                self.status_message.emit(f"Port {port}: exposure change failed: {e}")
        else:
            self.status_message.emit(f"Port {port}: exposure set to {value} (will apply on preview)")

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.base_output_path = Path(folder)
            self.output_label.setText(str(folder) + "/")

    # ── Open/close cameras ──

    def _get_resolution_for_port(self, port: int) -> tuple[int, int]:
        """Get selected resolution from table for a port."""
        cameras = self.state_manager.state.cameras
        for row, (p, _) in enumerate(sorted(cameras.items())):
            if p == port:
                res_widget = self.camera_table.cellWidget(row, 3)
                if res_widget and isinstance(res_widget, QComboBox):
                    res = res_widget.currentData()
                    if res:
                        return res
        return (1280, 720)

    def _get_exposure_for_port(self, port: int) -> int:
        """Get selected exposure from table for a port."""
        cameras = self.state_manager.state.cameras
        for row, (p, _) in enumerate(sorted(cameras.items())):
            if p == port:
                exp_widget = self.camera_table.cellWidget(row, 4)
                if exp_widget and isinstance(exp_widget, QSpinBox):
                    return exp_widget.value()
        return -4

    def _open_cameras(self) -> dict[int, str]:
        """Open enabled cameras. Returns camera_info dict."""
        from ...camera_binding import open_camera, set_resolution, set_exposure

        cameras = self.state_manager.state.cameras
        self.opened_cameras = []
        camera_info = {}

        for port, cam_state in cameras.items():
            if not cam_state.enabled:
                continue
            try:
                cam = cam_state.info
                open_camera(cam)

                # Apply selected resolution
                w, h = self._get_resolution_for_port(port)
                set_resolution(cam, w, h)

                # Apply selected exposure
                exposure = self._get_exposure_for_port(port)
                set_exposure(cam, exposure)

                self.opened_cameras.append(cam)
                camera_info[port] = cam.display_name
                self.state_manager.update_camera(port, is_open=True)
            except Exception as e:
                self.status_message.emit(f"Failed to open camera {port}: {e}")

        return camera_info

    def _close_cameras(self):
        from ...camera_binding import close_camera

        for cam in self.opened_cameras:
            try:
                close_camera(cam)
            except Exception:
                pass

        for port in self.state_manager.state.cameras:
            self.state_manager.update_camera(port, is_open=False)

        self.opened_cameras = []

    # ── Preview ──

    def toggle_preview(self):
        if self.preview_button.isChecked():
            self.start_preview()
        else:
            self.stop_preview()

    def start_preview(self):
        camera_info = self._open_cameras()

        if not self.opened_cameras:
            self.preview_button.setChecked(False)
            self.status_message.emit("No cameras opened")
            return

        self.camera_grid.set_cameras(camera_info)
        self.fps_graph.set_cameras(camera_info)
        self.fps_graph.set_target_fps(self.fps_spin.value())
        self._last_frame_time.clear()

        fps = self.fps_spin.value()
        self.preview_worker = CameraPreviewWorker(self.opened_cameras, fps=fps)
        self.preview_worker.frame_captured.connect(self._on_frame_received)
        self.preview_worker.error.connect(self._on_preview_error)
        self.preview_worker.start()

        self.preview_button.setText("⏸ Stop")
        self.record_button.setEnabled(True)
        self.state_manager.update_state(is_previewing=True)
        self.status_message.emit("Preview started")

    def stop_preview(self):
        if self.preview_worker:
            self.preview_worker.stop()
            self.preview_worker.wait()
            self.preview_worker = None

        if not self._is_recording:
            self._close_cameras()
            self.camera_grid.clear_all()

        self.preview_button.setText("▶ Preview")
        self.preview_button.setChecked(False)
        self.state_manager.update_state(is_previewing=False)
        self.status_message.emit("Preview stopped")

    # ── Recording ──

    def _start_recording(self):
        # Stop preview if running
        if self.preview_worker:
            self.preview_worker.stop()
            self.preview_worker.wait()
            self.preview_worker = None

        # Open cameras if not already
        if not self.opened_cameras:
            camera_info = self._open_cameras()
            if not self.opened_cameras:
                self.status_message.emit("No cameras could be opened")
                return
            self.camera_grid.set_cameras(camera_info)
            self.fps_graph.set_cameras(camera_info)
            self._last_frame_time.clear()

        self._is_recording = True

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = self.base_output_path / timestamp
        self.output_path.mkdir(parents=True, exist_ok=True)

        duration = self.duration_spin.value()
        fps = self.fps_spin.value()

        # Show log
        self.log_text.setVisible(True)
        self.log_text.clear()
        self._log(f"Recording to {self.output_path}")
        self._log(f"Duration: {duration}s, FPS: {fps}")

        # UI state
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_button.setStyleSheet("color: #ff6666; font-weight: bold;")
        self.refresh_button.setEnabled(False)
        self.preview_button.setEnabled(False)

        total_frames = int(duration * fps)
        self.progress_bar.setMaximum(total_frames)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # Start worker
        codec = self.codec_combo.currentData()
        self.recording_worker = RecordingWorker(
            self.opened_cameras, self.output_path, duration, fps, codec=codec
        )
        self.recording_worker.log_message.connect(self._log)
        self.recording_worker.progress_update.connect(self._on_record_progress)
        self.recording_worker.frame_captured.connect(self._on_frame_received)
        self.recording_worker.recording_finished.connect(self._on_record_finished)
        self.recording_worker.error.connect(self._on_record_error)
        self.recording_worker.start()

        self.state_manager.update_recording(is_recording=True, output_path=self.output_path)
        self.status_message.emit("Recording...")

    def _stop_all(self):
        if self.recording_worker:
            self._log("Stopping...")
            self.recording_worker.stop()
        elif self.preview_worker:
            self.stop_preview()
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("color: #888;")

    def _on_record_progress(self, current: int, total: int):
        self.progress_bar.setValue(current)

    def _on_record_finished(self, stats: dict):
        self._is_recording = False
        self._close_cameras()

        # UI
        self.record_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("color: #888;")
        self.refresh_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.preview_button.setChecked(False)
        self.progress_bar.setVisible(False)

        sync_count = stats.get("sync_count", 0)
        duration = stats.get("duration", 0)
        self._log(f"Complete: {sync_count} frames, {duration:.2f}s")
        self._log(f"Output: {self.output_path}")

        self.camera_grid.clear_all()
        self.state_manager.update_recording(is_recording=False)
        self.status_message.emit("Recording complete")

    def _on_record_error(self, error: str):
        self._log(f"ERROR: {error}")
        self._on_record_finished({"sync_count": 0, "duration": 0})

    # ── Frame handling ──

    def _on_frame_received(self, port: int, frame):
        self.camera_grid.update_frame(port, frame)

        # Compute FPS
        now = time.perf_counter()
        if port in self._last_frame_time:
            dt = now - self._last_frame_time[port]
            if dt > 0:
                self.fps_graph.push_fps(port, 1.0 / dt)
        self._last_frame_time[port] = now

    def _on_preview_error(self, error: str):
        self.stop_preview()
        self.status_message.emit(f"Preview error: {error}")

    def _on_cameras_changed(self, cameras: dict):
        self._update_camera_table(cameras)

    def _log(self, message: str):
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def showEvent(self, event):
        super().showEvent(event)
        if not self.state_manager.state.cameras:
            self.refresh_cameras()

    def hideEvent(self, event):
        super().hideEvent(event)
        if self.preview_worker and not self._is_recording:
            self.stop_preview()
