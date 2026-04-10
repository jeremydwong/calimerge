"""
Record tab - camera detection, preview, configuration, and recording.

Merges camera detection/preview with recording functionality.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from pathlib import Path
from datetime import datetime
import time

import cv2
import numpy as np

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
    QSlider,
    QSplitter,
    QProgressBar,
    QTextEdit,
    QFileDialog,
    QSizePolicy,
    QLineEdit,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QPainter, QColor, QPen

from ..widgets.camera_grid import CameraGrid
from ..state import StateManager, CameraState
from ..workers import CameraEnumerateWorker, CameraPreviewWorker, RecordingWorker, PoseDetectionWorker


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
    project_folder_changed = Signal(Path)  # emitted when the project folder changes
    save_settings_requested = Signal()     # emitted when settings should be persisted

    def __init__(self, state_manager: StateManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.state_manager = state_manager

        self.enumerate_worker: CameraEnumerateWorker | None = None
        self.preview_worker: CameraPreviewWorker | None = None
        self.recording_worker: RecordingWorker | None = None
        self.detection_worker: PoseDetectionWorker | None = None
        self._last_annotated: dict[int, "np.ndarray"] = {}  # port -> last annotated frame
        self.opened_cameras: list = []
        self.opened_ports: list[int] = []
        self.base_output_path = Path("recordings")
        self.output_path: Path | None = None
        self._is_recording = False
        # Per-serial camera preferences loaded from project settings
        self._serial_prefs: dict[str, dict] = {}

        # Restore last project folder from app settings
        try:
            from ...config import load_app_settings
            app_settings = load_app_settings()
            last = app_settings.get("last_project_folder")
            if last:
                p = Path(last)
                if p.is_dir():
                    self.base_output_path = p
        except Exception:
            pass
        self._updating_table = False
        self._cap_1080p = True  # default: hide resolutions above 1080p

        # Per-camera last-frame timestamp for FPS computation
        self._last_frame_time: dict[int, float] = {}

        # Live 3D projection state
        # _view_rotation: rotation-only 4x4 (set by "Rotate to Human")
        # _view_has_origin: True once "Set Origin at L_Ankle" has been applied
        # Default matches _DEFAULT_TRANSFORM in skeleton_view.py
        self._view_rotation: np.ndarray = np.array([
            [1,  0,  0,  0],
            [0,  0,  1,  0],
            [0, -1,  0,  0],
            [0,  0,  0,  1],
        ], dtype=float)
        self._view_has_origin: bool = False
        self._rotate_timer: QTimer | None = None
        self._rotate_countdown = 0
        self._zero_timer: QTimer | None = None
        self._zero_countdown = 0

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

        self.cap_1080p_checkbox = QCheckBox("Cap 1080p")
        self.cap_1080p_checkbox.setChecked(self._cap_1080p)
        self.cap_1080p_checkbox.setToolTip("Hide resolutions above 1920x1080")
        self.cap_1080p_checkbox.stateChanged.connect(self._on_cap_1080p_changed)
        toolbar.addWidget(self.cap_1080p_checkbox)

        self.detect_checkbox = QCheckBox("Live Detection")
        self.detect_checkbox.setChecked(False)
        self.detect_checkbox.setToolTip("Overlay 2D pose detection on preview (YOLO + VitPose)")
        self.detect_checkbox.toggled.connect(self._on_detect_toggled)
        toolbar.addWidget(self.detect_checkbox)

        # Detection confidence threshold slider (for YOLO person detector)
        self.detect_conf_label = QLabel("Det: 0.30")
        self.detect_conf_label.setToolTip("Person detection confidence threshold")
        self.detect_conf_label.setEnabled(False)
        toolbar.addWidget(self.detect_conf_label)
        self.detect_conf_slider = QSlider(Qt.Horizontal)
        self.detect_conf_slider.setRange(5, 95)  # 0.05 to 0.95
        self.detect_conf_slider.setValue(30)
        self.detect_conf_slider.setFixedWidth(80)
        self.detect_conf_slider.setEnabled(False)
        self.detect_conf_slider.setToolTip("Person detection confidence threshold")
        self.detect_conf_slider.valueChanged.connect(self._on_detect_conf_changed)
        toolbar.addWidget(self.detect_conf_slider)

        # IoU threshold slider (for person tracking across frames)
        self.match_thresh_label = QLabel("IoU: 0.20")
        self.match_thresh_label.setToolTip("Bounding box IoU threshold for person tracking")
        self.match_thresh_label.setEnabled(False)
        toolbar.addWidget(self.match_thresh_label)
        self.match_thresh_slider = QSlider(Qt.Horizontal)
        self.match_thresh_slider.setRange(5, 95)  # 0.05 to 0.95
        self.match_thresh_slider.setValue(20)
        self.match_thresh_slider.setFixedWidth(80)
        self.match_thresh_slider.setEnabled(False)
        self.match_thresh_slider.setToolTip("Bounding box IoU threshold for person tracking")
        self.match_thresh_slider.valueChanged.connect(self._on_match_thresh_changed)
        toolbar.addWidget(self.match_thresh_slider)

        toolbar.addStretch()

        self.camera_count_label = QLabel("No cameras")
        toolbar.addWidget(self.camera_count_label)

        layout.addLayout(toolbar)

        # === Project/Recording settings (top, always visible) ===
        settings_group = QGroupBox("Recording Settings")
        settings_layout = QHBoxLayout(settings_group)

        settings_layout.addWidget(QLabel("Project:"))
        self.output_label = QLabel(str(self.base_output_path) + "/")
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
        self.camera_table.setColumnCount(8)
        self.camera_table.setHorizontalHeaderLabels(
            ["", "Port", "Nickname", "Name", "Resolution", "Exposure", "Brightness", "Enabled"]
        )
        header = self.camera_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(0, 24)  # Color
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, 40)  # Port
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(2, 60)  # Nickname
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)  # Name
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(4, 100)  # Resolution
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(5, 80)  # Exposure
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(6, 80)  # Brightness
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(7, 60)  # Enabled
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

        # === Bottom: Preview grid + skeleton view (horizontal split) ===
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.camera_grid = CameraGrid()
        self.camera_grid.setMinimumHeight(200)
        preview_layout.addWidget(self.camera_grid)

        from ..widgets.skeleton_view import SkeletonViewWidget
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.addWidget(preview_group)

        # Skeleton view panel
        skel_panel = QWidget()
        skel_layout = QVBoxLayout(skel_panel)
        skel_layout.setContentsMargins(4, 4, 4, 4)
        skel_header = QHBoxLayout()
        skel_label = QLabel("Live 3D Projection")
        skel_label.setFont(QFont("monospace", 9))
        skel_header.addWidget(skel_label)
        skel_header.addStretch()
        self.rotate_to_human_button = QPushButton("Rotate to Human")
        self.rotate_to_human_button.setEnabled(False)
        self.rotate_to_human_button.setToolTip(
            "Orient view: Y=up (head), X=foot-to-foot, Z=forward. Stand still, triggers in 3s."
        )
        self.rotate_to_human_button.clicked.connect(self._on_rotate_to_human)
        skel_header.addWidget(self.rotate_to_human_button)

        self.zero_origin_button = QPushButton("Zero at L_Ankle")
        self.zero_origin_button.setEnabled(False)
        self.zero_origin_button.setToolTip(
            "Set left ankle as floor origin (0,0,0). Stand still, triggers in 3s."
        )
        self.zero_origin_button.clicked.connect(self._on_zero_at_left_foot)
        skel_header.addWidget(self.zero_origin_button)

        skel_layout.addLayout(skel_header)
        self.skeleton_view = SkeletonViewWidget()
        skel_layout.addWidget(self.skeleton_view)
        bottom_splitter.addWidget(skel_panel)
        bottom_splitter.setSizes([600, 300])

        main_splitter.addWidget(bottom_splitter)
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
        # NOTE: cameras_changed is NOT connected here — cameras_tab owns the table
        # and rebuilds it explicitly (on enumerate, settings load, cap-1080p toggle).
        # Connecting it caused every update_camera() call to destroy/recreate combos.
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
        # Load nicknames from DB
        from ...config import load_all_nicknames
        nicknames = load_all_nicknames()

        camera_states = {}
        for port, cam in enumerate(cameras):
            nickname = nicknames.get(cam.serial_number, "")
            # Seed selected_resolution from saved prefs so intrinsic tab sees it immediately
            pref = self._serial_prefs.get(cam.serial_number, {})
            raw_res = pref.get("resolution")
            initial_res = tuple(raw_res) if raw_res else None
            pref_enabled = pref.get("enabled", True)
            camera_states[port] = CameraState(
                info=cam, enabled=pref_enabled, is_open=False, nickname=nickname,
                selected_resolution=initial_res,
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

    def _save_table_settings(self) -> dict[int, dict]:
        """Save current resolution/exposure/enabled/nickname from table widgets."""
        settings = {}
        cameras = self.state_manager.state.cameras
        for row, (port, _) in enumerate(sorted(cameras.items())):
            if row >= self.camera_table.rowCount():
                break
            entry = {}
            nick_widget = self.camera_table.cellWidget(row, 2)
            if nick_widget and isinstance(nick_widget, QLineEdit):
                entry["nickname"] = nick_widget.text()
            res_widget = self.camera_table.cellWidget(row, 4)
            if res_widget and isinstance(res_widget, QComboBox):
                entry["resolution"] = tuple(res_widget.currentData())
            exp_widget = self.camera_table.cellWidget(row, 5)
            if exp_widget and isinstance(exp_widget, QSpinBox):
                entry["exposure"] = exp_widget.value()
            bright_widget = self.camera_table.cellWidget(row, 6)
            if bright_widget and isinstance(bright_widget, QSpinBox):
                entry["brightness"] = bright_widget.value()
            en_widget = self.camera_table.cellWidget(row, 7)
            if en_widget:
                cb = en_widget.findChild(QCheckBox)
                if cb:
                    entry["enabled"] = cb.isChecked()
            settings[port] = entry
        return settings

    def _get_filtered_resolutions(self, resolutions: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Filter resolutions based on the 1080p cap toggle."""
        if self._cap_1080p:
            filtered = [(w, h) for w, h in resolutions if w <= 1920 and h <= 1080]
            return filtered if filtered else resolutions[:1]  # keep at least one
        return resolutions

    def _update_camera_table(self, cameras: dict[int, CameraState]):
        self._updating_table = True

        # Save current user selections before rebuilding
        saved = self._save_table_settings() if self.camera_table.rowCount() > 0 else {}

        self.camera_table.setRowCount(len(cameras))

        for row, (port, cam_state) in enumerate(sorted(cameras.items())):
            info = cam_state.info
            prev = saved.get(port, {})

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

            # Nickname (editable)
            nickname_edit = QLineEdit()
            nickname_edit.setPlaceholderText("A")
            saved_nick = prev.get("nickname", "")
            prev_nick = saved_nick if saved_nick else cam_state.nickname
            nickname_edit.setText(prev_nick)
            nickname_edit.setMaxLength(16)
            nickname_edit.editingFinished.connect(
                lambda p=port, le=nickname_edit: self._on_nickname_changed(p, le.text())
            )
            self.camera_table.setCellWidget(row, 2, nickname_edit)

            # Name (brand + serial for unique identification)
            name_text = f"{info.display_name} [{info.serial_number}]"
            name_item = QTableWidgetItem(name_text)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 3, name_item)

            # Resolution dropdown
            res_combo = QComboBox()
            all_resolutions = info.supported_resolutions
            resolutions = self._get_filtered_resolutions(all_resolutions)
            for w, h in resolutions:
                res_combo.addItem(f"{w}x{h}", (w, h))

            # Restore previous selection, fall back to serial-based project prefs,
            # then default to lowest resolution (list is sorted largest-first)
            selected_idx = res_combo.count() - 1
            prev_res = prev.get("resolution")
            if not prev_res:
                # Check project-settings prefs keyed by serial number
                serial_pref = self._serial_prefs.get(info.serial_number, {})
                raw = serial_pref.get("resolution")
                if raw:
                    prev_res = tuple(raw)
            if prev_res:
                prev_res_t = tuple(prev_res)
                for idx in range(res_combo.count()):
                    if tuple(res_combo.itemData(idx)) == prev_res_t:
                        selected_idx = idx
                        break
            res_combo.setCurrentIndex(selected_idx)
            res_combo.currentIndexChanged.connect(
                lambda idx, p=port, cb=res_combo: self._on_resolution_changed(p, cb)
            )
            self.camera_table.setCellWidget(row, 4, res_combo)

            # Exposure spinbox (read-only — use Brightness instead)
            exposure_spin = QSpinBox()
            exposure_spin.setRange(-13, 0)
            serial_pref_exp = self._serial_prefs.get(info.serial_number, {}).get("exposure")
            prev_exp = prev.get("exposure", serial_pref_exp if serial_pref_exp is not None else info.exposure)
            exposure_spin.setValue(prev_exp)
            exposure_spin.setToolTip("Exposure (log2 seconds) — read-only, use Brightness to adjust")
            exposure_spin.setReadOnly(True)
            exposure_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            exposure_spin.setStyleSheet("QSpinBox { color: #888; background: #2a2a2a; }")
            self.camera_table.setCellWidget(row, 5, exposure_spin)

            # Brightness spinbox
            brightness_spin = QSpinBox()
            brightness_spin.setRange(-13, 0)
            serial_pref_bright = self._serial_prefs.get(info.serial_number, {}).get("brightness")
            prev_bright = prev.get("brightness", serial_pref_bright if serial_pref_bright is not None else prev_exp)
            brightness_spin.setValue(prev_bright)
            brightness_spin.setToolTip("Brightness (-13 to 0, maps to camera brightness control)")
            brightness_spin.valueChanged.connect(
                lambda val, p=port: self._on_brightness_changed(p, val)
            )
            self.camera_table.setCellWidget(row, 6, brightness_spin)

            # Enabled checkbox
            checkbox = QCheckBox()
            serial_pref_en = self._serial_prefs.get(info.serial_number, {}).get("enabled")
            prev_en = prev.get("enabled", serial_pref_en if serial_pref_en is not None else cam_state.enabled)
            checkbox.setChecked(prev_en)
            checkbox.stateChanged.connect(
                lambda state, p=port: self._on_enabled_changed(p, state)
            )
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.camera_table.setCellWidget(row, 7, checkbox_widget)

        self._updating_table = False

        # Sync enabled states from widgets back to state manager (without triggering rebuild)
        if saved:
            needs_sync = False
            new_cameras = dict(cameras)
            for port, cam_state in cameras.items():
                prev = saved.get(port, {})
                prev_en = prev.get("enabled")
                if prev_en is not None and prev_en != cam_state.enabled:
                    new_cameras[port] = replace(cam_state, enabled=prev_en)
                    needs_sync = True
            if needs_sync:
                self._updating_table = True
                self.state_manager.set_cameras(new_cameras)
                self._updating_table = False

    def _on_enabled_changed(self, port: int, state: int):
        if self._updating_table:
            return
        enabled = state == Qt.CheckState.Checked.value
        self.state_manager.update_camera(port, enabled=enabled)

        # If previewing, restart preview so the change takes effect immediately
        if self.preview_worker:
            self.stop_preview()
            self.start_preview()

    def _on_nickname_changed(self, port: int, nickname: str):
        if self._updating_table:
            return
        self.state_manager.update_camera(port, nickname=nickname)
        # Persist to DB
        cam_state = self.state_manager.state.cameras.get(port)
        if cam_state:
            from ...config import save_nickname
            save_nickname(cam_state.info.serial_number, nickname)

    def _on_resolution_changed(self, port: int, combo: QComboBox):
        if self._updating_table:
            return
        res = combo.currentData()
        if res is None:
            return

        # Defer state update to next event loop iteration — avoids destroying the
        # combo widget synchronously while Qt is still processing the selection event.
        QTimer.singleShot(0, lambda: self.state_manager.update_camera(port, selected_resolution=res))

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

        # Persist immediately so the choice survives even if the app is force-closed
        self.save_settings_requested.emit()

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

    def _on_brightness_changed(self, port: int, value: int):
        if self._updating_table:
            return

        cameras = self.state_manager.state.cameras
        cam_state = cameras.get(port)
        if cam_state and cam_state.is_open:
            try:
                from ...camera_binding import set_exposure
                set_exposure(cam_state.info, value)
                self.status_message.emit(f"Port {port}: brightness set to {value}")
            except Exception as e:
                self.status_message.emit(f"Port {port}: brightness change failed: {e}")
        else:
            self.status_message.emit(f"Port {port}: brightness set to {value} (will apply on preview)")

    # ── Project settings ──

    def apply_project_settings(self, settings: dict) -> None:
        """Apply loaded project settings to this tab's UI controls."""
        # Store per-serial camera preferences for use when table is (re)built
        self._serial_prefs = settings.get("cameras", {})

        fps = settings.get("fps")
        if fps is not None:
            self.fps_spin.setValue(int(fps))

        codec = settings.get("codec", "h264")
        for i in range(self.codec_combo.count()):
            if self.codec_combo.itemData(i) == codec:
                self.codec_combo.setCurrentIndex(i)
                break

        # Rebuild table so serial prefs take effect immediately
        cameras = self.state_manager.state.cameras
        if cameras:
            self._update_camera_table(cameras)

    def get_project_settings(self) -> dict:
        """Return this tab's contribution to the project settings dict."""
        # Start with previously loaded prefs so cameras not currently enumerated
        # (unplugged, not refreshed) keep their saved settings.
        cameras_section: dict[str, dict] = {k: dict(v) for k, v in self._serial_prefs.items()}

        cam_states = self.state_manager.state.cameras
        for row, (port, cam_state) in enumerate(sorted(cam_states.items())):
            serial = cam_state.info.serial_number
            entry: dict = {}
            res_widget = self.camera_table.cellWidget(row, 4)
            if res_widget and isinstance(res_widget, QComboBox):
                res = res_widget.currentData()
                if res:
                    entry["resolution"] = list(res)
            en_widget = self.camera_table.cellWidget(row, 7)
            if en_widget:
                cb = en_widget.findChild(QCheckBox)
                if cb:
                    entry["enabled"] = cb.isChecked()
            exp_widget = self.camera_table.cellWidget(row, 5)
            if exp_widget and isinstance(exp_widget, QSpinBox):
                entry["exposure"] = exp_widget.value()
            bright_widget = self.camera_table.cellWidget(row, 6)
            if bright_widget and isinstance(bright_widget, QSpinBox):
                entry["brightness"] = bright_widget.value()
            cameras_section[serial] = entry  # overwrite with live table values

        return {
            "fps": self.fps_spin.value(),
            "codec": self.codec_combo.currentData() or "h264",
            "cameras": cameras_section,
        }

    def _on_cap_1080p_changed(self, state: int):
        self._cap_1080p = state == Qt.CheckState.Checked.value
        cameras = self.state_manager.state.cameras
        if cameras:
            self._update_camera_table(cameras)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.base_output_path = Path(folder)
            self.output_label.setText(str(folder) + "/")
            # Persist last project folder
            try:
                from ...config import load_app_settings, save_app_settings
                app_settings = load_app_settings()
                app_settings["last_project_folder"] = str(folder)
                save_app_settings(app_settings)
            except Exception:
                pass
            self.project_folder_changed.emit(self.base_output_path)
            # Load view transform from new folder if present
            self._load_view_transform()

    # ── Open/close cameras ──

    def _get_resolution_for_port(self, port: int) -> tuple[int, int]:
        """Get selected resolution from table for a port."""
        cameras = self.state_manager.state.cameras
        for row, (p, _) in enumerate(sorted(cameras.items())):
            if p == port:
                res_widget = self.camera_table.cellWidget(row, 4)
                if res_widget and isinstance(res_widget, QComboBox):
                    res = res_widget.currentData()
                    if res:
                        return res
        return (640, 480)

    def _get_exposure_for_port(self, port: int) -> int:
        """Get selected exposure from table for a port."""
        cameras = self.state_manager.state.cameras
        for row, (p, _) in enumerate(sorted(cameras.items())):
            if p == port:
                exp_widget = self.camera_table.cellWidget(row, 5)
                if exp_widget and isinstance(exp_widget, QSpinBox):
                    return exp_widget.value()
        return -4

    def _get_brightness_for_port(self, port: int) -> int:
        """Get selected brightness from table for a port."""
        cameras = self.state_manager.state.cameras
        for row, (p, _) in enumerate(sorted(cameras.items())):
            if p == port:
                bright_widget = self.camera_table.cellWidget(row, 6)
                if bright_widget and isinstance(bright_widget, QSpinBox):
                    return bright_widget.value()
        return -4

    def _is_camera_enabled_in_table(self, port: int) -> bool:
        """Read enabled state directly from the table checkbox widget."""
        cameras = self.state_manager.state.cameras
        for row, (p, _) in enumerate(sorted(cameras.items())):
            if p == port:
                en_widget = self.camera_table.cellWidget(row, 7)
                if en_widget:
                    cb = en_widget.findChild(QCheckBox)
                    if cb:
                        return cb.isChecked()
        return True  # default enabled if widget not found

    def _open_cameras(self) -> dict[int, str]:
        """Open enabled cameras and apply table settings. Returns camera_info dict."""
        from ...camera_binding import open_camera, set_resolution, set_exposure

        cameras = self.state_manager.state.cameras
        self.opened_cameras = []
        self.opened_ports = []
        camera_info = {}
        opened_ports = []

        for port, cam_state in sorted(cameras.items()):
            if not self._is_camera_enabled_in_table(port):
                continue
            try:
                cam = cam_state.info
                open_camera(cam)
                self.opened_cameras.append(cam)
                self.opened_ports.append(port)
                nick = cam_state.nickname
                camera_info[port] = nick if nick else cam.display_name
                opened_ports.append(port)
            except Exception as e:
                self.status_message.emit(f"Failed to open camera {port}: {e}")

        # Apply settings from the table widgets AFTER all cameras are open
        for port in opened_ports:
            cam_state = cameras[port]
            cam = cam_state.info
            try:
                w, h = self._get_resolution_for_port(port)
                set_resolution(cam, w, h)
            except Exception as e:
                self.status_message.emit(f"Port {port}: resolution failed: {e}")
            try:
                brightness = self._get_brightness_for_port(port)
                set_exposure(cam, brightness)
            except Exception as e:
                self.status_message.emit(f"Port {port}: brightness failed: {e}")

        # Batch-update state (triggers one table rebuild, which preserves settings)
        if opened_ports:
            new_cameras = dict(cameras)
            for port in opened_ports:
                new_cameras[port] = replace(cameras[port], is_open=True)
            self.state_manager.set_cameras(new_cameras)

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
        self.opened_ports = []

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
        self.preview_worker = CameraPreviewWorker(self.opened_cameras, self.opened_ports, fps=fps)
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

        # Stop detection worker if running
        self._stop_detection()
        self.detect_checkbox.blockSignals(True)
        self.detect_checkbox.setChecked(False)
        self.detect_checkbox.blockSignals(False)

        if not self._is_recording:
            self._close_cameras()
            self.camera_grid.clear_all()

        self.preview_button.setText("▶ Preview")
        self.preview_button.setChecked(False)
        self.state_manager.update_state(is_previewing=False)
        self.status_message.emit("Preview stopped")

    # ── Recording ──

    def _start_recording(self):
        # Stop live detection first (frees GPU for recording)
        if self.detection_worker is not None:
            self._stop_detection()
            self.detect_checkbox.blockSignals(True)
            self.detect_checkbox.setChecked(False)
            self.detect_checkbox.blockSignals(False)

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
            self.opened_cameras, self.opened_ports, self.output_path, duration, fps, codec=codec
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
        # If live detection is active, send frame to detection worker
        if self.detection_worker is not None and self.detection_worker.isRunning():
            self.detection_worker.submit_frame(port, frame)
            # Show last annotated frame (faded) instead of raw to avoid flicker
            if port in self._last_annotated:
                # Fade overlay by 5%: blend 95% last annotated + 5% raw
                blended = cv2.addWeighted(
                    self._last_annotated[port], 0.95, frame, 0.05, 0
                )
                self._last_annotated[port] = blended
                self.camera_grid.update_frame(port, blended)
            else:
                # No annotation yet for this port — show raw
                self.camera_grid.update_frame(port, frame)
        else:
            self.camera_grid.update_frame(port, frame)

        # Compute FPS
        now = time.perf_counter()
        if port in self._last_frame_time:
            dt = now - self._last_frame_time[port]
            if dt > 0:
                self.fps_graph.push_fps(port, 1.0 / dt)
        self._last_frame_time[port] = now

    def _on_detection_ready(self, port: int, annotated_frame):
        """Replace the camera grid frame with the annotated version."""
        self._last_annotated[port] = annotated_frame.copy()
        self.camera_grid.update_frame(port, annotated_frame)

    def _on_preview_error(self, error: str):
        self.stop_preview()
        self.status_message.emit(f"Preview error: {error}")

    # ── Live Detection ──

    def _on_detect_toggled(self, checked: bool):
        """Handle live detection checkbox toggle."""
        import traceback
        print(f"[detect] toggled checked={checked}")
        print(f"[detect] caller: {''.join(traceback.format_stack()[-3:-1]).strip()}")
        # Enable/disable threshold sliders
        self.detect_conf_label.setEnabled(checked)
        self.detect_conf_slider.setEnabled(checked)
        self.match_thresh_label.setEnabled(checked)
        self.match_thresh_slider.setEnabled(checked)
        if checked:
            self._start_detection()
        else:
            self._stop_detection()

    def _on_detect_conf_changed(self, value: int):
        """Update person detection confidence threshold."""
        thresh = value / 100.0
        self.detect_conf_label.setText(f"Det: {thresh:.2f}")
        if self.detection_worker is not None:
            self.detection_worker.confidence_threshold = thresh

    def _on_match_thresh_changed(self, value: int):
        """Update IoU matching threshold."""
        thresh = value / 100.0
        self.match_thresh_label.setText(f"IoU: {thresh:.2f}")
        if self.detection_worker is not None:
            self.detection_worker.match_threshold = thresh

    def _load_calibration_from_disk(self) -> dict | None:
        """Scan recordings subdirectories for the most recent calibration.toml."""
        try:
            from ...config import load_calibration_from_toml
            candidates = sorted(self.base_output_path.glob("*/calibration.toml"))
            if not candidates:
                return None
            # Last alphabetically = most recent (timestamp-named folders)
            cal_file = candidates[-1]
            cameras = load_calibration_from_toml(cal_file)
            if cameras:
                print(f"[detect] loaded calibration from {cal_file} ({len(cameras)} cameras)")
                # Persist into state so other tabs (Process) also see it
                self.state_manager.update_calibration(calibrated_cameras=cameras)
            return cameras or None
        except Exception as e:
            print(f"[detect] calibration load failed: {e}")
            return None

    def _start_detection(self):
        """Start the pose detection worker."""
        if self.detection_worker is not None:
            print("[detect] worker already exists, skipping")
            return

        # Get calibrated cameras: prefer in-memory state, fall back to disk
        cal_state = self.state_manager.state.calibration
        cameras = cal_state.calibrated_cameras if cal_state.calibrated_cameras else None

        if cameras is None:
            cameras = self._load_calibration_from_disk()

        if cameras is None:
            self.skeleton_view.set_message("No extrinsic calibration")

        print("[detect] creating PoseDetectionWorker...")
        self.detection_worker = PoseDetectionWorker(device_name="auto", cameras=cameras)
        self.detection_worker.confidence_threshold = self.detect_conf_slider.value() / 100.0
        self.detection_worker.match_threshold = self.match_thresh_slider.value() / 100.0
        self.detection_worker.models_loaded.connect(
            lambda: print("[detect] models loaded, live detection active")
        )
        self.detection_worker.detection_ready.connect(self._on_detection_ready)
        self.detection_worker.log_message.connect(lambda msg: print(f"[detect] {msg}"))
        self.detection_worker.error.connect(self._on_detection_error)
        self.detection_worker.finished.connect(self._on_detection_finished)
        if cameras is not None:
            self.detection_worker.keypoints_3d_ready.connect(self._on_keypoints_3d)
        self.detection_worker.start()
        print("[detect] worker started")

    def _stop_detection(self):
        """Stop the pose detection worker."""
        if self.detection_worker is not None:
            print("[detect] stopping worker...")
            self.detection_worker.stop()
            self.detection_worker.wait()
            self.detection_worker = None
            self._last_annotated.clear()
            print("[detect] stopped")
        self.skeleton_view.clear()
        self.rotate_to_human_button.setEnabled(False)
        self.zero_origin_button.setEnabled(False)

    def _on_detection_finished(self):
        """Handle detection worker thread finishing (could be normal or crash)."""
        print("[detect] worker thread finished")
        if self.detect_checkbox.isChecked():
            print("[detect] worker died unexpectedly while checkbox was checked")
            self.detect_checkbox.blockSignals(True)
            self.detect_checkbox.setChecked(False)
            self.detect_checkbox.blockSignals(False)
            self.detection_worker = None

    def _on_detection_error(self, error: str):
        """Handle detection worker error."""
        print(f"[detect] ERROR: {error}")
        self.detect_checkbox.blockSignals(True)
        self.detect_checkbox.setChecked(False)
        self.detect_checkbox.blockSignals(False)
        self._stop_detection()

    def _on_keypoints_3d(self, persons: list):
        """Handle multi-person 3D keypoints from live triangulation.

        persons: list[list[np.ndarray(3,) | None]]
        """
        # Clean NaNs per person
        clean_persons = []
        for kps_3d in persons:
            clean = [
                kp if (kp is not None and not np.isnan(kp).any()) else None
                for kp in kps_3d
            ]
            clean_persons.append(clean)

        self.skeleton_view.update_keypoints(clean_persons)
        has_kps = any(any(k is not None for k in p) for p in clean_persons)
        self.rotate_to_human_button.setEnabled(has_kps)
        self.zero_origin_button.setEnabled(has_kps)

    # ── Rotate to Human ──────────────────────────────────────────────────

    def _on_rotate_to_human(self):
        """Start 5s countdown then capture rotation from current skeleton."""
        self._rotate_countdown = 5
        self.rotate_to_human_button.setEnabled(False)
        self.rotate_to_human_button.setText(f"Rotating in {self._rotate_countdown}s...")
        self._rotate_timer = QTimer()
        self._rotate_timer.timeout.connect(self._rotate_countdown_tick)
        self._rotate_timer.start(1000)

    def _rotate_countdown_tick(self):
        self._rotate_countdown -= 1
        if self._rotate_countdown > 0:
            self.rotate_to_human_button.setText(f"Rotating in {self._rotate_countdown}s...")
        else:
            self._rotate_timer.stop()
            self._rotate_timer = None
            self.rotate_to_human_button.setText("Rotate to Human")
            self._compute_rotate_to_human()

    def _compute_rotate_to_human(self):
        """Compute a rotation-only view transform from the current skeleton.

        Axis definitions (user-specified):
          X  = normalize(R_Ankle − L_Ankle)       foot-to-foot / rightward
          Z  = normalize(head − avg_feet)          body-up
          Y  = normalize(cross(Z, X))              forward / depth

        X and Z are orthogonalised via Gram-Schmidt so the frame is
        guaranteed orthonormal.  Stores rotation-only (no translation);
        the widget auto-centres on the body until "Set Origin at L_Ankle"
        is pressed.
        """
        kps = self.skeleton_view.get_keypoints()
        if not kps or not any(k is not None for k in kps):
            self.rotate_to_human_button.setEnabled(True)
            return

        def get_pt(idx):
            if idx < len(kps) and kps[idx] is not None:
                return np.array(kps[idx], dtype=float)
            return None

        # COCO-17 indices
        l_ankle = get_pt(15)
        r_ankle = get_pt(16)
        nose    = get_pt(0)
        l_hip   = get_pt(11)
        r_hip   = get_pt(12)
        l_sho   = get_pt(5)
        r_sho   = get_pt(6)

        # ── Z axis: avg_feet → head (body-up) ──
        foot_ref = (
            (l_ankle + r_ankle) / 2 if l_ankle is not None and r_ankle is not None
            else l_ankle if l_ankle is not None
            else r_ankle if r_ankle is not None
            else (l_hip + r_hip) / 2 if l_hip is not None and r_hip is not None
            else None
        )
        head_ref = (
            nose if nose is not None
            else (l_sho + r_sho) / 2 if l_sho is not None and r_sho is not None
            else (l_hip + r_hip) / 2 if l_hip is not None and r_hip is not None
            else None
        )
        if foot_ref is None or head_ref is None:
            self.rotate_to_human_button.setEnabled(True)
            return

        Z = head_ref - foot_ref
        z_norm = np.linalg.norm(Z)
        if z_norm < 0.01:
            self.rotate_to_human_button.setEnabled(True)
            return
        Z = Z / z_norm

        # ── X axis: L_Ankle → R_Ankle, orthogonalised against Z ──
        if l_ankle is not None and r_ankle is not None:
            X_raw = r_ankle - l_ankle
        elif l_hip is not None and r_hip is not None:
            X_raw = r_hip - l_hip
        else:
            X_raw = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(X_raw, Z)) > 0.9:
                X_raw = np.array([0.0, 0.0, 1.0])

        X = X_raw - np.dot(X_raw, Z) * Z   # Gram-Schmidt
        x_norm = np.linalg.norm(X)
        if x_norm < 1e-4:
            self.rotate_to_human_button.setEnabled(True)
            return
        X = X / x_norm

        # ── Y axis: forward = cross(Z, X) ──
        Y = np.cross(Z, X)
        Y = Y / np.linalg.norm(Y)

        # R columns are the world vectors that map to view X, Y, Z.
        # R.T (the inverse rotation) maps world → view space.
        R = np.column_stack([X, Y, Z])
        T = np.eye(4)
        T[:3, :3] = R.T   # rotation only, no translation

        self._view_rotation = T
        self._view_has_origin = False
        self.skeleton_view.set_view_transform(T, has_origin=False)
        self._save_view_transform(T, has_origin=False)
        self.rotate_to_human_button.setEnabled(True)

    # ── Zero at Left Foot ─────────────────────────────────────────────────

    def _on_zero_at_left_foot(self):
        """Start 5s countdown then anchor view origin to current L_Ankle."""
        self._zero_countdown = 5
        self.zero_origin_button.setEnabled(False)
        self.zero_origin_button.setText(f"Zeroing in {self._zero_countdown}s...")
        self._zero_timer = QTimer()
        self._zero_timer.timeout.connect(self._zero_countdown_tick)
        self._zero_timer.start(1000)

    def _zero_countdown_tick(self):
        self._zero_countdown -= 1
        if self._zero_countdown > 0:
            self.zero_origin_button.setText(f"Zeroing in {self._zero_countdown}s...")
        else:
            self._zero_timer.stop()
            self._zero_timer = None
            self.zero_origin_button.setText("Zero at L_Ankle")
            self._compute_zero_origin()

    def _compute_zero_origin(self):
        """Translate the view so the current L_Ankle maps to view origin (0,0,0).

        Composites the stored rotation (_view_rotation) with a translation
        that places L_Ankle at the origin.  Enables the floor grid.
        """
        kps = self.skeleton_view.get_keypoints()
        l_ankle = None
        if kps and len(kps) > 15 and kps[15] is not None:
            l_ankle = np.array(kps[15], dtype=float)

        if l_ankle is None:
            self.zero_origin_button.setEnabled(True)
            return

        R = self._view_rotation[:3, :3]   # rotation part only
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ l_ankle   # translate so L_Ankle → view origin

        self._view_has_origin = True
        self.skeleton_view.set_view_transform(T, has_origin=True)
        self._save_view_transform(T, has_origin=True)
        self._save_body_transform(R, l_ankle)
        self.zero_origin_button.setEnabled(True)

    def _save_view_transform(self, T: np.ndarray, has_origin: bool = False):
        """Save view transform to camera_rig.toml in the project folder."""
        try:
            import rtoml
            rig_path = self._get_camera_rig_path()
            if rig_path is None:
                return
            data = {}
            if rig_path.exists():
                data = rtoml.load(rig_path)
            data["live_view"] = {
                "transform": T.flatten().tolist(),
                "has_origin": has_origin,
            }
            with open(rig_path, "w") as f:
                rtoml.dump(data, f)
        except Exception:
            pass

    def _save_body_transform(self, R: np.ndarray, origin: np.ndarray):
        """Save the world-to-body transform (rotation + origin) to camera_rig.toml.

        This records the coordinate frame where L_Ankle is at origin with
        anatomically meaningful axes, so recorded poses can be expressed
        in body-centred coordinates.
        """
        try:
            import rtoml
            rig_path = self._get_camera_rig_path()
            if rig_path is None:
                return
            data = {}
            if rig_path.exists():
                data = rtoml.load(rig_path)
            data["body_transform"] = {
                "rotation": R.flatten().tolist(),
                "origin_world": origin.tolist(),
                "description": "World-to-body transform: R rotates world axes to body axes, origin is L_Ankle in world coords",
            }
            with open(rig_path, "w") as f:
                rtoml.dump(data, f)
        except Exception:
            pass

    def _get_camera_rig_path(self) -> Path | None:
        if self.base_output_path:
            return self.base_output_path / "camera_rig.toml"
        return None

    def _load_view_transform(self):
        """Load view transform from camera_rig.toml if present."""
        try:
            import rtoml
            rig_path = self._get_camera_rig_path()
            if rig_path is None or not rig_path.exists():
                return
            data = rtoml.load(rig_path)
            lv = data.get("live_view", {})
            if "transform" in lv:
                T = np.array(lv["transform"]).reshape(4, 4)
                has_origin = bool(lv.get("has_origin", False))
                if not has_origin:
                    self._view_rotation = T
                self._view_has_origin = has_origin
                self.skeleton_view.set_view_transform(T, has_origin=has_origin)
        except Exception:
            pass

    def _log(self, message: str):
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def showEvent(self, event):
        super().showEvent(event)
        if not self.state_manager.state.cameras:
            self.refresh_cameras()
        elif self.preview_worker and not self._is_recording:
            # Resume preview when returning to this tab
            self.preview_worker.resume()
            self.status_message.emit("Preview resumed")

    def hideEvent(self, event):
        super().hideEvent(event)
        if self.preview_worker and not self._is_recording:
            # Pause preview when leaving tab (cameras stay open)
            self.preview_worker.pause()
            self.status_message.emit("Preview paused")
