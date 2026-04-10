"""
Workout playback dialog — synchronized video + 3D keypoint replay.

Shows each camera's video side-by-side with a single master play control
and a 3D skeleton view driven by the saved keypoints_3d.npz file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QSplitter,
    QSlider,
    QDialogButtonBox,
    QSizePolicy,
)
from PySide6.QtGui import QFont

from .widgets.video_player import VideoPlayer
from .widgets.skeleton_view import SkeletonViewWidget
from ..types import extrinsic_to_view_transform


class WorkoutPlaybackDialog(QDialog):
    """Playback dialog with synchronized videos + 3D skeleton replay."""

    def __init__(self, session: dict, parent=None):
        super().__init__(parent)
        self.session = session
        self.rec_dir = Path(session["recording_path"])

        self.players: list[VideoPlayer] = []
        self.keypoints: np.ndarray | None = None
        self.kp_timestamps: np.ndarray | None = None
        self.kp_view_transform: np.ndarray | None = None

        # video_frame_index → keypoints index (precomputed on load)
        self._video_to_kp: np.ndarray | None = None

        self.master_fps: float = 30.0
        self.total_frames: int = 0
        self.current_frame: int = 0
        self.is_playing: bool = False

        self.setWindowTitle(f"Playback — {self.rec_dir.name}")
        self.setMinimumSize(1000, 650)

        self._init_ui()
        self._load_videos()
        self._load_keypoints()
        self._build_video_to_kp_mapping()
        self._load_view_transform_from_session()

        # Master playback timer
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._advance_frame)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Top: videos (left) + 3D skeleton (right) splitter
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Video panel — one VideoPlayer per camera
        video_panel = QWidget()
        self.video_layout = QHBoxLayout(video_panel)
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        video_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        top_splitter.addWidget(video_panel)

        # 3D skeleton view
        skel_panel = QWidget()
        skel_layout = QVBoxLayout(skel_panel)
        skel_layout.setContentsMargins(4, 4, 4, 4)
        skel_label = QLabel("3D Replay")
        skel_label.setFont(QFont("monospace", 9))
        skel_layout.addWidget(skel_label)
        self.skeleton_view = SkeletonViewWidget()
        skel_layout.addWidget(self.skeleton_view)
        top_splitter.addWidget(skel_panel)

        top_splitter.setSizes([700, 300])
        layout.addWidget(top_splitter, stretch=1)

        # Master controls
        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setMinimumWidth(80)
        self.play_button.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_button)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        controls.addWidget(self.slider, stretch=1)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setFixedWidth(100)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        controls.addWidget(self.frame_label)

        layout.addLayout(controls)

        # Close button
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def _load_videos(self):
        """Load all port_*.mp4 files from the session directory."""
        video_files = sorted(self.rec_dir.glob("port_*.mp4"))
        if not video_files:
            label = QLabel("No videos found in this session.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video_layout.addWidget(label)
            return

        for video in video_files:
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(2, 2, 2, 2)
            name_label = QLabel(video.name)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            panel_layout.addWidget(name_label)

            player = VideoPlayer()
            if player.load_video(video):
                # Hide per-player play button and slider — we drive everything from master
                player.play_button.hide()
                player.slider.hide()
                player.frame_label_info.hide()
                self.players.append(player)
                if self.total_frames == 0 or player.total_frames < self.total_frames:
                    self.total_frames = player.total_frames
                if self.master_fps == 30.0 and player.fps > 0:
                    self.master_fps = player.fps
            panel_layout.addWidget(player)
            self.video_layout.addWidget(panel)

        # Update slider range
        if self.total_frames > 0:
            self.slider.setRange(0, self.total_frames - 1)
            self.frame_label.setText(f"1 / {self.total_frames}")

    def _load_keypoints(self):
        """Load keypoints_3d.npz if it exists."""
        kp_file = self.rec_dir / "keypoints_3d.npz"
        if not kp_file.exists():
            self.skeleton_view.set_message("No keypoints_3d.npz in session")
            return

        try:
            from ..analysis.keypoints_io import load_keypoints_3d
            data = load_keypoints_3d(kp_file)
            self.keypoints = data["keypoints_3d"]         # (N, P, K, 3)
            self.kp_timestamps = data["timestamps"]       # (N,)
        except Exception as e:
            self.skeleton_view.set_message(f"Failed to load keypoints: {e}")
            return

        # If keypoints span more frames than the video, cap at keypoint count
        if self.total_frames == 0 and len(self.kp_timestamps) > 0:
            self.total_frames = len(self.kp_timestamps)
            self.slider.setRange(0, self.total_frames - 1)

    def _build_video_to_kp_mapping(self):
        """For each video frame, find the keypoint entry with the closest timestamp.

        Used during playback to drive the skeleton view in lock-step with the
        videos even when keypoints and video frames are not at the same rate.
        """
        if self.keypoints is None or self.kp_timestamps is None:
            return
        if self.total_frames == 0 or self.master_fps <= 0:
            return

        kp_times = np.asarray(self.kp_timestamps, dtype=float)
        if len(kp_times) == 0:
            return

        video_times = np.arange(self.total_frames, dtype=float) / self.master_fps

        # For each video frame time, find the nearest keypoint index.
        # np.searchsorted gives us the insertion point; we then compare
        # neighbours to pick whichever is closer.
        idx = np.searchsorted(kp_times, video_times)
        idx = np.clip(idx, 0, len(kp_times) - 1)
        prev_idx = np.clip(idx - 1, 0, len(kp_times) - 1)
        pick_prev = (idx > 0) & (
            np.abs(kp_times[prev_idx] - video_times)
            < np.abs(kp_times[idx] - video_times)
        )
        idx[pick_prev] = prev_idx[pick_prev]

        self._video_to_kp = idx.astype(np.int64)

    def _load_view_transform_from_session(self):
        """Set the 3D view to match the first calibrated camera's extrinsic.

        Uses the compact config_blob stored on the session row if present,
        otherwise falls back to the camera_rig.toml live_view transform.
        """
        # 1) Try to reconstruct from the session config blob
        try:
            from ..config import get_session_config
            from ..types import CameraExtrinsics
            from ..types import compute_transformation_matrix
            session_id = self.session.get("id")
            if session_id is not None:
                config = get_session_config(int(session_id))
                if config:
                    first_port = sorted(config.keys())[0]
                    cam = config[first_port]
                    extr = CameraExtrinsics(
                        rotation=np.asarray(cam["rotation"]),
                        translation=np.asarray(cam["translation"]),
                    )
                    T = extrinsic_to_view_transform(extr)
                    self.kp_view_transform = T
                    self.skeleton_view.set_view_transform(T, has_origin=False)
                    return
        except Exception:
            pass

        # 2) Fall back to the project-level camera_rig.toml live_view entry
        try:
            import rtoml
            from ..config import load_app_settings
            app = load_app_settings()
            folder = app.get("last_project_folder")
            if not folder:
                return
            rig_path = Path(folder) / "camera_rig.toml"
            if not rig_path.exists():
                return
            data = rtoml.load(rig_path)
            lv = data.get("live_view", {})
            if "transform" in lv:
                T = np.array(lv["transform"]).reshape(4, 4)
                has_origin = bool(lv.get("has_origin", False))
                self.kp_view_transform = T
                self.skeleton_view.set_view_transform(T, has_origin=has_origin)
        except Exception:
            pass

    def _toggle_play(self):
        if self.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if self.total_frames == 0:
            return
        if self.current_frame >= self.total_frames - 1:
            self._seek(0)
        self.is_playing = True
        self.play_button.setText("Pause")
        interval_ms = int(1000.0 / self.master_fps)
        self.play_timer.start(interval_ms)

    def _pause(self):
        self.is_playing = False
        self.play_timer.stop()
        self.play_button.setText("Play")

    def _advance_frame(self):
        if self.current_frame >= self.total_frames - 1:
            self._pause()
            return
        self._seek(self.current_frame + 1, from_timer=True)

    def _on_slider_changed(self, value: int):
        if not self.is_playing:
            self._seek(value, from_slider=True)

    def _seek(self, frame_index: int, from_slider: bool = False, from_timer: bool = False):
        if self.total_frames == 0:
            return
        frame_index = max(0, min(frame_index, self.total_frames - 1))
        self.current_frame = frame_index

        # Sync all video players
        for player in self.players:
            player._show_frame(frame_index)  # direct call bypasses play state

        # Sync slider
        if not from_slider:
            self.slider.blockSignals(True)
            self.slider.setValue(frame_index)
            self.slider.blockSignals(False)

        self.frame_label.setText(f"{frame_index + 1} / {self.total_frames}")

        # Sync skeleton
        self._update_skeleton(frame_index)

    def _update_skeleton(self, frame_index: int):
        """Push the keypoints at the video frame's timestamp into the skeleton view."""
        if self.keypoints is None:
            return

        # Map video frame → keypoint entry (precomputed by timestamp closeness)
        if self._video_to_kp is not None and 0 <= frame_index < len(self._video_to_kp):
            kp_idx = int(self._video_to_kp[frame_index])
        else:
            kp_idx = min(frame_index, len(self.keypoints) - 1)

        if kp_idx < 0 or kp_idx >= len(self.keypoints):
            return

        frame_kps = self.keypoints[kp_idx]
        persons = []
        for p_idx in range(frame_kps.shape[0]):
            person_kps = []
            any_valid = False
            for k_idx in range(frame_kps.shape[1]):
                kp = frame_kps[p_idx, k_idx]
                if np.isnan(kp).any():
                    person_kps.append(None)
                else:
                    person_kps.append(kp)
                    any_valid = True
            if any_valid:
                persons.append(person_kps)

        self.skeleton_view.update_keypoints(persons)

    def closeEvent(self, event):
        self._pause()
        for player in self.players:
            player.unload()
        super().closeEvent(event)
