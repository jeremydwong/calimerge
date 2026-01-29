#!/usr/bin/env python3
"""
Multi-camera video recorder GUI.

Records synchronized video to:
- port_X.mp4 for each camera
- frame_time_history.csv with synchronization data

Compatible with caliscope post-processing pipeline.
"""

import sys
import csv
import time
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QGroupBox, QTextEdit,
    QDoubleSpinBox, QFileDialog, QProgressBar
)
from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QFont

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("WARNING: OpenCV not available, video recording disabled")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from calimerge.camera_binding import (
    init, shutdown, enumerate_cameras,
    open_camera, close_camera, capture_synced, CameraInfo
)


class VideoRecorderWorker(QThread):
    """Worker thread for recording synchronized video."""
    log_message = Signal(str)
    progress_update = Signal(int, int)  # current_frame, total_frames
    recording_finished = Signal(dict)  # stats

    def __init__(self, cameras: list, output_path: Path, duration: float, fps: int):
        super().__init__()
        self.cameras = cameras
        self.output_path = output_path
        self.duration = duration
        self.fps = fps
        self.running = True

        # Recording state
        self.writers = {}
        self.frame_counts = {}
        self.frame_times = []

    def run(self):
        target_frames = int(self.duration * self.fps)
        frame_interval = 1.0 / self.fps

        # Initialize video writers lazily (after first frame)
        self.writers = {}
        self.frame_counts = {i: 0 for i in range(len(self.cameras))}
        self.frame_times = []

        start_time = time.perf_counter()
        sync_index = 0

        self.log_message.emit(f"Recording {self.duration}s at {self.fps}fps...")

        for frame_num in range(target_frames):
            if not self.running:
                break

            try:
                # Capture synchronized frames
                frameset = capture_synced(self.cameras)
                current_time = time.perf_counter()
                frame_time = current_time - start_time

                for cam_idx, frame in frameset.frames.items():
                    if frame is None:
                        continue

                    # Initialize writer on first frame
                    if cam_idx not in self.writers:
                        video_path = self.output_path / f"port_{cam_idx}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.writers[cam_idx] = cv2.VideoWriter(
                            str(video_path), fourcc, self.fps,
                            (frame.width, frame.height)
                        )
                        self.log_message.emit(f"  Camera {cam_idx}: {frame.width}x{frame.height}")

                    # Write frame
                    self.writers[cam_idx].write(frame.pixels)
                    self.frame_counts[cam_idx] += 1

                    # Record frame time
                    self.frame_times.append({
                        'sync_index': sync_index,
                        'port': cam_idx,
                        'frame_index': self.frame_counts[cam_idx] - 1,
                        'frame_time': frame_time
                    })

                sync_index += 1
                self.progress_update.emit(frame_num + 1, target_frames)

                # Progress logging every second
                if (frame_num + 1) % self.fps == 0:
                    elapsed = time.perf_counter() - start_time
                    actual_fps = (frame_num + 1) / elapsed
                    self.log_message.emit(
                        f"  {frame_num + 1}/{target_frames} frames "
                        f"({elapsed:.1f}s, {actual_fps:.1f} fps)"
                    )

                # Pace to target FPS
                target_time = start_time + (frame_num + 1) * frame_interval
                sleep_time = target_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                self.log_message.emit(f"Frame {frame_num}: ERROR - {e}")

        # Close writers
        for writer in self.writers.values():
            writer.release()

        # Save frame_time_history.csv
        self._save_frame_times()

        # Save camera_mapping.csv
        self._save_camera_mapping()

        # Return stats
        stats = {
            'sync_count': sync_index,
            'duration': time.perf_counter() - start_time,
            'cameras': {
                cam_idx: {
                    'frame_count': count,
                    'video_file': f"port_{cam_idx}.mp4"
                }
                for cam_idx, count in self.frame_counts.items()
            }
        }
        self.recording_finished.emit(stats)

    def _save_frame_times(self):
        csv_path = self.output_path / "frame_time_history.csv"
        with open(csv_path, 'w', newline='') as f:
            # Write serial number mapping as comment header
            serial_mapping = ','.join(
                f"{i}={cam.serial_number}" for i, cam in enumerate(self.cameras)
            )
            f.write(f"# cameras: {serial_mapping}\n")

            writer = csv.writer(f)
            writer.writerow(['sync_index', 'port', 'frame_index', 'frame_time'])
            for entry in self.frame_times:
                writer.writerow([
                    entry['sync_index'],
                    entry['port'],
                    entry['frame_index'],
                    entry['frame_time']
                ])
        self.log_message.emit(f"Saved: frame_time_history.csv ({len(self.frame_times)} entries)")

    def _save_camera_mapping(self):
        csv_path = self.output_path / "camera_mapping.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['port', 'serial_number', 'display_name'])
            for i, cam in enumerate(self.cameras):
                writer.writerow([i, cam.serial_number, cam.display_name])
        self.log_message.emit(f"Saved: camera_mapping.csv")

    def stop(self):
        self.running = False


class RecorderWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calimerge Video Recorder")

        self.cameras = []
        self.opened_cameras = []
        self.worker = None
        self.output_path = None

        self.init_ui()
        self.init_cameras()

    def init_ui(self):
        layout = QVBoxLayout()

        # Status group
        status_group = QGroupBox("Camera Status")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(QFont("Courier", 12))
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Recording settings group
        settings_group = QGroupBox("Recording Settings")
        settings_layout = QHBoxLayout()

        settings_layout.addWidget(QLabel("Duration (sec):"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 3600.0)
        self.duration_spin.setValue(10.0)
        self.duration_spin.setSingleStep(1.0)
        settings_layout.addWidget(self.duration_spin)

        settings_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        settings_layout.addWidget(self.fps_spin)

        settings_layout.addStretch()

        self.browse_button = QPushButton("Output Folder...")
        self.browse_button.clicked.connect(self.browse_output)
        settings_layout.addWidget(self.browse_button)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Output path display
        self.output_label = QLabel("Output: ./recordings/<timestamp>/")
        self.output_label.setFont(QFont("Courier", 10))
        layout.addWidget(self.output_label)

        # Controls group
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Recording")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_recording)
        self.start_button.setMinimumWidth(150)
        self.start_button.setMinimumHeight(50)
        self.start_button.setStyleSheet("font-size: 14px; font-weight: bold;")
        controls_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setMinimumHeight(50)
        controls_layout.addWidget(self.stop_button)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Log group
        log_group = QGroupBox("Recording Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 10))
        self.log_text.setMinimumHeight(200)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Results group
        results_group = QGroupBox("Recording Results")
        results_layout = QVBoxLayout()
        self.results_label = QLabel("No recording yet")
        self.results_label.setFont(QFont("Courier", 11))
        results_layout.addWidget(self.results_label)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        self.setLayout(layout)
        self.setMinimumSize(700, 650)

        # Default output path
        self.base_output_path = Path("recordings")

    def init_cameras(self):
        if not HAS_OPENCV:
            self.log("ERROR: OpenCV required for video recording")
            self.status_label.setText("OpenCV not available")
            return

        self.log("Initializing camera subsystem...")
        try:
            init()
            self.cameras = enumerate_cameras()
            self.log(f"Found {len(self.cameras)} camera(s)")

            if len(self.cameras) < 1:
                self.log("ERROR: No cameras found")
                self.status_label.setText("No cameras found")
                return

            # Print camera info
            status_lines = []
            for i, cam in enumerate(self.cameras):
                self.log(f"  Camera {i}: {cam.display_name} ({cam.serial_number})")
                status_lines.append(f"Port {i}: {cam.display_name}")

            # Open cameras
            self.log("Opening cameras...")
            for i, cam in enumerate(self.cameras):
                try:
                    open_camera(cam)
                    self.opened_cameras.append(cam)
                    self.log(f"  Camera {i}: OK")
                except Exception as e:
                    self.log(f"  Camera {i}: FAILED ({e})")

            if self.opened_cameras:
                self.status_label.setText(
                    "\n".join(status_lines) +
                    f"\n\n{len(self.opened_cameras)} camera(s) ready"
                )
                self.start_button.setEnabled(True)
                self.log("\nReady to record!")
            else:
                self.status_label.setText("Failed to open any cameras")
                self.log("ERROR: Failed to open any cameras")

        except Exception as e:
            self.log(f"ERROR: {e}")
            self.status_label.setText(f"Error: {e}")

    def log(self, message: str):
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.base_output_path = Path(folder)
            self.output_label.setText(f"Output: {folder}/<timestamp>/")

    def start_recording(self):
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = self.base_output_path / timestamp
        self.output_path.mkdir(parents=True, exist_ok=True)

        duration = self.duration_spin.value()
        fps = self.fps_spin.value()

        self.log(f"\n{'='*50}")
        self.log(f"Starting recording")
        self.log(f"  Duration: {duration}s")
        self.log(f"  FPS: {fps}")
        self.log(f"  Output: {self.output_path}")
        self.log(f"{'='*50}\n")

        # UI state
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.duration_spin.setEnabled(False)
        self.fps_spin.setEnabled(False)
        self.browse_button.setEnabled(False)

        # Progress bar
        total_frames = int(duration * fps)
        self.progress_bar.setMaximum(total_frames)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # Start worker
        self.worker = VideoRecorderWorker(
            self.opened_cameras, self.output_path, duration, fps
        )
        self.worker.log_message.connect(self.log)
        self.worker.progress_update.connect(self.on_progress)
        self.worker.recording_finished.connect(self.on_finished)
        self.worker.start()

    def stop_recording(self):
        if self.worker:
            self.log("\nStopping recording...")
            self.worker.stop()

    def on_progress(self, current: int, total: int):
        self.progress_bar.setValue(current)

    def on_finished(self, stats: dict):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.duration_spin.setEnabled(True)
        self.fps_spin.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.log(f"\n{'='*50}")
        self.log("Recording complete!")
        self.log(f"  Sync frames: {stats['sync_count']}")
        self.log(f"  Duration: {stats['duration']:.2f}s")
        for cam_idx, cam_stats in stats['cameras'].items():
            self.log(f"  Camera {cam_idx}: {cam_stats['frame_count']} frames -> {cam_stats['video_file']}")
        self.log(f"\nOutput: {self.output_path}")
        self.log(f"{'='*50}")

        # Update results
        result_lines = [
            f"Sync frames: {stats['sync_count']}",
            f"Duration: {stats['duration']:.2f}s",
            f"Output: {self.output_path}",
            "",
            "Files:"
        ]
        for cam_idx, cam_stats in stats['cameras'].items():
            result_lines.append(f"  {cam_stats['video_file']} ({cam_stats['frame_count']} frames)")
        result_lines.append("  frame_time_history.csv")

        self.results_label.setText("\n".join(result_lines))
        self.worker = None

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            self.worker.wait()

        self.log("Closing cameras...")
        for cam in self.opened_cameras:
            close_camera(cam)

        shutdown()
        event.accept()


def main():
    app = QApplication(sys.argv)
    widget = RecorderWidget()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
