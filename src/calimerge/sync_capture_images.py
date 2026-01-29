#!/usr/bin/env python3
"""
Synchronized capture GUI for visual sync verification.

Usage:
    1. Run clock_widget.py in one terminal: python3 clock_widget.py
    2. Point your cameras at the clock display
    3. Run this script: python3 sync_capture_images.py
    4. Click "Start Recording" to capture frames
    5. Compare the clock times visible in each camera's captured images
"""

import sys
import os
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QGroupBox, QTextEdit
)
from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QFont

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from calimerge.camera_binding import (
    init, shutdown, enumerate_cameras,
    open_camera, close_camera, capture_synced, CameraInfo
)

OUTPUT_DIR = Path("sync_test_images")


def save_frame(pixels: np.ndarray, filepath: Path):
    """Save frame as PNG image."""
    if HAS_OPENCV:
        cv2.imwrite(str(filepath), pixels)
    else:
        np.save(str(filepath.with_suffix('.npy')), pixels)


class CaptureWorker(QThread):
    """Worker thread for capturing synchronized frames."""
    log_message = Signal(str)
    capture_complete = Signal(int, float)  # capture_idx, spread_ms
    finished_all = Signal(list)  # stats list

    def __init__(self, cameras: list, num_captures: int, output_path: Path):
        super().__init__()
        self.cameras = cameras
        self.num_captures = num_captures
        self.output_path = output_path
        self.running = True

    def run(self):
        stats = []

        for capture_idx in range(self.num_captures):
            if not self.running:
                break

            try:
                frameset = capture_synced(self.cameras)

                # Calculate spread
                corrected_times = []
                for cam_idx, frame in frameset.frames.items():
                    if frame is not None:
                        corrected_times.append(frame.corrected_ns)

                if len(corrected_times) >= 2:
                    spread_ns = max(corrected_times) - min(corrected_times)
                    spread_ms = spread_ns / 1e6
                else:
                    spread_ms = 0

                stats.append(spread_ms)

                msg_parts = [f"Sync {capture_idx:3d}: "]

                # Save each frame
                for cam_idx, frame in frameset.frames.items():
                    if frame is not None:
                        corrected_ms = frame.corrected_ns / 1e6
                        filename = f"sync_{capture_idx:03d}_cam{cam_idx}_corrected_{corrected_ms:.2f}ms.png"
                        filepath = self.output_path / filename
                        save_frame(frame.pixels, filepath)
                        msg_parts.append(f"[cam{cam_idx}: {corrected_ms:.2f}ms] ")
                    else:
                        msg_parts.append(f"[cam{cam_idx}: DROPPED] ")

                msg_parts.append(f"(spread: {spread_ms:.2f}ms)")
                self.log_message.emit("".join(msg_parts))
                self.capture_complete.emit(capture_idx, spread_ms)

                # Pace at ~2 fps for visual inspection
                self.msleep(500)

            except Exception as e:
                self.log_message.emit(f"Sync {capture_idx:3d}: FAILED ({e})")

        self.finished_all.emit(stats)

    def stop(self):
        self.running = False


class SyncCaptureWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calimerge Sync Capture")

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

        # Controls group
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Frames to capture:"))
        self.num_frames_spin = QSpinBox()
        self.num_frames_spin.setRange(1, 1000)
        self.num_frames_spin.setValue(10)
        controls_layout.addWidget(self.num_frames_spin)

        controls_layout.addStretch()

        self.start_button = QPushButton("Start Recording")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_recording)
        self.start_button.setMinimumWidth(150)
        self.start_button.setMinimumHeight(40)
        controls_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_recording)
        controls_layout.addWidget(self.stop_button)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Log group
        log_group = QGroupBox("Capture Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 10))
        self.log_text.setMinimumHeight(300)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Stats group
        stats_group = QGroupBox("Statistics")
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("No captures yet")
        self.stats_label.setFont(QFont("Courier", 11))
        stats_layout.addWidget(self.stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        self.setLayout(layout)
        self.setMinimumSize(700, 550)

    def init_cameras(self):
        self.log("Initializing camera subsystem...")
        try:
            init()
            self.cameras = enumerate_cameras()
            self.log(f"Found {len(self.cameras)} camera(s)")

            if len(self.cameras) < 2:
                self.log("WARNING: Need at least 2 cameras for sync test")
                self.status_label.setText("Need at least 2 cameras")
                return

            # Print camera info
            status_lines = []
            for i, cam in enumerate(self.cameras):
                self.log(f"  Camera {i}: {cam.display_name} ({cam.serial_number})")
                status_lines.append(f"Cam {i}: {cam.display_name}")

            # Open cameras
            self.log("Opening cameras...")
            for i, cam in enumerate(self.cameras):
                try:
                    open_camera(cam)
                    self.opened_cameras.append(cam)
                    self.log(f"  Camera {i}: OK")
                except Exception as e:
                    self.log(f"  Camera {i}: FAILED ({e})")

            if len(self.opened_cameras) >= 2:
                self.status_label.setText("\n".join(status_lines) + "\n\nReady to record")
                self.start_button.setEnabled(True)
                self.log("\nReady! Point cameras at clock widget and click Start Recording.")
            else:
                self.status_label.setText("Failed to open enough cameras")
                self.log("ERROR: Failed to open at least 2 cameras")

        except Exception as e:
            self.log(f"ERROR: {e}")
            self.status_label.setText(f"Error: {e}")

    def log(self, message: str):
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def start_recording(self):
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = OUTPUT_DIR / timestamp
        self.output_path.mkdir(parents=True, exist_ok=True)

        num_captures = self.num_frames_spin.value()
        self.log(f"\n{'='*50}")
        self.log(f"Starting capture of {num_captures} frame sets")
        self.log(f"Output: {self.output_path.absolute()}")
        self.log(f"{'='*50}\n")

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.num_frames_spin.setEnabled(False)

        # Start worker thread
        self.worker = CaptureWorker(self.opened_cameras, num_captures, self.output_path)
        self.worker.log_message.connect(self.log)
        self.worker.capture_complete.connect(self.on_capture_complete)
        self.worker.finished_all.connect(self.on_finished)
        self.worker.start()

    def stop_recording(self):
        if self.worker:
            self.log("\nStopping...")
            self.worker.stop()

    def on_capture_complete(self, idx: int, spread_ms: float):
        # Update stats incrementally
        pass

    def on_finished(self, stats: list):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.num_frames_spin.setEnabled(True)

        if stats:
            self.log(f"\n{'='*50}")
            self.log(f"Capture Statistics ({len(stats)} frame sets)")
            self.log(f"{'='*50}")
            self.log(f"  Mean spread: {np.mean(stats):.2f} ms")
            self.log(f"  Min spread:  {np.min(stats):.2f} ms")
            self.log(f"  Max spread:  {np.max(stats):.2f} ms")
            self.log(f"\nImages saved to: {self.output_path.absolute()}")

            self.stats_label.setText(
                f"Mean: {np.mean(stats):.2f}ms  |  "
                f"Min: {np.min(stats):.2f}ms  |  "
                f"Max: {np.max(stats):.2f}ms  |  "
                f"Frames: {len(stats)}"
            )
        else:
            self.log("No frames captured")

        self.worker = None

    def closeEvent(self, event):
        # Stop worker if running
        if self.worker:
            self.worker.stop()
            self.worker.wait()

        # Close cameras
        self.log("Closing cameras...")
        for cam in self.opened_cameras:
            close_camera(cam)

        shutdown()
        event.accept()


def main():
    app = QApplication(sys.argv)
    widget = SyncCaptureWidget()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
