"""
QThread workers for async operations.

Each worker calls pure functions and emits results via signals.
Workers do NOT modify state directly - they emit results to StateManager.
"""

from __future__ import annotations

import time
import csv
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal

if TYPE_CHECKING:
    import numpy as np
    from ..camera_binding import CameraInfo
    from ..types import CameraIntrinsics, CalibratedCamera, CharucoConfig


class CameraEnumerateWorker(QThread):
    """Enumerate available cameras."""

    cameras_found = Signal(list)  # list[CameraInfo]
    error = Signal(str)

    def run(self):
        try:
            from ..camera_binding import init, enumerate_cameras

            init()
            cameras = enumerate_cameras()
            self.cameras_found.emit(cameras)
        except Exception as e:
            self.error.emit(str(e))


class CameraPreviewWorker(QThread):
    """Capture frames from cameras for live preview."""

    frame_captured = Signal(int, object)  # port, np.ndarray
    error = Signal(str)

    def __init__(self, cameras: list, fps: int = 30):
        super().__init__()
        self.cameras = cameras
        self.fps = fps
        self.running = True

    def run(self):
        from ..camera_binding import capture_synced

        frame_interval = 1.0 / self.fps

        while self.running:
            try:
                start = time.perf_counter()

                frameset = capture_synced(self.cameras)
                for port, frame in frameset.frames.items():
                    if frame is not None:
                        self.frame_captured.emit(port, frame.pixels)

                # Pace to target FPS
                elapsed = time.perf_counter() - start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                self.error.emit(str(e))
                break

    def stop(self):
        self.running = False


class RecordingWorker(QThread):
    """Record synchronized video to files."""

    log_message = Signal(str)
    progress_update = Signal(int, int)  # current, total
    recording_finished = Signal(dict)  # stats
    error = Signal(str)

    def __init__(
        self,
        cameras: list,
        output_path: Path,
        duration: float,
        fps: int,
    ):
        super().__init__()
        self.cameras = cameras
        self.output_path = output_path
        self.duration = duration
        self.fps = fps
        self.running = True

    def run(self):
        try:
            import cv2
            from ..camera_binding import capture_synced

            target_frames = int(self.duration * self.fps)
            frame_interval = 1.0 / self.fps

            writers = {}
            frame_counts = {i: 0 for i in range(len(self.cameras))}
            frame_times = []

            start_time = time.perf_counter()
            sync_index = 0

            self.log_message.emit(f"Recording {self.duration}s at {self.fps}fps...")

            for frame_num in range(target_frames):
                if not self.running:
                    break

                frameset = capture_synced(self.cameras)
                current_time = time.perf_counter()
                frame_time = current_time - start_time

                for cam_idx, frame in frameset.frames.items():
                    if frame is None:
                        continue

                    # Initialize writer on first frame
                    if cam_idx not in writers:
                        video_path = self.output_path / f"port_{cam_idx}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writers[cam_idx] = cv2.VideoWriter(
                            str(video_path),
                            fourcc,
                            self.fps,
                            (frame.width, frame.height),
                        )
                        self.log_message.emit(
                            f"  Camera {cam_idx}: {frame.width}x{frame.height}"
                        )

                    writers[cam_idx].write(frame.pixels)
                    frame_counts[cam_idx] += 1

                    frame_times.append(
                        {
                            "sync_index": sync_index,
                            "port": cam_idx,
                            "frame_index": frame_counts[cam_idx] - 1,
                            "frame_time": frame_time,
                        }
                    )

                sync_index += 1
                self.progress_update.emit(frame_num + 1, target_frames)

                # Pace to target FPS
                target_time = start_time + (frame_num + 1) * frame_interval
                sleep_time = target_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Close writers
            for writer in writers.values():
                writer.release()

            # Save frame_time_history.csv
            self._save_frame_times(frame_times)

            # Save camera_mapping.csv
            self._save_camera_mapping()

            stats = {
                "sync_count": sync_index,
                "duration": time.perf_counter() - start_time,
                "cameras": {
                    idx: {"frame_count": count, "video_file": f"port_{idx}.mp4"}
                    for idx, count in frame_counts.items()
                },
            }
            self.recording_finished.emit(stats)

        except Exception as e:
            self.error.emit(str(e))

    def _save_frame_times(self, frame_times: list):
        csv_path = self.output_path / "frame_time_history.csv"
        with open(csv_path, "w", newline="") as f:
            serial_mapping = ",".join(
                f"{i}={cam.serial_number}" for i, cam in enumerate(self.cameras)
            )
            f.write(f"# cameras: {serial_mapping}\n")

            writer = csv.writer(f)
            writer.writerow(["sync_index", "port", "frame_index", "frame_time"])
            for entry in frame_times:
                writer.writerow(
                    [
                        entry["sync_index"],
                        entry["port"],
                        entry["frame_index"],
                        entry["frame_time"],
                    ]
                )

    def _save_camera_mapping(self):
        csv_path = self.output_path / "camera_mapping.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["port", "serial_number", "display_name"])
            for i, cam in enumerate(self.cameras):
                writer.writerow([i, cam.serial_number, cam.display_name])

    def stop(self):
        self.running = False


class IntrinsicCalibrationWorker(QThread):
    """Calibrate intrinsics from video."""

    log_message = Signal(str)
    progress_update = Signal(int, int)  # current, total
    calibration_finished = Signal(object)  # CameraIntrinsics
    error = Signal(str)

    def __init__(
        self,
        video_path: Path,
        serial_number: str,
        charuco_config: "CharucoConfig",
        sample_interval: int = 10,
    ):
        super().__init__()
        self.video_path = video_path
        self.serial_number = serial_number
        self.charuco_config = charuco_config
        self.sample_interval = sample_interval
        self.running = True

    def run(self):
        try:
            import cv2
            from ..calibration.intrinsic import detect_charuco_points, calibrate_intrinsics
            from ..calibration.charuco import create_charuco_board
            from ..types import PointPacket

            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                self.error.emit(f"Cannot open video: {self.video_path}")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution = (width, height)

            self.log_message.emit(
                f"Processing {self.video_path.name}: {total_frames} frames at {width}x{height}"
            )

            board = create_charuco_board(self.charuco_config)
            point_packets = []
            frame_idx = 0

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.sample_interval == 0:
                    packet = detect_charuco_points(frame, self.charuco_config, board)
                    if packet.point_id is not None and len(packet.point_id) >= 4:
                        point_packets.append(packet)
                        self.log_message.emit(
                            f"  Frame {frame_idx}: {len(packet.point_id)} corners"
                        )

                frame_idx += 1
                self.progress_update.emit(frame_idx, total_frames)

            cap.release()

            if len(point_packets) < 10:
                self.error.emit(
                    f"Only {len(point_packets)} valid frames, need at least 10"
                )
                return

            self.log_message.emit(f"Calibrating from {len(point_packets)} frames...")

            intrinsics = calibrate_intrinsics(
                point_packets, resolution, self.serial_number
            )

            self.log_message.emit(f"Calibration complete, error: {intrinsics.error:.4f}")
            self.calibration_finished.emit(intrinsics)

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.running = False


class ExtrinsicCalibrationWorker(QThread):
    """Calibrate extrinsics via bundle adjustment."""

    log_message = Signal(str)
    progress_update = Signal(float)  # 0.0 to 1.0
    calibration_finished = Signal(dict, float)  # cameras, error
    error = Signal(str)

    def __init__(
        self,
        video_paths: dict[int, Path],
        intrinsics: dict[int, "CameraIntrinsics"],
        charuco_config: "CharucoConfig",
    ):
        super().__init__()
        self.video_paths = video_paths
        self.intrinsics = intrinsics
        self.charuco_config = charuco_config
        self.running = True

    def run(self):
        try:
            # This is a placeholder - full implementation would:
            # 1. Detect charuco in all videos synced by frame
            # 2. Build SyncedPoints list
            # 3. Run stereo calibration for pairs
            # 4. Run bundle adjustment

            self.log_message.emit("Extrinsic calibration not yet implemented")
            self.error.emit("Extrinsic calibration not yet implemented")

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.running = False


class ProcessingWorker(QThread):
    """Run tracking and triangulation pipeline."""

    log_message = Signal(str)
    progress_update = Signal(str, float)  # step_name, progress
    processing_finished = Signal(Path)  # output file
    error = Signal(str)

    def __init__(
        self,
        video_paths: dict[int, Path],
        cameras: dict[int, "CalibratedCamera"],
        output_path: Path,
        tracker_backend: str = "charuco",
    ):
        super().__init__()
        self.video_paths = video_paths
        self.cameras = cameras
        self.output_path = output_path
        self.tracker_backend = tracker_backend
        self.running = True

    def run(self):
        try:
            # Placeholder - full implementation would:
            # 1. Run 2D tracking on all videos
            # 2. Sync points across cameras
            # 3. Triangulate to 3D
            # 4. Export results

            self.log_message.emit("Processing not yet implemented")
            self.error.emit("Processing not yet implemented")

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.running = False
