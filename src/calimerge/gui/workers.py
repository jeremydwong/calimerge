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

    cameras_found = Signal(object)  # list[CameraInfo]
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

    def __init__(self, cameras: list, ports: list[int], fps: int = 30):
        super().__init__()
        self.cameras = cameras
        self.ports = ports
        self.fps = fps
        self.running = True
        self._paused = False

    def run(self):
        from ..camera_binding import capture_synced

        frame_interval = 1.0 / self.fps
        consecutive_errors = 0
        max_retries = 5

        while self.running:
            if self._paused:
                time.sleep(0.05)
                continue

            try:
                start = time.perf_counter()

                frameset = capture_synced(self.cameras)
                consecutive_errors = 0  # reset on success

                for i, (_, frame) in enumerate(frameset.frames.items()):
                    if frame is not None:
                        self.frame_captured.emit(self.ports[i], frame.pixels)

                # Pace to target FPS
                elapsed = time.perf_counter() - start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_retries:
                    self.error.emit(f"Preview failed after {max_retries} retries: {e}")
                    break
                # Brief backoff before retry
                time.sleep(0.1)

    def stop(self):
        self.running = False

    def pause(self):
        """Pause frame capture (cameras stay open)."""
        self._paused = True

    def resume(self):
        """Resume frame capture."""
        self._paused = False


class RecordingWorker(QThread):
    """Record synchronized video to files with hardware encoding."""

    log_message = Signal(str)
    progress_update = Signal(int, int)  # current, total
    frame_captured = Signal(int, object)  # port, np.ndarray (for preview/FPS)
    recording_finished = Signal(object)  # stats dict
    error = Signal(str)

    def __init__(
        self,
        cameras: list,
        ports: list[int],
        output_path: Path,
        duration: float,
        fps: int,
        codec: str = "h264",
    ):
        super().__init__()
        self.cameras = cameras
        self.ports = ports
        self.output_path = output_path
        self.duration = duration
        self.fps = fps
        self.codec = codec
        self.running = True

    def run(self):
        try:
            from ..camera_binding import capture_synced
            from .video_utils import create_video_writer, write_frame, release_writer, detect_encoders

            target_frames = int(self.duration * self.fps)
            frame_interval = 1.0 / self.fps

            writers = {}
            frame_counts = {port: 0 for port in self.ports}
            frame_times = []

            # Log encoder info
            info = detect_encoders()
            if info.ffmpeg_path and info.has_h264_hw:
                self.log_message.emit(f"Using {info.h264_hw_encoder} hardware encoder")
            elif info.ffmpeg_path and info.has_libx264:
                self.log_message.emit(f"Using libx264 software encoder")
            elif info.ffmpeg_path:
                self.log_message.emit(f"Using ffmpeg mpeg4 encoder")
            else:
                self.log_message.emit("Using OpenCV mp4v fallback encoder")

            start_time = time.perf_counter()
            sync_index = 0

            self.log_message.emit(f"Recording {self.duration}s at {self.fps}fps...")

            for frame_num in range(target_frames):
                if not self.running:
                    break

                frameset = capture_synced(self.cameras)
                current_time = time.perf_counter()
                frame_time = current_time - start_time

                for i, (_, frame) in enumerate(frameset.frames.items()):
                    if frame is None:
                        continue

                    port = self.ports[i]
                    serial = self.cameras[i].serial_number
                    sanitized_serial = serial.replace("&", "-")

                    # Initialize writer on first frame
                    if port not in writers:
                        video_path = self.output_path / f"port_{port}_{sanitized_serial}.mp4"
                        writers[port] = create_video_writer(
                            video_path,
                            frame.width,
                            frame.height,
                            self.fps,
                            codec=self.codec,
                            metadata={"comment": f"serial:{serial}"},
                        )
                        self.log_message.emit(
                            f"  Port {port} [{serial}]: {frame.width}x{frame.height}"
                        )

                    write_frame(writers[port], frame.pixels)
                    frame_counts[port] += 1

                    # Emit for preview/FPS tracking
                    self.frame_captured.emit(port, frame.pixels)

                    frame_times.append(
                        {
                            "sync_index": sync_index,
                            "port": port,
                            "frame_index": frame_counts[port] - 1,
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
                release_writer(writer)

            # Save frame_time_history.csv
            self._save_frame_times(frame_times)

            # Save camera_mapping.csv
            self._save_camera_mapping()

            stats = {
                "sync_count": sync_index,
                "duration": time.perf_counter() - start_time,
                "cameras": {
                    port: {"frame_count": count}
                    for port, count in frame_counts.items()
                },
            }
            self.recording_finished.emit(stats)

        except Exception as e:
            self.error.emit(str(e))

    def _save_frame_times(self, frame_times: list):
        csv_path = self.output_path / "frame_time_history.csv"
        with open(csv_path, "w", newline="") as f:
            serial_mapping = ",".join(
                f"{self.ports[i]}={cam.serial_number}" for i, cam in enumerate(self.cameras)
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
                writer.writerow([self.ports[i], cam.serial_number, cam.display_name])

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
    calibration_finished = Signal(object, float)  # cameras dict, error
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
        frame_time_csv: Path | None = None,
        device_name: str = "auto",
        max_persons: int = 2,
        batch_size: int = 8,
        skip_sync_indices: int = 1,
    ):
        super().__init__()
        self.video_paths = video_paths
        self.cameras = cameras
        self.output_path = output_path
        self.tracker_backend = tracker_backend
        self.frame_time_csv = frame_time_csv
        self.device_name = device_name
        self.max_persons = max_persons
        self.batch_size = batch_size
        self.skip_sync_indices = skip_sync_indices
        self.running = True

    def run(self):
        try:
            from ..tracking.pipeline import run_pose_tracking

            result = run_pose_tracking(
                cameras=self.cameras,
                video_paths=self.video_paths,
                frame_time_csv=self.frame_time_csv,
                output_path=self.output_path,
                device_name=self.device_name,
                skip_sync_indices=self.skip_sync_indices,
                max_persons=self.max_persons,
                batch_size=self.batch_size,
                progress_callback=lambda step, frac: self.progress_update.emit(step, frac),
                log_callback=lambda msg: self.log_message.emit(msg),
            )

            if result is not None:
                self.processing_finished.emit(result)
            else:
                self.error.emit("Processing pipeline returned no results")

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.running = False
