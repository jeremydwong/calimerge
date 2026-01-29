#!/usr/bin/env python3
"""
Video recorder for synchronized multi-camera capture.

Saves:
- port_X.mp4 for each camera
- frame_time_history.csv with synchronization data

Compatible with caliscope post-processing pipeline.
"""

import sys
import csv
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("ERROR: OpenCV required for video recording")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from calimerge.camera_binding import (
    init, shutdown, enumerate_cameras,
    open_camera, close_camera, capture_synced, CameraInfo, SyncedFrameSet
)


@dataclass
class FrameTimeEntry:
    """Entry for frame_time_history.csv"""
    sync_index: int
    port: int
    frame_index: int
    frame_time: float


@dataclass
class CameraRecorder:
    """Manages video recording for a single camera."""
    port: int
    output_path: Path
    fps: int = 30
    codec: str = "mp4v"

    writer: Optional[cv2.VideoWriter] = field(default=None, init=False)
    frame_count: int = field(default=0, init=False)
    width: int = field(default=0, init=False)
    height: int = field(default=0, init=False)

    def start(self, width: int, height: int):
        """Initialize video writer with frame dimensions."""
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV required for video recording")

        self.width = width
        self.height = height
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        video_path = self.output_path / f"port_{self.port}.mp4"
        self.writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for port {self.port}")

    def write_frame(self, pixels: np.ndarray):
        """Write a frame to the video file."""
        if self.writer is None:
            self.start(pixels.shape[1], pixels.shape[0])
        self.writer.write(pixels)
        self.frame_count += 1

    def stop(self):
        """Close the video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class MultiCameraRecorder:
    """
    Records synchronized video from multiple cameras.

    Output format compatible with caliscope:
    - port_0.mp4, port_1.mp4, ... (one video per camera)
    - frame_time_history.csv (sync_index, port, frame_index, frame_time)
    - camera_mapping.csv (port, serial_number, display_name)
    """

    def __init__(self, cameras: list[CameraInfo], output_dir: Path, fps: int = 30):
        self.cameras = cameras
        self.output_dir = Path(output_dir)
        self.fps = fps

        self.recorders: dict[int, CameraRecorder] = {}
        self.camera_serials: dict[int, str] = {}  # port -> serial_number
        self.camera_names: dict[int, str] = {}    # port -> display_name
        self.frame_times: list[FrameTimeEntry] = []
        self.sync_index = 0
        self.start_time: Optional[float] = None
        self.is_recording = False

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize recorders for each camera
        for i, cam in enumerate(cameras):
            self.recorders[i] = CameraRecorder(
                port=i,
                output_path=self.output_dir,
                fps=fps
            )
            self.camera_serials[i] = cam.serial_number
            self.camera_names[i] = cam.display_name

    def start(self):
        """Start recording."""
        self.is_recording = True
        self.start_time = time.perf_counter()
        self.sync_index = 0
        self.frame_times = []

    def record_frameset(self, frameset: SyncedFrameSet) -> int:
        """
        Record a synchronized frame set.

        Returns the sync_index for this frame set.
        """
        if not self.is_recording:
            raise RuntimeError("Recording not started")

        current_time = time.perf_counter()
        frame_time = current_time - self.start_time

        for cam_idx, frame in frameset.frames.items():
            if frame is not None and cam_idx in self.recorders:
                recorder = self.recorders[cam_idx]

                # Write video frame
                recorder.write_frame(frame.pixels)

                # Record frame time entry
                entry = FrameTimeEntry(
                    sync_index=self.sync_index,
                    port=cam_idx,
                    frame_index=recorder.frame_count - 1,
                    frame_time=frame_time
                )
                self.frame_times.append(entry)

        self.sync_index += 1
        return self.sync_index - 1

    def stop(self):
        """Stop recording and save metadata files."""
        self.is_recording = False

        # Close all video writers
        for recorder in self.recorders.values():
            recorder.stop()

        # Save frame_time_history.csv
        self._save_frame_time_history()

        # Save camera_mapping.csv
        self._save_camera_mapping()

    def _save_frame_time_history(self):
        """Save frame_time_history.csv file with serial numbers in header comment."""
        csv_path = self.output_dir / "frame_time_history.csv"

        with open(csv_path, 'w', newline='') as f:
            # Write serial number mapping as comment header
            serial_mapping = ','.join(
                f"{port}={serial}" for port, serial in sorted(self.camera_serials.items())
            )
            f.write(f"# cameras: {serial_mapping}\n")

            writer = csv.writer(f)
            writer.writerow(['sync_index', 'port', 'frame_index', 'frame_time'])

            for entry in self.frame_times:
                writer.writerow([
                    entry.sync_index,
                    entry.port,
                    entry.frame_index,
                    entry.frame_time
                ])

    def _save_camera_mapping(self):
        """Save camera_mapping.csv with port-to-serial mapping."""
        csv_path = self.output_dir / "camera_mapping.csv"

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['port', 'serial_number', 'display_name'])

            for port in sorted(self.camera_serials.keys()):
                writer.writerow([
                    port,
                    self.camera_serials[port],
                    self.camera_names[port]
                ])

    def get_stats(self) -> dict:
        """Get recording statistics."""
        stats = {
            'sync_count': self.sync_index,
            'cameras': {}
        }
        for cam_idx, recorder in self.recorders.items():
            stats['cameras'][cam_idx] = {
                'frame_count': recorder.frame_count,
                'video_file': f"port_{cam_idx}.mp4"
            }
        return stats


def record_session(
    output_dir: Path,
    duration_seconds: float = 10.0,
    fps: int = 30
) -> dict:
    """
    Record a synchronized video session.

    Args:
        output_dir: Directory to save output files
        duration_seconds: Recording duration in seconds
        fps: Target frame rate

    Returns:
        Recording statistics
    """
    if not HAS_OPENCV:
        raise RuntimeError("OpenCV required for video recording")

    print(f"Initializing cameras...")
    init()

    cameras = enumerate_cameras()
    print(f"Found {len(cameras)} camera(s)")

    if len(cameras) < 1:
        shutdown()
        raise RuntimeError("No cameras found")

    # Open cameras
    opened_cameras = []
    for i, cam in enumerate(cameras):
        try:
            open_camera(cam)
            opened_cameras.append(cam)
            print(f"  Camera {i}: {cam.display_name} - OK")
        except Exception as e:
            print(f"  Camera {i}: {cam.display_name} - FAILED ({e})")

    if not opened_cameras:
        shutdown()
        raise RuntimeError("Failed to open any cameras")

    # Warm up
    print("Warming up (1 second)...")
    time.sleep(1)

    # Create recorder
    recorder = MultiCameraRecorder(opened_cameras, output_dir, fps)

    # Calculate frames to capture
    target_frames = int(duration_seconds * fps)
    frame_interval = 1.0 / fps

    print(f"\nRecording {duration_seconds}s at {fps}fps ({target_frames} frames)...")
    print(f"Output: {output_dir}")

    recorder.start()
    last_frame_time = time.perf_counter()

    try:
        for frame_num in range(target_frames):
            # Capture synchronized frames
            frameset = capture_synced(opened_cameras)
            sync_idx = recorder.record_frameset(frameset)

            # Progress update
            if (frame_num + 1) % 30 == 0:
                elapsed = time.perf_counter() - recorder.start_time
                actual_fps = (frame_num + 1) / elapsed
                print(f"  Frame {frame_num + 1}/{target_frames} "
                      f"(sync_index={sync_idx}, actual_fps={actual_fps:.1f})")

            # Pace to target FPS
            target_time = recorder.start_time + (frame_num + 1) * frame_interval
            sleep_time = target_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nRecording interrupted by user")

    finally:
        recorder.stop()

    # Close cameras
    print("\nClosing cameras...")
    for cam in opened_cameras:
        close_camera(cam)
    shutdown()

    stats = recorder.get_stats()
    print(f"\nRecording complete!")
    print(f"  Sync frames: {stats['sync_count']}")
    for cam_idx, cam_stats in stats['cameras'].items():
        print(f"  Camera {cam_idx}: {cam_stats['frame_count']} frames -> {cam_stats['video_file']}")
    print(f"  Frame times: frame_time_history.csv")

    return stats


def main():
    """CLI entry point for video recording."""
    import argparse

    parser = argparse.ArgumentParser(description="Record synchronized multi-camera video")
    parser.add_argument("-o", "--output", type=str, default="recording",
                        help="Output directory (default: recording)")
    parser.add_argument("-d", "--duration", type=float, default=10.0,
                        help="Recording duration in seconds (default: 10)")
    parser.add_argument("-f", "--fps", type=int, default=30,
                        help="Target frame rate (default: 30)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / timestamp

    try:
        record_session(output_dir, args.duration, args.fps)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
