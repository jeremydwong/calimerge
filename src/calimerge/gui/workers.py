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
    detection_frame = Signal(int, int, object, int)  # port, frame_index, frame (BGR), corner_count
    selected_frames = Signal(list)  # list of BGR frames actually used for calibration
    calibration_finished = Signal(object)  # CameraIntrinsics
    error = Signal(str)

    def __init__(
        self,
        video_path: Path,
        serial_number: str,
        charuco_config: "CharucoConfig",
        port: int = 0,
        sample_interval: int = 10,
        max_calibration_frames: int = 40,
    ):
        super().__init__()
        self.video_path = video_path
        self.serial_number = serial_number
        self.charuco_config = charuco_config
        self.port = port
        self.sample_interval = sample_interval
        self.max_calibration_frames = max_calibration_frames
        self.running = True

    def _draw_charuco_overlay(self, frame, packet, board, color=(0, 220, 80)):
        """Draw charuco detection visualization on frame using a single color."""
        import cv2

        vis = frame.copy()
        ids = packet.point_id
        img_loc = packet.img_loc
        n = len(ids)

        if n == 0:
            return vis

        for i in range(n):
            pt = (int(img_loc[i, 0]), int(img_loc[i, 1]))
            cv2.circle(vis, pt, 6, color, 2)
            cv2.circle(vis, pt, 2, color, -1)

        # Draw lines connecting adjacent corners (grid structure)
        cols = self.charuco_config.columns - 1
        for i in range(n):
            cid_i = int(ids[i])
            row_i, col_i = divmod(cid_i, cols)
            for j in range(i + 1, n):
                cid_j = int(ids[j])
                row_j, col_j = divmod(cid_j, cols)
                if (row_i == row_j and abs(col_i - col_j) == 1) or \
                   (col_i == col_j and abs(row_i - row_j) == 1):
                    pt1 = (int(img_loc[i, 0]), int(img_loc[i, 1]))
                    pt2 = (int(img_loc[j, 0]), int(img_loc[j, 1]))
                    cv2.line(vis, pt1, pt2, color, 1, cv2.LINE_AA)

        cv2.putText(
            vis, f"{n} corners", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )

        return vis

    def run(self):
        try:
            import cv2
            from ..calibration.intrinsic import (
                detect_charuco_points,
                calibrate_intrinsics,
                filter_frames_for_calibration,
            )
            from ..calibration.charuco import create_charuco_board

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
            vis_frames = []  # parallel list of rendered overlay frames
            frame_idx = 0

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.sample_interval == 0:
                    packet = detect_charuco_points(frame, self.charuco_config, board)
                    if packet.point_id is not None and len(packet.point_id) >= 4:
                        vis = self._draw_charuco_overlay(frame, packet, board)
                        point_packets.append(packet)
                        vis_frames.append(vis)
                        corner_count = len(packet.point_id)
                        self.log_message.emit(
                            f"  Frame {frame_idx}: {corner_count} corners"
                        )
                        self.detection_frame.emit(self.port, frame_idx, vis, corner_count)

                frame_idx += 1
                self.progress_update.emit(frame_idx, total_frames)

            cap.release()

            if len(point_packets) < 10:
                self.error.emit(
                    f"Only {len(point_packets)} valid frames, need at least 10"
                )
                return

            # Downsample to well-distributed temporal subset
            if len(point_packets) > self.max_calibration_frames:
                self.log_message.emit(
                    f"Downsampling to {self.max_calibration_frames} frames..."
                )
                # Track which packets survive so we can filter vis_frames in parallel
                vis_by_id = {id(p): v for p, v in zip(point_packets, vis_frames)}
                point_packets = filter_frames_for_calibration(
                    point_packets, target_count=self.max_calibration_frames
                )
                vis_frames = [vis_by_id[id(p)] for p in point_packets if id(p) in vis_by_id]

            self.log_message.emit(f"Calibrating from {len(point_packets)} frames...")
            self.selected_frames.emit(vis_frames)

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
    detection_frame = Signal(int, int, object, int)  # port, frame_index, frame (BGR), corner_count
    calibration_finished = Signal(object, float)  # cameras dict, error
    error = Signal(str)

    def __init__(
        self,
        video_paths: dict[int, Path],
        intrinsics: dict[int, "CameraIntrinsics"],
        charuco_config: "CharucoConfig",
        frame_time_csv: "Path | None" = None,
    ):
        super().__init__()
        self.video_paths = video_paths
        self.intrinsics = intrinsics
        self.charuco_config = charuco_config
        self.frame_time_csv = frame_time_csv
        self.running = True

    def _draw_charuco_overlay(self, frame, packet):
        """Draw charuco detection visualization on frame."""
        import cv2
        import numpy as np

        vis = frame.copy()
        ids = packet.point_id
        img_loc = packet.img_loc
        n = len(ids)

        if n == 0:
            return vis

        cols = self.charuco_config.columns - 1

        for i in range(n):
            pt = (int(img_loc[i, 0]), int(img_loc[i, 1]))
            cid = int(ids[i])

            hue = (cid * 17) % 180
            color_bgr = cv2.cvtColor(
                np.array([[[hue, 200, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
            )[0, 0]
            color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

            cv2.circle(vis, pt, 6, color, 2)
            cv2.circle(vis, pt, 2, color, -1)

            cv2.putText(
                vis, str(cid), (pt[0] + 8, pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
            )

        # Connect adjacent corners
        for i in range(n):
            cid_i = int(ids[i])
            row_i, col_i = divmod(cid_i, cols)
            for j in range(i + 1, n):
                cid_j = int(ids[j])
                row_j, col_j = divmod(cid_j, cols)
                if (row_i == row_j and abs(col_i - col_j) == 1) or \
                   (col_i == col_j and abs(row_i - row_j) == 1):
                    pt1 = (int(img_loc[i, 0]), int(img_loc[i, 1]))
                    pt2 = (int(img_loc[j, 0]), int(img_loc[j, 1]))
                    cv2.line(vis, pt1, pt2, (0, 200, 0), 1, cv2.LINE_AA)

        cv2.putText(
            vis, f"{n} corners detected", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )

        return vis

    def run(self):
        try:
            from ..calibration.extrinsic import run_extrinsic_from_videos

            def progress(fraction: float, message: str):
                self.log_message.emit(message)
                self.progress_update.emit(fraction)

            def on_frame(port, frame_idx, frame_bgr, packet):
                corner_count = len(packet.point_id)
                vis = self._draw_charuco_overlay(frame_bgr, packet)
                self.detection_frame.emit(port, frame_idx, vis, corner_count)

            cameras, rmse = run_extrinsic_from_videos(
                video_paths=self.video_paths,
                intrinsics=self.intrinsics,
                charuco_config=self.charuco_config,
                frame_time_csv=self.frame_time_csv,
                progress_callback=progress,
                frame_callback=on_frame,
            )

            self.calibration_finished.emit(cameras, rmse)

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.running = False


class _PersonTrack:
    """Frame-to-frame 3D person track with COM-based identity persistence.

    Each track has a stable ``track_id`` that survives brief detection
    dropouts (up to ``grace_frames``).  Matching is done via 3D hip-COM
    proximity between consecutive frames.
    """

    __slots__ = (
        "track_id",
        "last_com_3d",
        "last_keypoints_3d",
        "frames_since_seen",
        "is_active",
    )

    def __init__(self, track_id: int, com_3d: "np.ndarray", keypoints_3d: list):
        self.track_id = track_id
        self.last_com_3d = com_3d            # (3,) hip midpoint
        self.last_keypoints_3d = keypoints_3d  # list[np.ndarray | None], len 17
        self.frames_since_seen: int = 0
        self.is_active: bool = True

    def update(self, com_3d: "np.ndarray", keypoints_3d: list):
        self.last_com_3d = com_3d
        self.last_keypoints_3d = keypoints_3d
        self.frames_since_seen = 0

    def increment_lost(self, grace_frames: int = 5):
        self.frames_since_seen += 1
        if self.frames_since_seen > grace_frames:
            self.is_active = False


class _PersonTracker:
    """Manages a set of ``_PersonTrack`` instances across frames.

    Responsibilities:
    * Match new 3D detections to existing tracks via COM proximity.
    * Hold last-known keypoints for tracks that temporarily lose detection.
    * Cap the number of tracked persons at ``max_persons``.
    * Determine which person is the "primary" subject (closest to the
      calibrated origin, i.e. the person doing the exercise).
    """

    def __init__(self, max_persons: int = 2, grace_frames: int = 5,
                 max_com_jump_m: float = 0.3):
        self.max_persons = max_persons
        self.grace_frames = grace_frames
        self.max_com_jump_m = max_com_jump_m
        self._tracks: list[_PersonTrack] = []
        self._next_id: int = 0
        # Index into _tracks of the person closest to origin
        self.primary_track_id: int | None = None

    # ── public API ────────────────────────────────────────────────────

    def update(self, detections: list[tuple["np.ndarray", list]]):
        """Accept this frame's detections and update tracks.

        Parameters
        ----------
        detections : list of (com_3d, keypoints_3d)
            Each entry is a detected person in the current frame.
            ``com_3d`` is a (3,) hip midpoint, ``keypoints_3d`` is the
            list of 17 keypoints (np.ndarray | None).
        """
        import numpy as np

        active = [t for t in self._tracks if t.is_active]

        if not active and not detections:
            return

        # ── Greedy assignment by COM distance ──
        used_det: set[int] = set()
        used_track: set[int] = set()

        if active and detections:
            # Build cost matrix
            n_tracks = len(active)
            n_dets = len(detections)
            costs = np.full((n_tracks, n_dets), 1e9)
            for ti, track in enumerate(active):
                for di, (com, _kps) in enumerate(detections):
                    costs[ti, di] = float(np.linalg.norm(track.last_com_3d - com))

            # Greedy assign smallest cost first (fast enough for <=4 persons)
            for _ in range(min(n_tracks, n_dets)):
                idx = np.argmin(costs)
                ti, di = divmod(int(idx), n_dets)
                dist = costs[ti, di]
                if dist > self.max_com_jump_m:
                    break
                active[ti].update(detections[di][0], detections[di][1])
                used_track.add(ti)
                used_det.add(di)
                costs[ti, :] = 1e9
                costs[:, di] = 1e9

        # Increment lost counter for unmatched tracks
        for ti, track in enumerate(active):
            if ti not in used_track:
                track.increment_lost(self.grace_frames)

        # Create new tracks for unmatched detections (up to max_persons)
        n_active = sum(1 for t in self._tracks if t.is_active)
        for di, (com, kps) in enumerate(detections):
            if di in used_det:
                continue
            if n_active >= self.max_persons:
                break
            new_track = _PersonTrack(self._next_id, com, kps)
            self._next_id += 1
            self._tracks.append(new_track)
            n_active += 1

        # Prune dead tracks
        self._tracks = [t for t in self._tracks if t.is_active]

        # Update primary person (closest to world origin)
        self._update_primary()

    def get_ordered_persons(self) -> tuple[list[list], int]:
        """Return (persons_3d, primary_index).

        ``persons_3d`` is a list of keypoint lists, ordered by stable
        track ID.  ``primary_index`` is the index into that list of the
        person closest to the calibrated origin.
        """
        ordered = sorted(self._tracks, key=lambda t: t.track_id)
        persons = [t.last_keypoints_3d for t in ordered]
        primary_idx = 0
        for i, t in enumerate(ordered):
            if t.track_id == self.primary_track_id:
                primary_idx = i
                break
        return persons, primary_idx

    def reset(self):
        self._tracks.clear()
        self._next_id = 0
        self.primary_track_id = None

    # ── internals ─────────────────────────────────────────────────────

    def _update_primary(self):
        import numpy as np
        if not self._tracks:
            self.primary_track_id = None
            return
        best_track = min(self._tracks,
                         key=lambda t: float(np.linalg.norm(t.last_com_3d)))
        self.primary_track_id = best_track.track_id


class PoseDetectionWorker(QThread):
    """Live 2D pose detection on preview frames.

    Loads YOLO + VitPose once, then processes frames from a queue.
    Emits annotated BGR frames back to the GUI.
    """

    models_loaded = Signal()           # emitted once models are ready
    detection_ready = Signal(int, object)  # port, annotated BGR frame
    keypoints_3d_ready = Signal(list, int)  # list[list[np.ndarray | None]], primary_person_index
    log_message = Signal(str)
    error = Signal(str)

    # SynthPose 52-keypoint skeleton (same as skeleton_view.py)
    _SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (0, 17), (17, 5), (17, 6), (17, 48),
        (5, 19), (6, 18),
        (5, 7), (7, 9), (7, 21), (7, 23), (9, 25), (9, 27),
        (6, 8), (8, 10), (8, 20), (8, 22), (10, 24), (10, 26),
        (5, 11), (6, 12), (11, 12),
        (48, 51), (51, 50), (50, 49), (49, 29), (49, 28), (29, 31), (28, 30),
        (11, 13), (13, 15), (13, 33), (13, 35), (15, 37), (15, 39),
        (12, 14), (14, 16), (14, 32), (14, 34), (16, 36), (16, 38),
        (15, 46), (15, 41), (41, 43), (43, 45),
        (16, 47), (16, 40), (40, 42), (42, 44),
        (5, 6),
    ]

    # Per-person color palette (BGR) — 8 distinct colors
    _PERSON_COLORS = [
        (120, 200, 80),   # green
        (255, 160, 100),  # blue
        (80, 180, 255),   # orange
        (220, 100, 220),  # purple
        (100, 100, 255),  # red
        (220, 220, 100),  # cyan
        (80, 220, 255),   # yellow
        (255, 140, 180),  # lavender
    ]

    def __init__(self, device_name: str = "auto", cameras: dict | None = None,
                 max_persons: int = 2):
        super().__init__()
        self.device_name = device_name
        self.cameras = cameras  # dict[port, CalibratedCamera] or None
        self.running = True
        self._models = None
        self._device = None

        # Tunable thresholds (set from GUI sliders)
        self.confidence_threshold = 0.3
        # Raw 2D keypoints per port for live triangulation
        # port -> list of (keypoints (17,2), scores (17,)) — one entry per detected person
        self._last_kps_per_port: dict[int, list[tuple["np.ndarray", "np.ndarray"]]] = {}

        # Frame-to-frame person tracker (COM-based identity persistence)
        self._person_tracker = _PersonTracker(
            max_persons=max_persons,
            grace_frames=5,
            max_com_jump_m=0.3,
        )

        # Frame queue: stores (port, frame_bgr). Only keep latest per port.
        import threading
        self._lock = threading.Lock()
        self._pending: dict[int, "np.ndarray"] = {}
        self._has_work = threading.Event()

    def submit_frame(self, port: int, frame: "np.ndarray"):
        """Submit a frame for detection (non-blocking, keeps only latest per port)."""
        with self._lock:
            self._pending[port] = frame
        self._has_work.set()

    def run(self):
        try:
            from ..tracking.pose_detector import setup_device, load_models
            self._device = setup_device(self.device_name)
            self.log_message.emit(f"Loading pose models on {self._device}...")

            person_model, pose_processor, pose_model = load_models(
                device=self._device,
                log_fn=lambda msg: self.log_message.emit(msg),
            )
            self._models = (person_model, pose_processor, pose_model)
            self.models_loaded.emit()
            self.log_message.emit("Live detection ready")

        except Exception as e:
            self.error.emit(f"Failed to load pose models: {e}")
            return

        # Main loop: wait for frames, run detection
        while self.running:
            self._has_work.wait(timeout=0.1)
            if not self.running:
                break
            self._has_work.clear()

            # Grab all pending frames
            with self._lock:
                work = dict(self._pending)
                self._pending.clear()

            if not work:
                continue

            for port, frame_bgr in work.items():
                if not self.running:
                    break
                try:
                    annotated = self._detect_and_draw(port, frame_bgr)
                    self.detection_ready.emit(port, annotated)
                except Exception:
                    # On detection error, just pass through original frame
                    self.detection_ready.emit(port, frame_bgr)

            # Attempt live triangulation if calibration available
            if self.cameras is not None and len(self._last_kps_per_port) >= 2:
                self._triangulate_live()

    def _detect_and_draw(self, port: int, frame_bgr: "np.ndarray") -> "np.ndarray":
        """Run detection on a single frame and draw keypoints with per-person colors."""
        import cv2
        import numpy as np
        from PIL import Image

        from ..tracking.pose_detector import detect_persons, estimate_poses

        person_model, pose_processor, pose_model = self._models

        # BGR -> RGB -> PIL
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Detect persons
        boxes_voc, boxes_coco, scores = detect_persons(
            pil_image, person_model, self._device, confidence_threshold=self.confidence_threshold
        )

        if boxes_voc.size == 0:
            # No detections: remove stale keypoints for this port
            self._last_kps_per_port.pop(port, None)
            return frame_bgr

        # Estimate poses
        all_keypoints, all_scores = estimate_poses(
            pil_image, boxes_coco, pose_processor, pose_model, self._device,
        )

        # Store all persons' raw keypoints for live triangulation
        if all_keypoints:
            self._last_kps_per_port[port] = list(zip(all_keypoints, all_scores))

        # Draw on frame
        vis = frame_bgr.copy()
        n_colors = len(self._PERSON_COLORS)

        for person_idx, (kps, kp_scores) in enumerate(zip(all_keypoints, all_scores)):
            color = self._PERSON_COLORS[person_idx % n_colors]

            # Brighter variant for keypoints
            kp_color = tuple(min(255, int(c * 1.3)) for c in color)

            n = kps.shape[0]

            # Draw limbs
            for i, j in self._SKELETON:
                if i >= n or j >= n:
                    continue
                if kp_scores[i] < 0.3 or kp_scores[j] < 0.3:
                    continue
                pt1 = (int(kps[i, 0]), int(kps[i, 1]))
                pt2 = (int(kps[j, 0]), int(kps[j, 1]))
                cv2.line(vis, pt1, pt2, color, 2, cv2.LINE_AA)

            # Draw keypoints
            for k in range(n):
                if kp_scores[k] < 0.3:
                    continue
                pt = (int(kps[k, 0]), int(kps[k, 1]))
                cv2.circle(vis, pt, 4, kp_color, -1, cv2.LINE_AA)

            # Draw bounding box + person index
            if person_idx < len(boxes_voc):
                box = boxes_voc[person_idx].astype(int)
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]),
                              color, 1, cv2.LINE_AA)
                cv2.putText(vis, f"P{person_idx}", (box[0], box[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return vis

    # ── Live triangulation helpers ────────────────────────────────────────

    @staticmethod
    def _hip_com_2d(kps: "np.ndarray", scores: "np.ndarray") -> "np.ndarray | None":
        """Return 2D hip COM as (x, y, score) array, or None if both hips are weak."""
        import numpy as np
        pts, sc = [], []
        for idx in (11, 12):   # L_hip=11, R_hip=12
            if scores[idx] >= 0.3:
                pts.append(kps[idx])
                sc.append(float(scores[idx]))
        if not pts:
            return None
        return np.array([*np.mean(pts, axis=0), np.mean(sc)], dtype=float)   # (3,)

    @staticmethod
    def _is_in_front(pt3d: "np.ndarray", cam_params_subset: list) -> bool:
        """Return True when pt3d has positive depth in ALL listed cameras."""
        import numpy as np, cv2
        for cp in cam_params_subset:
            R, _ = cv2.Rodrigues(np.array(cp["rotation"]))
            depth = float((R @ pt3d + cp["translation"])[2])
            if depth <= 0:
                return False
        return True

    def _match_persons_hip_com(
        self,
        port_persons: "dict[int, list[tuple]]",
        port_to_cam_index: "dict[int, int]",
        camera_params: list,
        projection_matrices: list,
    ) -> "list[dict[int, int]]":
        """Match persons across ports via 3D hip-COM triangulation.

        For each reference person i and each other port Q, triangulates the
        hip COM from (ref_port, Q) for all candidate j in Q.  Keeps only
        triangulations that land in front of both cameras (eliminates ghost
        positions), then greedily assigns the best j to each i.

        Returns a list of groups: each group is {port: person_idx} covering
        at least the reference port plus one other port.
        """
        import numpy as np
        from ..tracking.triangulation import triangulate_keypoints

        ref_port = max(port_persons, key=lambda p: len(port_persons[p]))
        n_ref = len(port_persons[ref_port])
        groups: list[dict[int, int]] = [{ref_port: i} for i in range(n_ref)]

        hip_coms = {
            port: [self._hip_com_2d(kps, sc) for kps, sc in persons]
            for port, persons in port_persons.items()
        }

        ref_cam = camera_params[port_to_cam_index[ref_port]]

        for other_port in port_persons:
            if other_port == ref_port:
                continue
            other_cam = camera_params[port_to_cam_index[other_port]]
            claimed: set[int] = set()
            n_other = len(port_persons[other_port])

            for i in range(n_ref):
                ref_hip = hip_coms[ref_port][i]
                if ref_hip is None:
                    continue

                # Triangulate hip COM for every unclaimed j in the other port
                valid_j: dict[int, "np.ndarray"] = {}
                for j in range(n_other):
                    if j in claimed:
                        continue
                    other_hip = hip_coms[other_port][j]
                    if other_hip is None:
                        continue
                    result = triangulate_keypoints(
                        {ref_port: ref_hip[np.newaxis], other_port: other_hip[np.newaxis]},
                        port_to_cam_index, camera_params, projection_matrices,
                    )
                    pt3d = result[0] if result else None
                    if pt3d is None or np.isnan(pt3d).any():
                        continue
                    # Only keep triangulations that are physically in front of both cameras
                    if self._is_in_front(pt3d, [ref_cam, other_cam]):
                        valid_j[j] = pt3d

                if not valid_j:
                    continue

                # Among valid candidates, prefer the one closest to the scene origin
                # (robust heuristic: real people are closer than ghost intersections)
                best_j = min(valid_j, key=lambda j: float(np.linalg.norm(valid_j[j])))
                groups[i][other_port] = best_j
                claimed.add(best_j)

        return [g for g in groups if len(g) >= 2]

    @staticmethod
    def _com_3d_from_keypoints(kps_3d: list) -> "np.ndarray | None":
        """Compute 3D hip midpoint from a triangulated keypoint list."""
        import numpy as np
        L_HIP, R_HIP = 11, 12
        pts = []
        for idx in (L_HIP, R_HIP):
            if idx < len(kps_3d) and kps_3d[idx] is not None and not np.isnan(kps_3d[idx]).any():
                pts.append(np.asarray(kps_3d[idx], dtype=float))
        if not pts:
            return None
        return np.mean(pts, axis=0)

    def _triangulate_live(self):
        """Triangulate 3D keypoints for all matched persons, feed the
        person tracker, and emit stable-ordered results."""
        try:
            import cv2
            import numpy as np
            from ..tracking.triangulation import calculate_projection_matrices, triangulate_keypoints

            # Build camera_params
            sorted_ports = sorted(self.cameras.keys())
            camera_params, port_to_cam_index = [], {}
            for i, port in enumerate(sorted_ports):
                cam = self.cameras[port]
                rvec, _ = cv2.Rodrigues(cam.extrinsics.rotation)
                camera_params.append({
                    "matrix": cam.intrinsics.matrix,
                    "distortions": cam.intrinsics.distortion,
                    "size": np.array(cam.intrinsics.resolution),
                    "rotation": rvec.flatten(),
                    "translation": cam.extrinsics.translation,
                    "port": port,
                })
                port_to_cam_index[port] = i

            projection_matrices = calculate_projection_matrices(camera_params)

            port_persons = {
                port: persons
                for port, persons in self._last_kps_per_port.items()
                if port in port_to_cam_index and persons
            }
            if len(port_persons) < 2:
                return

            # Match persons across ports by 3D hip COM
            groups = self._match_persons_hip_com(
                port_persons, port_to_cam_index, camera_params, projection_matrices
            )

            # Triangulate full 17 keypoints for each matched group
            frame_detections: list[tuple[np.ndarray, list]] = []
            for group in groups:
                kp_dict = {}
                for port, person_idx in group.items():
                    kps, scores = port_persons[port][person_idx]
                    kp_dict[port] = np.concatenate([kps, scores[:, None]], axis=1)
                kps_3d = triangulate_keypoints(
                    kp_dict, port_to_cam_index, camera_params, projection_matrices
                )
                com = self._com_3d_from_keypoints(kps_3d)
                if com is not None:
                    frame_detections.append((com, kps_3d))

            # Feed into the person tracker for stable IDs + grace period
            self._person_tracker.update(frame_detections)

            persons_3d, primary_idx = self._person_tracker.get_ordered_persons()
            if persons_3d:
                self.keypoints_3d_ready.emit(persons_3d, primary_idx)
        except Exception:
            pass

    def stop(self):
        self.running = False
        self._has_work.set()  # unblock the wait


class MediaPipeHandsDetectionWorker(QThread):
    """Live hand detection using MediaPipe Hands.

    Same signal interface as PoseDetectionWorker. Runs MediaPipe on each
    camera frame independently (2D only — no triangulation). Draws hand
    landmarks + connections on the annotated frame. Emits hand landmark
    positions via keypoints_3d_ready (using 2D pixel coords as x,y and 0 as z
    so the signal type matches, but the skeleton view won't be meaningful).
    """

    models_loaded = Signal()
    detection_ready = Signal(int, object)
    keypoints_3d_ready = Signal(list, int)
    log_message = Signal(str)
    error = Signal(str)

    def __init__(self, max_hands: int = 2):
        super().__init__()
        self.max_hands = max_hands
        self.running = True

        import threading
        self._lock = threading.Lock()
        self._pending_frames: dict[int, "np.ndarray"] = {}

    def submit_frame(self, port: int, frame: "np.ndarray"):
        with self._lock:
            self._pending_frames[port] = frame.copy()

    def run(self):
        try:
            from ..tracking.hand_detector import detect_hands, get_thumb_index_distance
            import cv2
            import numpy as np
        except Exception as e:
            self.error.emit(f"MediaPipe Hands init failed: {e}")
            return

        self.models_loaded.emit()
        self.log_message.emit("[mediapipe_hands] Ready")

        # MediaPipe hand connections for drawing
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # index
            (0, 9), (9, 10), (10, 11), (11, 12),  # middle
            (0, 13), (13, 14), (14, 15), (15, 16), # ring
            (0, 17), (17, 18), (18, 19), (19, 20), # pinky
            (5, 9), (9, 13), (13, 17),             # palm
        ]

        while self.running:
            with self._lock:
                if self._pending_frames:
                    frames_snapshot = dict(self._pending_frames)
                    self._pending_frames.clear()
                else:
                    frames_snapshot = {}

            if not frames_snapshot:
                time.sleep(0.01)
                continue

            for port, bgr in frames_snapshot.items():
                try:
                    hands = detect_hands(bgr, max_hands=self.max_hands)
                    annotated = bgr.copy()
                    h, w = bgr.shape[:2]

                    for hand in hands:
                        # hand is a list of 21 (x, y, z) normalized landmarks
                        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in hand]

                        # Draw connections
                        for i, j in HAND_CONNECTIONS:
                            if i < len(pts) and j < len(pts):
                                cv2.line(annotated, pts[i], pts[j],
                                         (0, 255, 128), 2, cv2.LINE_AA)

                        # Draw landmarks
                        for pt in pts:
                            cv2.circle(annotated, pt, 3, (0, 200, 255),
                                       -1, cv2.LINE_AA)

                        # Show thumb-index distance
                        if len(hand) >= 9:
                            dist = get_thumb_index_distance(hand)
                            thumb_pt = pts[4]
                            cv2.putText(annotated,
                                        f"{dist:.0f}px",
                                        (thumb_pt[0] + 10, thumb_pt[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 255), 1, cv2.LINE_AA)

                    self.detection_ready.emit(port, annotated)

                except Exception as e:
                    self.log_message.emit(f"[mediapipe_hands] Error on port {port}: {e}")

    def stop(self):
        self.running = False


class CudaStreamDetectionWorker(QThread):
    """Live 3D pose detection using the CUDA TensorRT streaming pipeline.

    Same signal interface as PoseDetectionWorker so the GUI can swap between
    them transparently. This worker:
    - Runs detection + matching + triangulation + tracking on GPU (~10ms/frame)
    - Reprojects 3D keypoints to 2D per camera for overlays (zero GPU cost)
    - Has its own built-in multiperson tracker with stable person IDs
    """

    models_loaded = Signal()
    detection_ready = Signal(int, object)      # port, annotated_frame
    keypoints_3d_ready = Signal(list, int)     # persons, primary_index
    log_message = Signal(str)
    error = Signal(str)

    # SynthPose 52-keypoint skeleton (CUDA overlay, same side-correct ordering)
    _SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (0, 17), (17, 5), (17, 6), (17, 48),
        (5, 19), (6, 18),
        (5, 7), (7, 9), (7, 21), (7, 23), (9, 25), (9, 27),
        (6, 8), (8, 10), (8, 20), (8, 22), (10, 24), (10, 26),
        (5, 11), (6, 12), (11, 12),
        (48, 51), (51, 50), (50, 49), (49, 29), (49, 28), (29, 31), (28, 30),
        (11, 13), (13, 15), (13, 33), (13, 35), (15, 37), (15, 39),
        (12, 14), (14, 16), (14, 32), (14, 34), (16, 36), (16, 38),
        # Left foot (15=L_Ankle)
        (15, 46), (15, 41), (41, 43), (43, 45),  # l_calc, l_5meta→l_toe→l_big_toe
        # Right foot (16=R_Ankle)
        (16, 47), (16, 40), (40, 42), (42, 44),  # r_calc, r_5meta→r_toe→r_big_toe
        (5, 6),
    ]
    _PERSON_COLORS = [
        (120, 200, 80), (255, 160, 100), (220, 100, 220),
        (100, 100, 255), (100, 220, 220), (255, 220, 80),
    ]

    def __init__(self, cameras: dict, calibration_path: str,
                 yolo_onnx: str = "", vitpose_onnx: str = "",
                 engine_cache: str = "", max_persons: int = 2):
        super().__init__()
        self.cameras = cameras
        self.calibration_path = calibration_path
        self.yolo_onnx = yolo_onnx
        self.vitpose_onnx = vitpose_onnx
        self.engine_cache = engine_cache
        self.max_persons = max_persons
        self.running = True
        self._pipeline = None
        self._proj_matrices: dict[int, "np.ndarray"] = {}  # port → 3x4

        import threading
        self._lock = threading.Lock()
        self._pending_frames: dict[int, "np.ndarray"] = {}
        self._sync_index = 0

    def submit_frame(self, port: int, frame: "np.ndarray"):
        """Submit a frame for detection (non-blocking, keeps only latest per port)."""
        with self._lock:
            self._pending_frames[port] = frame.copy()

    def _build_projection_matrices(self):
        """Precompute 3x4 projection matrices for each camera (for 3D→2D reprojection)."""
        import numpy as np
        for port, cam in self.cameras.items():
            K = np.asarray(cam.intrinsics.matrix, dtype=np.float64)
            R = np.asarray(cam.extrinsics.rotation, dtype=np.float64)
            t = np.asarray(cam.extrinsics.translation, dtype=np.float64).reshape(3, 1)
            Rt = np.hstack([R, t])
            self._proj_matrices[port] = K @ Rt

    def _project_3d_to_2d(self, point_3d, proj_matrix):
        """Project a 3D point to 2D pixel coordinates. Returns (x, y) or None."""
        import numpy as np
        if point_3d is None:
            return None
        pt = np.asarray(point_3d, dtype=np.float64)
        if pt.shape != (3,) or np.isnan(pt).any():
            return None
        h = proj_matrix @ np.append(pt, 1.0)
        if abs(h[2]) < 1e-6:
            return None
        return (int(h[0] / h[2]), int(h[1] / h[2]))

    def _draw_overlay(self, frame, persons_3d, port):
        """Draw reprojected 3D skeletons onto a BGR frame for one camera.

        Cost: ~0.2ms on CPU per frame (just cv2.circle + cv2.line).
        """
        import cv2
        import numpy as np

        proj = self._proj_matrices.get(port)
        if proj is None:
            return frame

        annotated = frame.copy()

        for p_idx, person_kps in enumerate(persons_3d):
            color = self._PERSON_COLORS[p_idx % len(self._PERSON_COLORS)]
            pts_2d = [self._project_3d_to_2d(kp, proj) for kp in person_kps]

            # Draw limbs
            for i, j in self._SKELETON:
                if i < len(pts_2d) and j < len(pts_2d):
                    pi, pj = pts_2d[i], pts_2d[j]
                    if pi is not None and pj is not None:
                        cv2.line(annotated, pi, pj, color, 2, cv2.LINE_AA)

            # Draw keypoints
            for pt in pts_2d:
                if pt is not None:
                    cv2.circle(annotated, pt, 4, color, -1, cv2.LINE_AA)

        return annotated

    def run(self):
        import numpy as np

        try:
            from ..tracking.cuda_stream_binding import CudaStreamPipeline

            sorted_ports = sorted(self.cameras.keys())
            first_cam = self.cameras[sorted_ports[0]]
            w, h = first_cam.intrinsics.resolution

            self.log_message.emit(
                f"[cuda_stream] Initializing pipeline "
                f"({len(sorted_ports)} cameras, {w}x{h})..."
            )

            def _log(msg):
                self.log_message.emit(f"[cuda_stream] {msg}")

            self._pipeline = CudaStreamPipeline(
                num_cameras=len(sorted_ports),
                frame_width=w,
                frame_height=h,
                calibration_toml_path=self.calibration_path,
                yolo_onnx_path=self.yolo_onnx,
                vitpose_onnx_path=self.vitpose_onnx,
                engine_cache_dir=self.engine_cache,
                max_persons=self.max_persons,
                log_callback=_log,
            )

            self._build_projection_matrices()

            self.models_loaded.emit()
            self.log_message.emit("[cuda_stream] Pipeline ready")

        except Exception as e:
            self.error.emit(f"CUDA pipeline init failed: {e}")
            return

        frames_snapshot = {}
        frame_count = 0

        while self.running:
            with self._lock:
                if self._pending_frames:
                    frames_snapshot = dict(self._pending_frames)
                    self._pending_frames.clear()

            if not frames_snapshot:
                time.sleep(0.005)
                continue

            try:
                frame_list = []
                for port in sorted(frames_snapshot.keys()):
                    frame_list.append((frames_snapshot[port], port))

                result = self._pipeline.process_frame(frame_list, self._sync_index)
                self._sync_index += 1
                frame_count += 1

                if frame_count <= 3 or frame_count % 100 == 0:
                    self.log_message.emit(
                        f"[cuda_stream] frame {frame_count}: "
                        f"{len(frame_list)} cams, "
                        f"{result.num_persons} persons, "
                        f"{result.processing_time_ms:.1f}ms"
                    )

                # Build persons list
                all_persons_3d = []
                primary_index = 0
                min_origin_dist = float("inf")

                for i, person in enumerate(result.persons):
                    all_persons_3d.append(person.keypoints_3d)
                    if person.com_3d is not None:
                        dist = float(np.linalg.norm(person.com_3d))
                        if dist < min_origin_dist:
                            min_origin_dist = dist
                            primary_index = i

                # Emit 3D keypoints
                if all_persons_3d:
                    self.keypoints_3d_ready.emit(all_persons_3d, primary_index)

                # Draw 2D overlays by reprojecting 3D → 2D per camera
                for port, bgr in frames_snapshot.items():
                    if all_persons_3d:
                        annotated = self._draw_overlay(bgr, all_persons_3d, port)
                    else:
                        annotated = bgr
                    self.detection_ready.emit(port, annotated)

            except Exception as e:
                self.log_message.emit(f"[cuda_stream] Frame error: {e}")
                import traceback
                self.log_message.emit(f"[cuda_stream] {traceback.format_exc()}")

            frames_snapshot = {}

        # Cleanup
        if self._pipeline is not None:
            stats = self._pipeline.get_stats()
            if stats.frames_processed > 0:
                avg = stats.total_ms / stats.frames_processed
                self.log_message.emit(
                    f"[cuda_stream] Processed {stats.frames_processed} frames, "
                    f"avg {avg:.1f}ms/frame"
                )
            self._pipeline.close()
            self._pipeline = None

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
        person_confidence: float = 0.30,
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
        self.person_confidence = person_confidence
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
                person_confidence=self.person_confidence,
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
