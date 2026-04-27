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


class _LiveTracker:
    """Cross-frame association of triangulated 3D persons by hip-COM continuity.

    Light port of tracker.assign_3d_candidates_to_tracks for the live PyTorch
    worker. The cost matrix is tiny (≤ N_persons × N_persons, typically 2×2 or
    3×3) so Hungarian + COM bookkeeping costs microseconds per frame. Stable
    track ids let downstream consumers (CSV export, foot-placement viz) trust
    that "person 0 in frame N+1" really is "person 0 from frame N".

    Track ids start at 1; 0 is reserved for "untrackable" (no valid hip COM).
    """

    def __init__(
        self,
        hip_indices: "tuple[int, int] | None" = None,
        max_match_distance: float = 0.5,   # meters; ≤ this still considered the same person
        patience: int = 10,                # frames a track can be lost before being dropped
        max_persons: int = 8,
    ):
        # Lazy-import HIP_INDICES so Qt threads don't trigger heavy imports
        # in module init.
        if hip_indices is None:
            from ..tracking.markers import HIP_INDICES as _HIP
            hip_indices = _HIP
        self.hip_indices = hip_indices
        self.max_match_distance = max_match_distance
        self.patience = patience
        self.max_persons = max_persons
        self._tracks: list[dict] = []   # each: {id, com (np.ndarray), frames_since_seen}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    def _hip_com_from_kps(self, kps_3d: list) -> "np.ndarray | None":
        import numpy as _np
        valid = []
        for hip_idx in self.hip_indices:
            if hip_idx < len(kps_3d):
                kp = kps_3d[hip_idx]
                if kp is not None:
                    arr = _np.asarray(kp, dtype=float)
                    if arr.size >= 3 and not _np.isnan(arr[:3]).any():
                        valid.append(arr[:3])
        if valid:
            return _np.mean(valid, axis=0)
        return None

    def step(self, persons_kps: list) -> list[int]:
        """Given the current frame's triangulated persons, return parallel ids.

        Each entry in `persons_kps` is `list[np.ndarray | None]` (length = num
        keypoints). Returns a list of the same length. id 0 means "could not
        compute COM, no track assigned this frame".
        """
        import numpy as _np
        from scipy.optimize import linear_sum_assignment as _lsa

        n = len(persons_kps)
        coms = [self._hip_com_from_kps(kps) for kps in persons_kps]

        if n == 0:
            for t in self._tracks:
                t["frames_since_seen"] += 1
            self._tracks = [t for t in self._tracks
                            if t["frames_since_seen"] <= self.patience]
            return []

        # No existing tracks: assign new ids to anyone with a valid COM.
        if not self._tracks:
            ids = [0] * n
            for j, c in enumerate(coms):
                if c is not None and len(self._tracks) < self.max_persons:
                    self._tracks.append({"id": self._next_id, "com": c,
                                         "frames_since_seen": 0})
                    ids[j] = self._next_id
                    self._next_id += 1
            return ids

        # Build cost matrix: tracks × candidates. Far-apart pairs get HI so
        # the linear assignment can't pick them.
        m = len(self._tracks)
        HI = 1e6
        cost = _np.full((m, n), HI)
        for i, t in enumerate(self._tracks):
            for j, c in enumerate(coms):
                if c is None:
                    continue
                d = float(_np.linalg.norm(t["com"] - c))
                if d <= self.max_match_distance:
                    cost[i, j] = d

        # Hungarian even for non-square matrices.
        rows, cols = _lsa(cost)

        ids = [0] * n
        track_seen = [False] * m
        person_assigned = [False] * n
        for r, c in zip(rows, cols):
            if cost[r, c] < HI:
                ids[c] = self._tracks[r]["id"]
                self._tracks[r]["com"] = coms[c]
                self._tracks[r]["frames_since_seen"] = 0
                track_seen[r] = True
                person_assigned[c] = True

        # Unmatched candidates spawn fresh tracks (subject to max_persons).
        for j in range(n):
            if person_assigned[j] or coms[j] is None:
                continue
            if len(self._tracks) >= self.max_persons:
                break
            self._tracks.append({"id": self._next_id, "com": coms[j],
                                 "frames_since_seen": 0})
            ids[j] = self._next_id
            self._next_id += 1

        # Age unmatched tracks; drop those past patience.
        for i, t in enumerate(self._tracks[:m]):  # only original tracks
            if not track_seen[i]:
                t["frames_since_seen"] += 1
        self._tracks = [t for t in self._tracks
                        if t["frames_since_seen"] <= self.patience]

        return ids


class PoseDetectionWorker(QThread):
    """Live 2D pose detection on preview frames.

    Loads YOLO + VitPose once, then processes frames from a queue.
    Emits annotated BGR frames back to the GUI.
    """

    models_loaded = Signal()           # emitted once models are ready
    detection_ready = Signal(int, object)  # port, annotated BGR frame
    keypoints_3d_ready = Signal(list)  # list[list[np.ndarray | None]] — one entry per person
    log_message = Signal(str)
    error = Signal(str)

    # SynthPose-52 skeleton (superset of COCO-17, matches skeleton_view.py)
    _SKELETON = [
        # Head
        (0, 1), (0, 2), (1, 3), (2, 4),
        # Neck / shoulders
        (0, 17), (17, 5), (17, 6), (17, 48),
        (5, 19), (6, 18),
        # Left arm: 5→7→9 with landmarks
        (5, 7), (7, 9),
        (7, 21), (7, 23),
        (9, 25), (9, 27),
        # Right arm: 6→8→10 with landmarks
        (6, 8), (8, 10),
        (8, 20), (8, 22),
        (10, 24), (10, 26),
        # Torso
        (5, 11), (6, 12), (11, 12),
        (48, 51), (51, 50), (50, 49),
        (49, 29), (49, 28),
        (29, 31), (28, 30),
        # Left leg: 11→13→15 with landmarks
        (11, 13), (13, 15),
        (13, 33), (13, 35),
        (15, 37), (15, 39),
        # Right leg: 12→14→16 with landmarks
        (12, 14), (14, 16),
        (14, 32), (14, 34),
        (16, 36), (16, 38),
        # Left foot
        (15, 46), (15, 41),
        (41, 43), (43, 45),
        # Right foot
        (16, 47), (16, 40),
        (40, 42), (42, 44),
        # Fallback (used when SynthPose kps 17+ absent)
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

    def __init__(self, device_name: str = "auto", cameras: dict | None = None):
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

        # Cross-frame person tracking. Stable ids parallel to the persons list
        # emitted on keypoints_3d_ready, exposed as `last_person_ids` so
        # downstream consumers (CSV export, foot-placement viz) can correlate
        # person N from frame T with the same person in frame T+1.
        self._tracker = _LiveTracker()
        self.last_person_ids: list[int] = []

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
            # Fresh tracker state for every detection-worker run so track ids
            # don't carry over from a previous session.
            self._tracker.reset()
            self.last_person_ids = []

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

    def _triangulate_live(self):
        """Triangulate 3D keypoints for all matched persons and emit the results.

        Uses the same posetrack-style cross-view association as the batch
        pipeline: per-pair epipolar Hungarian on hip-COM, then union-find to
        merge into groups. Greedy "closest to origin" matching mis-associates
        people who are at similar depths.
        """
        try:
            import cv2
            import numpy as np
            from ..tracking.triangulation import calculate_projection_matrices, triangulate_keypoints
            from ..tracking.tracker import (
                group_detections_across_views_bipartite,
                calculate_2d_com,
            )
            from ..tracking.markers import HIP_INDICES

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

            # Build the per-port detection-dict format the bipartite grouper
            # expects: each detection carries `keypoints` (N,3 with score) and
            # the 2D hip COM used for the cost matrix.
            detected_persons_2d: dict[int, list[dict]] = {}
            for port, persons in port_persons.items():
                port_dets: list[dict] = []
                for kps, scores in persons:
                    # Stitch (x,y,score) per keypoint and compute COM in the
                    # same way the batch pipeline does.
                    kps_with_score = np.concatenate([kps, scores[:, None]], axis=1)
                    com_2d = calculate_2d_com(kps_with_score.tolist(), HIP_INDICES)
                    if com_2d is None:
                        continue
                    port_dets.append({
                        "keypoints": kps_with_score,
                        "com_2d": com_2d,
                    })
                if port_dets:
                    detected_persons_2d[port] = port_dets

            if len(detected_persons_2d) < 2:
                return

            groups = group_detections_across_views_bipartite(
                detected_persons_2d,
                projection_matrices,
                port_to_cam_index,
                camera_params,
                # Live frames have more detection noise than batch processing,
                # so loosen the gate compared to the batch default (30).
                epipolar_threshold=50.0,
            )

            # Triangulate full keypoints for each matched group
            all_persons_3d = []
            for group in groups:
                if len(group) < 2:
                    continue
                kp_dict = {port: det["keypoints"] for port, det in group.items()}
                kps_3d = triangulate_keypoints(
                    kp_dict, port_to_cam_index, camera_params, projection_matrices
                )
                all_persons_3d.append(kps_3d)

            # Cross-frame association: turn the per-frame group list into a
            # set of stable track ids before emitting. Stored on the worker
            # so downstream consumers (CSV export, foot-placement viz) can
            # correlate without changing the existing signal contract.
            self.last_person_ids = self._tracker.step(all_persons_3d)

            if all_persons_3d:
                self.keypoints_3d_ready.emit(all_persons_3d)
        except Exception:
            pass

    def stop(self):
        self.running = False
        self._has_work.set()  # unblock the wait


class MediaPipeHandsDetectionWorker(QThread):
    """Live 2D hand detection with multi-view 3D triangulation.

    Uses MediaPipe Hands to detect 21 hand landmarks per camera,
    matches hands across cameras by wrist landmark triangulation,
    and emits triangulated 3D hand landmarks.
    """

    models_loaded = Signal()
    detection_ready = Signal(int, object)  # port, annotated BGR frame
    keypoints_3d_ready = Signal(list)  # list[list[np.ndarray | None]] — one entry per hand
    log_message = Signal(str)
    error = Signal(str)

    # MediaPipe hand landmark connections for drawing
    _HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17),
    ]

    # Per-hand colors keyed by handedness (BGR)
    _HAND_COLOR_BY_LABEL = {
        "Left":    (80, 200, 255),   # yellow-orange
        "Right":   (255, 160, 100),  # blue
        "Unknown": (80, 255, 80),    # green
    }

    def __init__(self, cameras: dict | None = None, max_hands: int = 2):
        super().__init__()
        self.cameras = cameras  # dict[port, CalibratedCamera] or None
        self.max_hands = max_hands
        self.running = True
        # One detector per port — VIDEO mode requires monotonic per-stream
        # timestamps and per-stream tracking state to be stable.
        self._mp_hands_per_port: dict[int, object] = {}
        self._mp_options = None  # cached HandLandmarkerOptions for lazy per-port creation
        self._mp_module = None
        # Per-port monotonic timestamp counter (ms) for detect_for_video
        self._port_ts_ms: dict[int, int] = {}

        # Detection threshold (used to gate first-frame detection only)
        self.confidence_threshold = 0.5

        # Raw 2D landmarks per port for triangulation
        # port -> list of (landmarks (21,2), handedness str)
        self._last_hands_per_port: dict[int, list[tuple["np.ndarray", str]]] = {}

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
            import mediapipe as mp
            from ..config import mediapipe_dir

            # Cache hand_landmarker.task under the app data dir
            model_dir = mediapipe_dir()
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "hand_landmarker.task"
            if not model_path.exists():
                msg = (f"[DOWNLOAD] hand_landmarker.task not cached at "
                       f"{model_path} -- downloading from Google CDN")
                print(msg, flush=True)
                self.log_message.emit(msg)
                import urllib.request
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
                urllib.request.urlretrieve(url, str(model_path))
                done = f"[DOWNLOAD] hand_landmarker.task saved to {model_path}"
                print(done, flush=True)
                self.log_message.emit(done)
            else:
                msg = (f"[CACHE HIT] hand_landmarker.task loaded from "
                       f"{model_path} (no download)")
                print(msg, flush=True)
                self.log_message.emit(msg)

            # VIDEO mode is the stable choice for live streams: when a hand is
            # already tracked the detector skips full re-detection and runs the
            # cheaper tracker, which kills the per-frame flicker. Detection
            # threshold stays at 0.5 (only used to acquire a hand); the
            # presence/tracking thresholds are dropped so partial occlusions
            # don't cause repeated full re-detections.
            self._mp_module = mp
            self._mp_options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.confidence_threshold,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
            self.models_loaded.emit()
            self.log_message.emit("MediaPipe Hands ready (VIDEO mode)")
        except Exception as e:
            self.error.emit(f"Failed to load MediaPipe Hands: {e}")
            return

        while self.running:
            self._has_work.wait(timeout=0.1)
            if not self.running:
                break
            self._has_work.clear()

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
                    self.detection_ready.emit(port, frame_bgr)

            # Attempt triangulation if calibration available
            if self.cameras is not None and len(self._last_hands_per_port) >= 2:
                self._triangulate_hands()

        for det in self._mp_hands_per_port.values():
            try:
                det.close()
            except Exception:
                pass
        self._mp_hands_per_port.clear()

    def _detect_and_draw(self, port: int, frame_bgr: "np.ndarray") -> "np.ndarray":
        """Run MediaPipe Hands on a single frame and draw landmarks."""
        import cv2
        import numpy as np
        import mediapipe as mp

        # Lazy-create one VIDEO-mode detector per port (each maintains its
        # own tracker state).
        detector = self._mp_hands_per_port.get(port)
        if detector is None:
            if self._mp_options is None:
                # run() failed to initialize options
                return frame_bgr
            detector = mp.tasks.vision.HandLandmarker.create_from_options(self._mp_options)
            self._mp_hands_per_port[port] = detector
            self._port_ts_ms[port] = 0

        # Per-port monotonically increasing timestamp (ms). MediaPipe rejects
        # non-increasing timestamps, so use a counter rather than wall time
        # (works even if multiple frames arrive within the same millisecond).
        self._port_ts_ms[port] = self._port_ts_ms[port] + 1
        ts_ms = self._port_ts_ms[port]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect_for_video(mp_image, ts_ms)

        if not results.hand_landmarks:
            # Don't clear — let stale data persist so triangulation can
            # continue using the last known hand positions for a few frames.
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        vis = frame_bgr.copy()
        port_hands: list[tuple[np.ndarray, str]] = []

        for hand_idx, hand_lm in enumerate(results.hand_landmarks):
            handedness = "Unknown"
            if hand_idx < len(results.handedness):
                handedness = results.handedness[hand_idx][0].category_name
            color = self._HAND_COLOR_BY_LABEL.get(handedness, (80, 255, 80))

            # Extract 21 landmarks as pixel coordinates
            landmarks = np.array(
                [(lm.x * w, lm.y * h) for lm in hand_lm],
                dtype=np.float32,
            )
            port_hands.append((landmarks, handedness))

            # Draw connections
            for i, j in self._HAND_CONNECTIONS:
                pt1 = (int(landmarks[i, 0]), int(landmarks[i, 1]))
                pt2 = (int(landmarks[j, 0]), int(landmarks[j, 1]))
                cv2.line(vis, pt1, pt2, color, 2, cv2.LINE_AA)

            # Draw landmarks
            for k in range(21):
                pt = (int(landmarks[k, 0]), int(landmarks[k, 1]))
                cv2.circle(vis, pt, 3, color, -1, cv2.LINE_AA)

            # Label
            wrist = (int(landmarks[0, 0]), int(landmarks[0, 1]) - 10)
            cv2.putText(vis, handedness, wrist,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        self._last_hands_per_port[port] = port_hands
        return vis

    def _triangulate_hands(self):
        """Triangulate 3D hand landmarks across cameras.

        Matches hands across cameras by triangulating wrist landmarks (index 0)
        and finding pairs with valid (in-front-of-camera) 3D positions.
        """
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

            port_hands = {
                port: hands
                for port, hands in self._last_hands_per_port.items()
                if port in port_to_cam_index and hands
            }
            if len(port_hands) < 2:
                return

            # Match hands across cameras using wrist landmark (index 0)
            groups = self._match_hands_wrist(
                port_hands, port_to_cam_index, camera_params, projection_matrices
            )

            # Triangulate full 21 landmarks for each matched hand
            hands_with_label = []  # (handedness_str, kps_3d)
            for group in groups:
                kp_dict = {}
                # Use reference port's handedness label for stable ordering
                ref_port = max(group, key=lambda p: len(port_hands.get(p, [])))
                _, handedness = port_hands[ref_port][group[ref_port]]
                for port, hand_idx in group.items():
                    landmarks, _ = port_hands[port][hand_idx]
                    kp_with_conf = np.concatenate(
                        [landmarks, np.ones((21, 1), dtype=np.float32)], axis=1
                    )
                    kp_dict[port] = kp_with_conf
                kps_3d = triangulate_keypoints(
                    kp_dict, port_to_cam_index, camera_params, projection_matrices
                )
                hands_with_label.append((handedness, kps_3d))

            # Sort consistently: Left first, Right second
            hands_with_label.sort(key=lambda x: (0 if x[0] == "Left" else 1))
            all_hands_3d = [kps for _, kps in hands_with_label]

            if all_hands_3d:
                self.keypoints_3d_ready.emit(all_hands_3d)
        except Exception:
            pass

    def _match_hands_wrist(
        self,
        port_hands: "dict[int, list[tuple]]",
        port_to_cam_index: "dict[int, int]",
        camera_params: list,
        projection_matrices: list,
    ) -> "list[dict[int, int]]":
        """Match hands across ports via wrist (landmark 0) triangulation.

        Similar to PoseDetectionWorker._match_persons_hip_com but uses
        wrist position instead of hip COM.
        """
        import numpy as np
        from ..tracking.triangulation import triangulate_keypoints

        ref_port = max(port_hands, key=lambda p: len(port_hands[p]))
        n_ref = len(port_hands[ref_port])
        groups: list[dict[int, int]] = [{ref_port: i} for i in range(n_ref)]

        # Extract wrist 2D + confidence for each hand in each port
        wrist_pts: dict[int, list["np.ndarray | None"]] = {}
        for port, hands in port_hands.items():
            wrist_pts[port] = []
            for landmarks, _handedness in hands:
                # wrist is landmark 0
                wrist_2d = landmarks[0]
                wrist_pts[port].append(np.array([wrist_2d[0], wrist_2d[1], 1.0]))

        for other_port in port_hands:
            if other_port == ref_port:
                continue
            claimed: set[int] = set()
            n_other = len(port_hands[other_port])

            for i in range(n_ref):
                ref_wrist = wrist_pts[ref_port][i]
                if ref_wrist is None:
                    continue

                best_j = None
                best_dist = float("inf")

                for j in range(n_other):
                    if j in claimed:
                        continue
                    other_wrist = wrist_pts[other_port][j]
                    if other_wrist is None:
                        continue

                    result = triangulate_keypoints(
                        {ref_port: ref_wrist[np.newaxis], other_port: other_wrist[np.newaxis]},
                        port_to_cam_index, camera_params, projection_matrices,
                    )
                    pt3d = result[0] if result else None
                    if pt3d is None or np.isnan(pt3d).any():
                        continue

                    # Check if point is in front of both cameras
                    ref_cam = camera_params[port_to_cam_index[ref_port]]
                    other_cam = camera_params[port_to_cam_index[other_port]]
                    if not PoseDetectionWorker._is_in_front(pt3d, [ref_cam, other_cam]):
                        continue

                    dist = float(np.linalg.norm(pt3d))
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j

                if best_j is not None:
                    groups[i][other_port] = best_j
                    claimed.add(best_j)

        return [g for g in groups if len(g) >= 2]

    def stop(self):
        self.running = False
        self._has_work.set()


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


class CudaStreamDetectionWorker(QThread):
    """Live 2D/3D pose detection using the CUDA TensorRT streaming pipeline.

    Same interface as PoseDetectionWorker but uses CudaStreamPipeline
    for ~10ms/frame inference instead of PyTorch.
    """

    detection_ready = Signal(int, object)  # port, annotated BGR frame
    keypoints_3d_ready = Signal(list)      # list of person keypoints
    log_message = Signal(str)
    error = Signal(str)

    # SynthPose-52 skeleton (matches skeleton_view.py and PoseDetectionWorker)
    _SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (0, 17), (17, 5), (17, 6), (17, 48),
        (5, 19), (6, 18),
        (5, 7), (7, 9), (7, 21), (7, 23), (9, 25), (9, 27),
        (6, 8), (8, 10), (8, 20), (8, 22), (10, 24), (10, 26),
        (5, 11), (6, 12), (11, 12),
        (48, 51), (51, 50), (50, 49),
        (49, 29), (49, 28), (29, 31), (28, 30),
        (11, 13), (13, 15), (13, 33), (13, 35), (15, 37), (15, 39),
        (12, 14), (14, 16), (14, 32), (14, 34), (16, 36), (16, 38),
        (15, 46), (15, 41), (41, 43), (43, 45),
        (16, 47), (16, 40), (40, 42), (42, 44),
        (5, 6),
    ]

    _PERSON_COLORS = [
        (120, 200, 80), (255, 160, 100), (80, 180, 255), (220, 100, 220),
        (100, 100, 255), (220, 220, 100), (80, 220, 255), (255, 140, 180),
    ]

    def __init__(
        self,
        cameras: dict,
        calibration_path: str,
        yolo_onnx: str,
        vitpose_onnx: str,
        engine_cache: str,
        max_persons: int = 2,
        person_confidence: float = 0.50,
    ):
        super().__init__()
        self._cameras = cameras
        self._calibration_path = calibration_path
        self._yolo_onnx = yolo_onnx
        self._vitpose_onnx = vitpose_onnx
        self._engine_cache = engine_cache
        self._max_persons = max_persons
        self._person_confidence = float(person_confidence)
        # Stable track ids (parallel to the persons list emitted on
        # keypoints_3d_ready). Populated from PT_StreamPerson.person_id which
        # the C tracker assigns on its own.
        self.last_person_ids: list[int] = []
        self._pipeline = None
        self._sync_index = 0
        self.running = True

        import threading
        self._lock = threading.Lock()
        self._pending: dict[int, object] = {}
        self._has_work = threading.Event()

    def submit_frame(self, port: int, frame):
        """Submit a frame for detection (non-blocking, keeps only latest per port)."""
        with self._lock:
            self._pending[port] = frame
            have_all = len(self._pending) >= len(self._cameras)
        if have_all:
            self._has_work.set()

    def run(self):
        try:
            from ..tracking.cuda_stream_binding import CudaStreamPipeline

            ports = sorted(self._cameras.keys())
            w, h = self._cameras[ports[0]].intrinsics.resolution

            self.log_message.emit("Initializing CUDA pipeline...")
            self._pipeline = CudaStreamPipeline(
                num_cameras=len(ports),
                frame_width=w,
                frame_height=h,
                calibration_toml_path=self._calibration_path,
                yolo_onnx_path=self._yolo_onnx,
                vitpose_onnx_path=self._vitpose_onnx,
                engine_cache_dir=self._engine_cache,
                max_persons=self._max_persons,
                person_confidence=self._person_confidence,
            )
            self.log_message.emit("CUDA pipeline ready")
        except Exception as e:
            import traceback
            self.error.emit(f"CUDA pipeline init failed: {e}\n{traceback.format_exc()}")
            return

        import cv2

        ports = sorted(self._cameras.keys())
        w, h = self._cameras[ports[0]].intrinsics.resolution
        # Keep latest frame per port across iterations
        latest_frames: dict[int, object] = {}

        while self.running:
            self._has_work.wait(timeout=0.05)
            if not self.running:
                break
            self._has_work.clear()

            with self._lock:
                latest_frames.update(self._pending)
                self._pending.clear()

            # Need at least 2 cameras to triangulate
            if len(latest_frames) < 2:
                continue

            # Build frame list from latest frames
            frame_list = []
            raw_frames = {}
            for port in ports:
                if port in latest_frames:
                    f = cv2.resize(latest_frames[port], (w, h))
                    frame_list.append((f, port))
                    raw_frames[port] = f

            if len(frame_list) < 2:
                continue

            try:
                result = self._pipeline.process_frame(frame_list, sync_index=self._sync_index)
                self._sync_index += 1

                # Reproject 3D keypoints onto each camera's frame
                for port, frame in raw_frames.items():
                    vis = self._draw_reprojected(port, frame, result)
                    self.detection_ready.emit(port, vis)

                # Emit 3D keypoints. The C tracker's stable id is exposed
                # via last_person_ids so downstream consumers can correlate
                # without changing the existing signal contract.
                if result.num_persons > 0:
                    self.last_person_ids = [
                        int(p.person_id) for p in result.persons
                    ]
                    self.keypoints_3d_ready.emit([
                        p.keypoints_3d for p in result.persons
                    ])
                else:
                    self.last_person_ids = []

            except Exception as e:
                import traceback
                self.error.emit(f"CUDA error: {e}\n{traceback.format_exc()}")

    def _build_projection_params(self):
        """Precompute per-camera projection parameters for 3D→2D reprojection."""
        import numpy as np
        import cv2
        self._proj_params = {}
        for port, cam in self._cameras.items():
            R = cam.extrinsics.rotation       # 3x3
            t = cam.extrinsics.translation    # (3,)
            rvec, _ = cv2.Rodrigues(R)
            self._proj_params[port] = {
                "rvec": rvec,
                "tvec": t.reshape(3, 1),
                "K": cam.intrinsics.matrix,
                "dist": cam.intrinsics.distortion,
            }

    def _draw_reprojected(self, port, frame, result):
        """Draw reprojected 3D keypoints and skeleton onto a camera frame."""
        import cv2
        import numpy as np

        if result.num_persons == 0:
            return frame

        if not hasattr(self, "_proj_params"):
            self._build_projection_params()

        params = self._proj_params.get(port)
        if params is None:
            return frame

        vis = frame.copy()
        n_colors = len(self._PERSON_COLORS)

        for pi, person in enumerate(result.persons):
            color = self._PERSON_COLORS[pi % n_colors]
            kp_color = tuple(min(255, int(c * 1.3)) for c in color)

            # Collect valid 3D keypoints
            pts_3d = {}
            for k, kp in enumerate(person.keypoints_3d):
                if kp is not None:
                    pts_3d[k] = kp

            if not pts_3d:
                continue

            # Project all valid 3D points to 2D at once
            indices = sorted(pts_3d.keys())
            arr_3d = np.array([pts_3d[k] for k in indices], dtype=np.float64)
            pts_2d, _ = cv2.projectPoints(
                arr_3d, params["rvec"], params["tvec"],
                params["K"], params["dist"]
            )
            pts_2d = pts_2d.reshape(-1, 2)

            # Map keypoint index -> 2D pixel
            kp_2d = {}
            for idx, ki in enumerate(indices):
                kp_2d[ki] = (int(pts_2d[idx, 0]), int(pts_2d[idx, 1]))

            # Draw skeleton limbs
            for i, j in self._SKELETON:
                if i in kp_2d and j in kp_2d:
                    cv2.line(vis, kp_2d[i], kp_2d[j], color, 2, cv2.LINE_AA)

            # Draw keypoints
            for ki, pt in kp_2d.items():
                cv2.circle(vis, pt, 4, kp_color, -1, cv2.LINE_AA)

            # Person label
            if 0 in kp_2d:
                lx, ly = kp_2d[0]
                cv2.putText(vis, f"P{pi}", (lx, ly - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return vis

    def stop(self):
        self.running = False
        self._has_work.set()  # unblock wait
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None
