"""
Unified offline post-tracking worker.

Drives the *exact same* per-sync inference + tracking primitive that the
live workers use, against decoded video frames instead of camera ring
buffer frames. The only differences between live and offline are:

  * Frame source: ``cv2.VideoCapture`` per port instead of the live camera
    binding.
  * Batch size: a Python-side knob exposed for future C-side multi-sync
    batching. With today's streaming primitive each sync is processed
    individually; the knob is plumbed through and used by the PyTorch
    backend's per-sync ``_detect_and_draw_batch`` call. CUDA / MPS still
    process one sync at a time until the C-side batch entry lands — see
    ``TODO.md``.

Why a unified path?

The previous offline worker (``OfflineProcessingWorker`` →
``run_cuda_pipeline`` / ``run_mps_pipeline`` → ``pt_main.cpp``) was a
*separate* implementation that batched multiple sync indices together for
throughput. That separation was the source of multiple bugs:

  * Different fragmentation behaviour (the C tracker's view-set match
    spawns fresh track ids, requiring a Python-side stitching pass that
    the live path doesn't need because the live tracker handles it
    differently).
  * Different person-confidence default (``run_cuda_pipeline`` defaulted
    to 0.1, low enough that mid-trial false positives snapped onto static
    objects in the room).
  * Different tracker config (``max_track_distance`` and
    ``track_patience`` were tighter offline, fragmenting one subject
    into 13 short tracks).

The unified path eliminates these by construction: same primitive, same
defaults, same tracker.

Trade-off
---------
Calling the streaming primitive in a Python loop is correct but loses the
batched-TRT-enqueue throughput of the deprecated path. A true multi-sync
``process_batch(frame_lists[B], sync_indices[B])`` on the C/Obj-C++ side
would recover that. Phase 2 — see ``TODO.md``. Today's win is bug-fix /
code-share, not raw throughput.

On-disk schema
--------------
Identical to ``OfflineProcessingWorker``: ``keypoints_3d.raw.npz`` (via
``write_raw_buffer``) + ``keypoints_3d.npz`` (via ``save_keypoints_3d``)
under the session dir, with ``view_rotation`` / ``view_translation``
applied. Downstream consumers (notebook, workout_playback,
csv_export_worker) are platform-blind.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PySide6.QtCore import QThread, Signal


if TYPE_CHECKING:
    import numpy as np
    from ..types import CalibratedCamera


# Default keypoint count when the backend never emits a person. SynthPose-52.
_DEFAULT_NUM_KEYPOINTS = 52


class UnifiedOfflineWorker(QThread):
    """Drive the live per-sync detection primitive over recorded videos.

    Same Qt signals as :class:`OfflineProcessingWorker` so the workout-page
    wiring is interchangeable.

    Backend dispatch
    ----------------
    * ``"pytorch"`` — instantiate :class:`PoseDetectionWorker` standalone
      (no QThread start), drive ``_detect_and_draw_batch`` + ``_triangulate_live``
      directly per sync. The 3D keypoints come back over the worker's
      ``keypoints_3d_ready`` signal; we connect a local collector before the
      call so emit() invokes our slot directly (no event loop required when
      emitter and slot share a thread).
    * ``"cuda"`` — instantiate :class:`CudaStreamPipeline` once (slow first
      call, builds TRT engines), then loop ``process_frame``.
    * ``"mps"`` — instantiate :class:`MpsStreamPipeline` once, then loop
      ``process_frame``.
    """

    progress = Signal(str, float)
    log_message = Signal(str)
    finished_ok = Signal(object)   # Path to session dir
    failed = Signal(str)

    def __init__(
        self,
        session_dir: "Path",
        cameras: "dict[int, CalibratedCamera]",
        port_to_video: "dict[int, Path]",
        frame_time_csv: "Path",
        backend: Literal["pytorch", "cuda", "mps"],
        view_rotation: "np.ndarray | None" = None,
        view_translation: "np.ndarray | None" = None,
        max_track_distance: float = 0.5,
        track_patience: int = 60,
        stitch_max_gap_frames: int = 90,
        stitch_max_distance_m: float = 0.6,
        # Today: PyTorch path drives ``_detect_and_draw_batch`` with this
        # many ports per call (i.e. always equal to camera count today,
        # since one sync == one port-batch). CUDA / MPS will start using
        # this when the C-side multi-sync batch entry lands. See TODO.md.
        batch_size: int = 8,
        # Match the live PyTorch slider default, NOT run_cuda_pipeline's
        # 0.1 — the latter caused mid-trial false-positive snaps onto
        # static objects (~t=11s in the head-turn trial that surfaced
        # the bug).
        person_confidence: float = 0.5,
    ):
        super().__init__()
        self._session_dir = session_dir
        self._cameras = cameras
        self._port_to_video = port_to_video
        self._frame_time_csv = frame_time_csv
        self._backend = backend
        self._view_rotation = view_rotation
        self._view_translation = view_translation
        self._max_track_distance = float(max_track_distance)
        self._track_patience = int(track_patience)
        self._stitch_max_gap_frames = int(stitch_max_gap_frames)
        self._stitch_max_distance_m = float(stitch_max_distance_m)
        self._batch_size = int(batch_size)
        self._person_confidence = float(person_confidence)

        # Cached on first use to avoid rebuilding per sync. Built off the
        # frame_time_csv in run().
        self._sync_to_ports: "dict[int, list[int]] | None" = None

        # Allow stop() to interrupt long runs from the GUI.
        self._cancel = False

    def stop(self) -> None:
        """Request a clean cancel. Checked between syncs."""
        self._cancel = True

    # ── Public Qt entrypoint ─────────────────────────────────────────────

    def run(self):
        try:
            self.progress.emit("starting", 0.0)
            self.log_message.emit(
                f"[unified-offline] backend={self._backend} "
                f"batch_size={self._batch_size} "
                f"person_confidence={self._person_confidence:.2f}"
            )

            sync_to_ports = self._read_sync_to_ports()
            if not sync_to_ports:
                raise RuntimeError(
                    f"frame_time_history.csv produced no sync indices: "
                    f"{self._frame_time_csv}"
                )
            self._sync_to_ports = sync_to_ports

            # Per-port video readers: open once, seek frame-by-frame.
            # cv2.VideoCapture.read() advances by one; we honour the
            # frame_index from frame_time_history when available.
            import cv2  # local — keeps module import cheap for tests
            captures: dict[int, "cv2.VideoCapture"] = {}
            for port, video_path in self._port_to_video.items():
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    self.log_message.emit(
                        f"[unified-offline] could not open {video_path}"
                    )
                    continue
                captures[port] = cap

            if not captures:
                raise RuntimeError("no video files could be opened")

            calibrated_ports = sorted(
                p for p in captures.keys() if p in self._cameras
            )
            self.log_message.emit(
                f"[unified-offline] calibrated ports={calibrated_ports}"
            )

            # Backend dispatch — each helper returns a list of per-sync
            # records, sorted by sync index. The records use the same
            # schema the live path emits to its in-memory recording
            # buffer: dict(time, persons, primary_index).
            if self._backend == "pytorch":
                recording, n_kps_global = self._run_pytorch(
                    captures, calibrated_ports, sync_to_ports,
                )
            elif self._backend == "cuda":
                recording, n_kps_global = self._run_cuda_stream(
                    captures, calibrated_ports, sync_to_ports,
                )
            elif self._backend == "mps":
                recording, n_kps_global = self._run_mps_stream(
                    captures, calibrated_ports, sync_to_ports,
                )
            else:
                raise ValueError(
                    f"unknown backend {self._backend!r}; expected one of "
                    "'pytorch', 'cuda', 'mps'"
                )

            # Always release captures — leaving them open holds Windows
            # file locks long enough to prevent a follow-up scrub.
            for cap in captures.values():
                try:
                    cap.release()
                except Exception:
                    pass

            if self._cancel:
                self.log_message.emit("[unified-offline] cancelled")
                return

            # ── Canonical re-tracking ─────────────────────────────────
            # Each backend's underlying tracker has different defaults
            # and different fragmentation behavior (PyTorch _LiveTracker
            # vs C pt_tracker). Throw away their track ids and re-run a
            # single canonical _LiveTracker over every recording so the
            # downstream stitching/output is identical regardless of the
            # inference backend. Inputs (kps_3d) still differ, but the
            # tracker code path doesn't.
            self.progress.emit("re-tracking", 0.90)
            self._retrack_recording(recording)

            # ── Stitch tracks across the recording ────────────────────
            self.progress.emit("stitching tracks", 0.92)
            tracks_by_id, n_kps_global = self._collect_tracks(
                recording, n_kps_global,
            )
            from ..tracking.track_stitch import stitch_tracks
            n_before = len(tracks_by_id)
            tracks_by_id = stitch_tracks(
                tracks_by_id,
                max_gap_frames=self._stitch_max_gap_frames,
                max_distance_m=self._stitch_max_distance_m,
            )
            if n_before != len(tracks_by_id):
                self.log_message.emit(
                    f"[unified-offline] stitched {n_before} tracks -> "
                    f"{len(tracks_by_id)} "
                    f"(gap<={self._stitch_max_gap_frames}f, "
                    f"dist<={self._stitch_max_distance_m:.2f}m)"
                )

            # Re-emit the recording from the stitched tracks so the
            # output ordering is canonical (lowest survivor id first).
            recording = self._tracks_to_recording(tracks_by_id, recording)

            # ── Write outputs ─────────────────────────────────────────
            self.progress.emit("writing outputs", 0.97)
            self._write_outputs(recording, n_kps_global)

            self.progress.emit("complete", 1.0)
            self.finished_ok.emit(self._session_dir)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()}")

    # ── Backend implementations ──────────────────────────────────────────

    def _run_pytorch(
        self,
        captures: "dict",
        calibrated_ports: list[int],
        sync_to_ports: "dict[int, list[int]]",
    ) -> "tuple[list[dict], int]":
        """Drive PoseDetectionWorker's per-sync primitives directly.

        We instantiate the worker but do NOT start its QThread. That keeps
        the run loop out of the picture — we own the iteration. The
        worker exposes ``_detect_and_draw_batch(work)`` which populates
        ``_last_kps_per_port``, and ``_triangulate_live()`` which emits
        ``keypoints_3d_ready`` with the per-frame persons. Connecting
        that signal to a local list before the call works without an
        event loop because ``Signal.emit`` invokes slots directly when
        the emitter and slot are on the same thread.
        """
        from .workers import PoseDetectionWorker
        from ..tracking.pose_detector import setup_device, load_models

        # Build the worker WITHOUT starting it. Then run the model
        # init that run() normally does. We never enter run()'s loop.
        worker = PoseDetectionWorker(device_name="auto", cameras=self._cameras)
        worker.confidence_threshold = self._person_confidence

        device = setup_device("auto")
        worker._device = device
        self.log_message.emit(
            f"[unified-offline] loading PyTorch models on {device}..."
        )
        person_model, pose_processor, pose_model = load_models(
            device=device,
            log_fn=lambda msg: self.log_message.emit(f"[unified-offline] {msg}"),
        )
        worker._models = (person_model, pose_processor, pose_model)
        # Replace the default-tuned _LiveTracker so the unified worker's
        # max_track_distance / track_patience actually take effect — the
        # constructor at workers.py:752 uses _LiveTracker() with library
        # defaults (0.5 m / 10 frames) which silently override anything
        # the GUI passes through this worker.
        from .workers import _LiveTracker as _LT
        worker._tracker = _LT(
            max_match_distance=self._max_track_distance,
            patience=self._track_patience,
        )

        # Connect the keypoints_3d_ready signal to a local collector.
        collected: list = []

        def _collect(persons: list) -> None:
            collected.append(list(persons))

        worker.keypoints_3d_ready.connect(_collect)

        sync_indices = sorted(sync_to_ports.keys())
        n_syncs = len(sync_indices)
        recording: list[dict] = []
        n_kps_global = 0
        fps_est = self._fps_estimate()

        # Decode frames in lockstep across captures. cv2.VideoCapture
        # only supports forward streaming reliably (random seek is slow
        # and inaccurate on some codecs), so we step by 1 per sync and
        # keep a per-port pending frame index.
        port_next_frame: dict[int, int] = {p: 0 for p in calibrated_ports}
        port_buf: dict[int, "object"] = {}

        import numpy as np  # local
        for i, sync in enumerate(sync_indices):
            if self._cancel:
                break

            wanted_ports = [p for p in sync_to_ports[sync] if p in calibrated_ports]
            work: dict[int, np.ndarray] = {}

            for port in wanted_ports:
                # Advance the capture until its next frame matches the
                # frame-index we want. With a contiguous video that is
                # simply read() once per sync; if a sync was missing in
                # the source we drop the sync entirely (None entry).
                cap = captures[port]
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                port_next_frame[port] += 1
                port_buf[port] = frame
                work[port] = frame

            if not work:
                continue

            # ── Feed the live primitives directly. ──
            # Snapshot what was emitted before, run, and capture
            # whatever new entry the keypoints_3d_ready signal added.
            n_before = len(collected)
            try:
                worker._detect_and_draw_batch(work)
            except Exception as e:
                self.log_message.emit(
                    f"[unified-offline] _detect_and_draw_batch raised: {e}"
                )
                continue

            # Triangulate iff we have ≥2 cameras with detections.
            if len(worker._last_kps_per_port) >= 2:
                worker._triangulate_live()

            persons = collected[-1] if len(collected) > n_before else []

            # Track the global max keypoint dimension we saw.
            for kps in persons:
                if kps is not None:
                    n_kps_global = max(n_kps_global, len(kps))

            recording.append({
                "time": sync / fps_est,
                "persons": persons,
                "primary_index": 0,
                "_sync_index": sync,
                "_person_ids": list(getattr(worker, "last_person_ids", [])),
            })

            if n_syncs and (i % max(1, n_syncs // 50) == 0):
                # ~50 progress ticks across the trial.
                self.progress.emit(
                    "detect+triangulate",
                    0.05 + 0.85 * (i / max(1, n_syncs)),
                )

        # Disconnect for cleanliness — defends against repeated runs in
        # the same process.
        try:
            worker.keypoints_3d_ready.disconnect(_collect)
        except Exception:
            pass

        if n_kps_global == 0:
            n_kps_global = _DEFAULT_NUM_KEYPOINTS

        return recording, n_kps_global

    def _run_cuda_stream(
        self,
        captures: "dict",
        calibrated_ports: list[int],
        sync_to_ports: "dict[int, list[int]]",
    ) -> "tuple[list[dict], int]":
        """Loop CudaStreamPipeline.process_frame over decoded videos."""
        from ..tracking.cuda_stream_binding import CudaStreamPipeline
        from ..config import (
            engine_cache_dir,
            models_dir,
            write_cuda_calibration_toml,
        )
        import tempfile
        import numpy as np

        if not calibrated_ports:
            raise RuntimeError("no calibrated ports for CUDA pipeline")

        first_port = calibrated_ports[0]
        first_cam = self._cameras[first_port]
        w, h = first_cam.intrinsics.resolution

        cal_path = Path(tempfile.gettempdir()) / "calimerge_unified_offline_cal.toml"
        # Filter cameras to only calibrated ports we actually have video for.
        cal_subset = {p: self._cameras[p] for p in calibrated_ports}
        write_cuda_calibration_toml(cal_subset, cal_path)

        onnx_dir = models_dir() / "onnx"
        yolo_onnx = onnx_dir / "yolo_v10s.onnx"
        vitpose_onnx = onnx_dir / "vitpose_synthpose.onnx"
        cache = engine_cache_dir()
        cache.mkdir(parents=True, exist_ok=True)

        self.log_message.emit("[unified-offline] initializing CUDA pipeline...")
        pipeline = CudaStreamPipeline(
            num_cameras=len(calibrated_ports),
            frame_width=w,
            frame_height=h,
            calibration_toml_path=str(cal_path),
            yolo_onnx_path=str(yolo_onnx) if yolo_onnx.exists() else "",
            vitpose_onnx_path=str(vitpose_onnx) if vitpose_onnx.exists() else "",
            engine_cache_dir=str(cache),
            max_persons=2,
            person_confidence=self._person_confidence,
            max_track_distance=self._max_track_distance,
            track_patience=self._track_patience,
            log_callback=lambda m: self.log_message.emit(f"[unified-offline][cuda] {m}"),
        )
        try:
            return self._loop_streaming_pipeline(
                pipeline, captures, calibrated_ports, sync_to_ports, w, h,
            )
        finally:
            try:
                pipeline.close()
            except Exception:
                pass

    def _run_mps_stream(
        self,
        captures: "dict",
        calibrated_ports: list[int],
        sync_to_ports: "dict[int, list[int]]",
    ) -> "tuple[list[dict], int]":
        """Loop MpsStreamPipeline.process_frame over decoded videos."""
        from ..tracking.mps_stream_binding import MpsStreamPipeline
        from ..config import models_dir, write_cuda_calibration_toml
        import tempfile

        if not calibrated_ports:
            raise RuntimeError("no calibrated ports for MPS pipeline")

        first_port = calibrated_ports[0]
        first_cam = self._cameras[first_port]
        w, h = first_cam.intrinsics.resolution

        cal_path = Path(tempfile.gettempdir()) / "calimerge_unified_offline_cal.toml"
        cal_subset = {p: self._cameras[p] for p in calibrated_ports}
        write_cuda_calibration_toml(cal_subset, cal_path)

        coreml_dir = models_dir() / "coreml"
        yolo_pkg = coreml_dir / "yolo_v10s.mlpackage"
        vitpose_rt = coreml_dir / "vitpose_synthpose_rt_batch4.mlpackage"
        vitpose_pkg = vitpose_rt if vitpose_rt.exists() else (
            coreml_dir / "vitpose_synthpose.mlpackage"
        )
        self.log_message.emit(
            f"[unified-offline] vitpose model: {vitpose_pkg.name}"
        )

        self.log_message.emit("[unified-offline] initializing MPS pipeline...")
        pipeline = MpsStreamPipeline(
            num_cameras=len(calibrated_ports),
            frame_width=w,
            frame_height=h,
            calibration_toml_path=str(cal_path),
            yolo_model_path=str(yolo_pkg) if yolo_pkg.exists() else "",
            vitpose_model_path=str(vitpose_pkg) if vitpose_pkg.exists() else "",
            max_persons=2,
            person_confidence=self._person_confidence,
            max_track_distance=self._max_track_distance,
            track_patience=self._track_patience,
        )
        try:
            return self._loop_streaming_pipeline(
                pipeline, captures, calibrated_ports, sync_to_ports, w, h,
            )
        finally:
            try:
                pipeline.close()
            except Exception:
                pass

    def _loop_streaming_pipeline(
        self,
        pipeline,
        captures: "dict",
        calibrated_ports: list[int],
        sync_to_ports: "dict[int, list[int]]",
        w: int,
        h: int,
    ) -> "tuple[list[dict], int]":
        """Common loop body for the CUDA + MPS streaming pipelines."""
        import cv2
        import numpy as np

        sync_indices = sorted(sync_to_ports.keys())
        n_syncs = len(sync_indices)
        fps_est = self._fps_estimate()
        recording: list[dict] = []
        n_kps_global = 0

        for i, sync in enumerate(sync_indices):
            if self._cancel:
                break

            wanted_ports = [
                p for p in sync_to_ports[sync] if p in calibrated_ports
            ]

            frame_list: list[tuple[np.ndarray, int]] = []
            for port in wanted_ports:
                ok, frame = captures[port].read()
                if not ok or frame is None:
                    continue
                # The C side expects every frame at (w, h); resize if
                # the recording resolution diverges (rare but possible
                # if a downstream codec rescaled).
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))
                frame_list.append((frame, port))

            if len(frame_list) < 2:
                # At least 2 cameras required for triangulation.
                recording.append({
                    "time": sync / fps_est,
                    "persons": [],
                    "primary_index": 0,
                    "_sync_index": sync,
                    "_person_ids": [],
                })
                continue

            try:
                result = pipeline.process_frame(frame_list, sync_index=sync)
            except Exception as e:
                self.log_message.emit(
                    f"[unified-offline] process_frame(sync={sync}) raised: {e}"
                )
                continue

            persons = [p.keypoints_3d for p in result.persons]
            person_ids = [int(p.person_id) for p in result.persons]
            for kps in persons:
                if kps is not None:
                    n_kps_global = max(n_kps_global, len(kps))

            recording.append({
                "time": sync / fps_est,
                "persons": persons,
                "primary_index": 0,
                "_sync_index": sync,
                "_person_ids": person_ids,
            })

            if n_syncs and (i % max(1, n_syncs // 50) == 0):
                self.progress.emit(
                    "detect+triangulate",
                    0.05 + 0.85 * (i / max(1, n_syncs)),
                )

        if n_kps_global == 0:
            n_kps_global = _DEFAULT_NUM_KEYPOINTS

        return recording, n_kps_global

    # ── Helpers ──────────────────────────────────────────────────────────

    def _read_sync_to_ports(self) -> "dict[int, list[int]]":
        """Parse frame_time_history.csv → {sync_index: [ports]} ordering."""
        import csv as _csv
        out: dict[int, list[int]] = {}
        with open(self._frame_time_csv, "r", newline="") as f:
            reader = _csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if row[0] == "sync_index":
                    continue
                try:
                    sync = int(row[0])
                    port = int(row[1])
                except (ValueError, IndexError):
                    continue
                out.setdefault(sync, []).append(port)
        return out

    def _fps_estimate(self) -> float:
        """Estimate fps from the frame_time_history.csv timestamps."""
        try:
            import csv as _csv
            with open(self._frame_time_csv, "r", newline="") as ft:
                ftr = _csv.reader(ft)
                next(ftr, None)
                rows = list(ftr)
                if len(rows) > 1:
                    times = [float(r[3]) for r in rows if len(r) > 3]
                    if len(times) > 1:
                        dt = (
                            (max(times) - min(times))
                            / max(1, (len(times) - 1))
                        )
                        if dt > 0:
                            return 1.0 / dt
        except Exception:
            pass
        return 30.0

    def _retrack_recording(self, recording: list[dict]) -> None:
        """Replace each backend's per-frame track ids with canonical ones.

        Builds a fresh ``_LiveTracker`` parameterised by this worker's
        ``max_track_distance`` / ``track_patience``, runs it across every
        frame in order, and overwrites ``entry["_person_ids"]`` with its
        output. After this pass, downstream code (``_collect_tracks``,
        ``stitch_tracks``) sees the same id-assignment policy regardless
        of which backend produced the keypoints.
        """
        from .workers import _LiveTracker
        tracker = _LiveTracker(
            max_match_distance=self._max_track_distance,
            patience=self._track_patience,
        )
        n_frames_with_persons = 0
        n_persons_total = 0
        n_ids_assigned_nonzero = 0
        for entry in recording:
            persons = entry.get("persons", []) or []
            if persons:
                n_frames_with_persons += 1
                n_persons_total += len(persons)
            new_ids = tracker.step(persons)
            n_ids_assigned_nonzero += sum(1 for i in new_ids if i != 0)
            entry["_person_ids"] = list(new_ids)
        self.log_message.emit(
            f"[unified-offline] retrack: frames_with_persons={n_frames_with_persons} "
            f"persons={n_persons_total} ids_nonzero={n_ids_assigned_nonzero}"
        )

    def _collect_tracks(
        self,
        recording: list[dict],
        n_kps_global: int,
    ) -> "tuple[dict[int, dict[int, list]], int]":
        """Group the recording into ``{track_id: {sync: kps}}`` for stitching.

        Track ids come from ``_person_ids`` if the streaming pipeline
        emitted them (CUDA / MPS path), or from the in-frame index if it
        didn't (PyTorch path uses _LiveTracker which already does
        cross-frame association — but we still run stitch_tracks on top
        in case _LiveTracker's patience window dropped a continuation).
        """
        tracks: dict[int, dict[int, list]] = {}
        for entry in recording:
            sync = entry.get("_sync_index", 0)
            persons = entry.get("persons", []) or []
            ids = entry.get("_person_ids", []) or []
            for j, kps in enumerate(persons):
                if kps is None:
                    continue
                # Use whatever id the underlying tracker assigned. Falling
                # back to position is a last-resort: it generates a fresh
                # id per (sync, slot) which would break stitching.
                if j < len(ids) and ids[j] != 0:
                    tid = ids[j]
                else:
                    tid = -(j + 1)  # negative pseudo-id; still merges by COM
                tracks.setdefault(tid, {})[sync] = kps
                n_kps_global = max(n_kps_global, len(kps))
        return tracks, n_kps_global

    def _tracks_to_recording(
        self,
        tracks_by_id: "dict[int, dict[int, list]]",
        original_recording: list[dict],
    ) -> list[dict]:
        """Re-emit the recording buffer from stitched tracks.

        Stable pid order so person 0 is the lowest-id (= first-seen)
        survivor — matches what the live path's ordering convention
        feeds to ``save_keypoints_3d``.
        """
        sorted_tids = sorted(tracks_by_id.keys())
        tid_to_slot = {tid: i for i, tid in enumerate(sorted_tids)}

        # Preserve the original (sync, time) pairs so timestamps stay
        # consistent. Rows with no surviving track produce empty persons.
        sync_to_time: dict[int, float] = {}
        for entry in original_recording:
            sync = entry.get("_sync_index", 0)
            sync_to_time[sync] = entry.get("time", 0.0)

        all_syncs = set(sync_to_time.keys())
        for sync_map in tracks_by_id.values():
            all_syncs.update(sync_map.keys())
        sorted_syncs = sorted(all_syncs)

        recording: list[dict] = []
        for sync in sorted_syncs:
            persons: list = [None] * len(sorted_tids)
            for tid, sync_map in tracks_by_id.items():
                kps = sync_map.get(sync)
                if kps is not None:
                    persons[tid_to_slot[tid]] = kps
            while persons and persons[-1] is None:
                persons.pop()
            recording.append({
                "time": sync_to_time.get(sync, sync / self._fps_estimate()),
                "persons": persons,
                "primary_index": 0,
            })
        return recording

    def _write_outputs(
        self,
        recording: list[dict],
        n_kps_global: int,
    ) -> None:
        """Write keypoints_3d.raw.npz + keypoints_3d.npz.

        Schema must match what ``OfflineProcessingWorker._convert_outputs``
        + the live path produce. Downstream consumers
        (notebook, workout_playback, csv_export_worker) are platform-blind.
        """
        from ..keypoint_export import write_raw_buffer
        from ..analysis.keypoints_io import save_keypoints_3d

        if not recording:
            self.log_message.emit("[unified-offline] no frames produced")
            return

        # The unified worker carries a backend choice (pytorch/cuda/mps);
        # record it in the npz so consumers can tell which pipeline
        # produced these keypoints. Same SynthPose model across all three
        # backends today; expose it as a constant string for now.
        # We ship the SynthPose-trained VitPose weights (52 anatomical
        # keypoints), so the canonical model_name is "synthpose" — same
        # string used as model_key in view_transforms.db.
        _model_name = "synthpose"

        raw_path = self._session_dir / "keypoints_3d.raw.npz"
        write_raw_buffer(
            raw_path,
            recording,
            view_rotation=self._view_rotation,
            view_translation=self._view_translation,
            model_backend=self._backend,
            model_name=_model_name,
        )

        npz_path = self._session_dir / "keypoints_3d.npz"
        try:
            save_keypoints_3d(
                npz_path,
                recording,
                num_keypoints=max(1, n_kps_global),
                view_rotation=self._view_rotation,
                view_translation=self._view_translation,
                model_backend=self._backend,
                model_name=_model_name,
                person_confidence=self._person_confidence,
                max_track_distance=self._max_track_distance,
                track_patience=self._track_patience,
            )
            self.log_message.emit(
                f"[unified-offline] wrote {raw_path.name} + {npz_path.name} "
                f"({len(recording)} frames)"
            )
        except Exception as e:
            self.log_message.emit(
                f"[unified-offline] wrote {raw_path.name} only; "
                f"{npz_path.name} failed: {e}"
            )
