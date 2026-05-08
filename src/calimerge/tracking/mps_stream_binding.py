"""
Python ctypes binding for the macOS MPS streaming pose tracking pipeline.

Wraps pt_mps_stream_create / pt_mps_stream_process_frame / pt_mps_stream_destroy
from libcalimerge_mps.dylib, giving the Python GUI access to the Apple Silicon
CoreML pipeline (ANE + GPU) with the same API shape as
:mod:`calimerge.tracking.cuda_stream_binding`.

Usage::

    pipeline = MpsStreamPipeline(
        num_cameras=3, frame_width=640, frame_height=480,
        calibration_toml_path="...",
        yolo_model_path="models/coreml/yolo_v10s.mlpackage",
        vitpose_model_path="models/coreml/vitpose_synthpose.mlpackage",
    )
    result = pipeline.process_frame(frames, sync_index)
    pipeline.close()

Layout of the C structs is taken from ``src/mps_pipeline/pt_stream_mps.h``;
the matching shared constants come from ``src/pt_shared/pt_common.h``.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── Constants (must match pt_common.h) ──
PT_MAX_CAMERAS = 16
PT_MAX_TRACKS = 32
PT_NUM_KEYPOINTS = 52  # SynthPose
PT_OK = 0


# ── ctypes struct definitions (must match pt_stream_mps.h) ──


class _StreamFrame(ctypes.Structure):
    _fields_ = [
        ("pixels", ctypes.POINTER(ctypes.c_uint8)),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("stride", ctypes.c_int),
        ("port", ctypes.c_int),
    ]


class _StreamFrameSet(ctypes.Structure):
    _fields_ = [
        ("frames", _StreamFrame * PT_MAX_CAMERAS),
        ("num_frames", ctypes.c_int),
        ("sync_index", ctypes.c_uint64),
    ]


class _StreamPerson(ctypes.Structure):
    _fields_ = [
        ("person_id", ctypes.c_int),
        ("keypoints_3d", (ctypes.c_double * 3) * PT_NUM_KEYPOINTS),
        ("keypoints_valid", ctypes.c_int * PT_NUM_KEYPOINTS),
        ("com_3d", ctypes.c_double * 3),
        ("com_valid", ctypes.c_int),
        ("num_views", ctypes.c_int),
    ]


class _StreamResult(ctypes.Structure):
    _fields_ = [
        ("persons", _StreamPerson * PT_MAX_TRACKS),
        ("num_persons", ctypes.c_int),
        ("sync_index", ctypes.c_uint64),
        ("processing_time_ms", ctypes.c_double),
    ]


class _StreamStats(ctypes.Structure):
    """Mirrors PT_MPS_StreamStats from pt_stream_mps.h."""
    _fields_ = [
        ("coreml_yolo_ms", ctypes.c_double),
        ("coreml_vitpose_ms", ctypes.c_double),
        ("preprocess_ms", ctypes.c_double),
        ("matching_ms", ctypes.c_double),
        ("triangulation_ms", ctypes.c_double),
        ("tracking_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("frames_processed", ctypes.c_int),
    ]


# Log callback: void (*)(const char *message, void *user_data)
_LOG_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


class _StreamConfig(ctypes.Structure):
    """Must match PT_MPS_StreamConfig in pt_stream_mps.h exactly (field order matters)."""
    _fields_ = [
        # Paths come first (3 × char[512] = 1536 bytes)
        ("yolo_model_path", ctypes.c_char * 512),
        ("vitpose_model_path", ctypes.c_char * 512),
        ("calibration_toml_path", ctypes.c_char * 512),
        # Camera params
        ("num_cameras", ctypes.c_int),
        ("frame_width", ctypes.c_int),
        ("frame_height", ctypes.c_int),
        # Processing params
        ("max_persons", ctypes.c_int),
        ("person_confidence", ctypes.c_float),
        ("keypoint_confidence", ctypes.c_float),
        ("epipolar_threshold", ctypes.c_float),
        ("max_track_distance", ctypes.c_float),
        ("track_patience", ctypes.c_int),
        # Callbacks
        ("log_callback", _LOG_FUNC),
        ("callback_user_data", ctypes.c_void_p),
    ]


# ── Library loading ──

_lib = None
_lib_path: Path | None = None


def _find_mps_lib() -> Path | None:
    """Search for libcalimerge_mps.dylib in standard locations.

    Returns None on non-Darwin platforms or if the dylib has not been
    built yet.
    """
    if sys.platform != "darwin":
        return None

    lib_name = "libcalimerge_mps.dylib"

    module_dir = Path(__file__).parent          # tracking/
    repo_root = module_dir.parent.parent.parent  # repo (3 levels up: tracking→calimerge→src→repo)

    candidates = [
        repo_root / "build" / "mps" / lib_name,
        repo_root / "src" / "mps_pipeline" / lib_name,           # in-source build
        repo_root / "src" / "mps_pipeline" / "calimerge_mps.dylib",  # legacy name
        module_dir / lib_name,
    ]

    for p in candidates:
        if p.exists():
            return p
    return None


def _load_lib():
    global _lib, _lib_path
    if _lib is not None:
        return _lib

    path = _find_mps_lib()
    if path is None:
        raise FileNotFoundError(
            "calimerge_mps library not found. "
            "Build with: bash src/mps_pipeline/build_mps.sh release"
        )

    _lib = ctypes.CDLL(str(path))
    _lib_path = path

    # pt_mps_stream_create
    _lib.pt_mps_stream_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_StreamConfig),
    ]
    _lib.pt_mps_stream_create.restype = ctypes.c_int

    # pt_mps_stream_process_frame
    _lib.pt_mps_stream_process_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_StreamFrameSet),
        ctypes.POINTER(_StreamResult),
    ]
    _lib.pt_mps_stream_process_frame.restype = ctypes.c_int

    # pt_mps_stream_destroy
    _lib.pt_mps_stream_destroy.argtypes = [ctypes.c_void_p]
    _lib.pt_mps_stream_destroy.restype = None

    # pt_mps_stream_get_stats
    _lib.pt_mps_stream_get_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_StreamStats),
    ]
    _lib.pt_mps_stream_get_stats.restype = None

    # pt_mps_stream_reset_tracks
    _lib.pt_mps_stream_reset_tracks.argtypes = [ctypes.c_void_p]
    _lib.pt_mps_stream_reset_tracks.restype = None

    return _lib


def is_available() -> bool:
    """Return True if the MPS streaming pipeline dylib can be loaded.

    Always returns False on non-Darwin platforms (no .dylib to load) and on
    Macs where the dylib has not been built or any of the .mlpackages have
    not been materialised.
    """
    if sys.platform != "darwin":
        return False
    if _find_mps_lib() is None:
        return False
    try:
        _load_lib()
        return True
    except Exception:
        return False


# ── High-level API ──

@dataclass
class StreamPersonResult:
    """One tracked person from a single frame."""
    person_id: int
    keypoints_3d: list  # list of np.ndarray(3,) or None, length 52
    com_3d: np.ndarray | None  # (3,) or None
    num_views: int


@dataclass
class StreamFrameResult:
    """Result from processing one synchronized frame set."""
    persons: list[StreamPersonResult]
    num_persons: int
    sync_index: int
    processing_time_ms: float


@dataclass
class StreamStatsResult:
    coreml_yolo_ms: float
    coreml_vitpose_ms: float
    preprocess_ms: float
    matching_ms: float
    triangulation_ms: float
    tracking_ms: float
    total_ms: float
    frames_processed: int


class MpsStreamPipeline:
    """High-level wrapper for the MPS (CoreML) streaming pose pipeline.

    Mirrors :class:`calimerge.tracking.cuda_stream_binding.CudaStreamPipeline`
    so :class:`MpsStreamDetectionWorker` and the GUI selector can swap
    backends without changing call sites.

    Create once (CoreML compiles each .mlpackage on first use, then caches
    a .mlmodelc), then call :meth:`process_frame` per sync frame.
    """

    def __init__(
        self,
        num_cameras: int,
        frame_width: int,
        frame_height: int,
        calibration_toml_path: str,
        yolo_model_path: str = "",
        vitpose_model_path: str = "",
        max_persons: int = 2,
        person_confidence: float = 0.50,
        keypoint_confidence: float = 0.10,
        max_track_distance: float = 0.5,
        track_patience: int = 60,
        epipolar_threshold: float = 10.0,
        log_callback=None,
    ):
        lib = _load_lib()

        self._handle = ctypes.c_void_p(None)
        self._log_ref = None  # prevent GC of callback

        config = _StreamConfig()
        config.num_cameras = num_cameras
        config.frame_width = frame_width
        config.frame_height = frame_height
        config.max_persons = max_persons
        config.person_confidence = float(person_confidence)
        config.keypoint_confidence = float(keypoint_confidence)
        config.epipolar_threshold = float(epipolar_threshold)
        config.max_track_distance = float(max_track_distance)
        config.track_patience = int(track_patience)

        config.yolo_model_path = yolo_model_path.encode("utf-8")[:511]
        config.vitpose_model_path = vitpose_model_path.encode("utf-8")[:511]
        config.calibration_toml_path = calibration_toml_path.encode("utf-8")[:511]

        if log_callback is not None:
            def _log_trampoline(msg, _ud):
                try:
                    log_callback(msg.decode("utf-8", errors="replace"))
                except Exception:
                    pass
            self._log_ref = _LOG_FUNC(_log_trampoline)
            config.log_callback = self._log_ref
        else:
            config.log_callback = _LOG_FUNC(0)

        config.callback_user_data = None

        rc = lib.pt_mps_stream_create(ctypes.byref(self._handle), ctypes.byref(config))
        if rc != PT_OK:
            raise RuntimeError(f"pt_mps_stream_create failed with code {rc}")

        self._lib = lib

    def process_frame(
        self,
        frames: list[tuple[np.ndarray, int]],
        sync_index: int = 0,
    ) -> StreamFrameResult:
        """Process one synchronized frame set.

        Args:
            frames: list of (bgr_array, port) tuples.
            sync_index: monotonic frame counter.

        Returns:
            :class:`StreamFrameResult` with tracked persons.
        """
        frameset = _StreamFrameSet()
        frameset.num_frames = min(len(frames), PT_MAX_CAMERAS)
        frameset.sync_index = sync_index

        # Keep references to prevent GC during the C call
        _pixel_refs = []
        for i, (bgr, port) in enumerate(frames):
            if i >= PT_MAX_CAMERAS:
                break
            bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
            _pixel_refs.append(bgr)
            frameset.frames[i].pixels = bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            frameset.frames[i].width = bgr.shape[1]
            frameset.frames[i].height = bgr.shape[0]
            frameset.frames[i].stride = bgr.strides[0]
            frameset.frames[i].port = port

        result = _StreamResult()
        rc = self._lib.pt_mps_stream_process_frame(
            self._handle, ctypes.byref(frameset), ctypes.byref(result)
        )
        if rc != PT_OK:
            raise RuntimeError(f"pt_mps_stream_process_frame failed with code {rc}")

        persons = []
        for i in range(result.num_persons):
            p = result.persons[i]
            kps = []
            for k in range(PT_NUM_KEYPOINTS):
                if p.keypoints_valid[k]:
                    kps.append(np.array([
                        p.keypoints_3d[k][0],
                        p.keypoints_3d[k][1],
                        p.keypoints_3d[k][2],
                    ], dtype=np.float64))
                else:
                    kps.append(None)

            com = None
            if p.com_valid:
                com = np.array([p.com_3d[0], p.com_3d[1], p.com_3d[2]], dtype=np.float64)

            persons.append(StreamPersonResult(
                person_id=p.person_id,
                keypoints_3d=kps,
                com_3d=com,
                num_views=p.num_views,
            ))

        return StreamFrameResult(
            persons=persons,
            num_persons=result.num_persons,
            sync_index=int(result.sync_index),
            processing_time_ms=result.processing_time_ms,
        )

    def get_stats(self) -> StreamStatsResult:
        stats = _StreamStats()
        self._lib.pt_mps_stream_get_stats(self._handle, ctypes.byref(stats))
        return StreamStatsResult(
            coreml_yolo_ms=stats.coreml_yolo_ms,
            coreml_vitpose_ms=stats.coreml_vitpose_ms,
            preprocess_ms=stats.preprocess_ms,
            matching_ms=stats.matching_ms,
            triangulation_ms=stats.triangulation_ms,
            tracking_ms=stats.tracking_ms,
            total_ms=stats.total_ms,
            frames_processed=stats.frames_processed,
        )

    def reset_tracks(self):
        self._lib.pt_mps_stream_reset_tracks(self._handle)

    def close(self):
        if self._handle and self._handle.value:
            self._lib.pt_mps_stream_destroy(self._handle)
            self._handle = ctypes.c_void_p(None)

    def __del__(self):
        try:
            self.close()
        except AttributeError:
            pass
