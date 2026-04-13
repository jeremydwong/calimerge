"""
Python ctypes binding for the CUDA streaming pose tracking pipeline.

Wraps pt_stream_create / pt_stream_process_frame / pt_stream_destroy
from calimerge_cuda.dll, giving the Python GUI access to the same
~10ms/frame TensorRT pipeline used by pt_stream_main.exe.

Usage:
    pipeline = CudaStreamPipeline(config)
    result = pipeline.process_frame(frames, ports, sync_index)
    # result.persons is a list of dicts with person_id, keypoints_3d, com_3d
    pipeline.close()
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── Constants (must match pt_common.h) ──
PT_MAX_CAMERAS = 16
PT_MAX_TRACKS = 8
PT_NUM_KEYPOINTS = 17  # COCO (must match pt_common.h)
PT_OK = 0


# ── ctypes struct definitions ──

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
    _fields_ = [
        ("upload_ms", ctypes.c_double),
        ("yolo_ms", ctypes.c_double),
        ("vitpose_ms", ctypes.c_double),
        ("matching_ms", ctypes.c_double),
        ("triangulation_ms", ctypes.c_double),
        ("tracking_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("frames_processed", ctypes.c_int),
    ]


# Log callback: void (*)(const char *message, void *user_data)
_LOG_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


class _StreamConfig(ctypes.Structure):
    _fields_ = [
        ("num_cameras", ctypes.c_int),
        ("frame_width", ctypes.c_int),
        ("frame_height", ctypes.c_int),
        ("max_persons", ctypes.c_int),
        ("person_confidence", ctypes.c_float),
        ("keypoint_confidence", ctypes.c_float),
        ("epipolar_threshold", ctypes.c_float),
        ("max_track_distance", ctypes.c_float),
        ("track_patience", ctypes.c_int),
        ("use_fp16_yolo", ctypes.c_int),
        ("yolo_onnx_path", ctypes.c_char * 512),
        ("vitpose_onnx_path", ctypes.c_char * 512),
        ("engine_cache_dir", ctypes.c_char * 512),
        ("calibration_toml_path", ctypes.c_char * 512),
        ("log_callback", _LOG_FUNC),
        ("callback_user_data", ctypes.c_void_p),
    ]


# ── Library loading ──

_lib = None
_lib_path: Path | None = None


def _find_cuda_lib() -> Path | None:
    """Search for calimerge_cuda.dll/.so in standard locations."""
    if sys.platform == "win32":
        lib_name = "calimerge_cuda.dll"
    elif sys.platform == "darwin":
        lib_name = "libcalimerge_cuda.dylib"
    else:
        lib_name = "libcalimerge_cuda.so"

    module_dir = Path(__file__).parent          # tracking/
    repo_root = module_dir.parent.parent.parent  # calimerge/ (3 levels: tracking→calimerge→src→repo)

    candidates = [
        repo_root / "build" / "cuda" / lib_name,
        repo_root / "src" / "cuda_pipeline" / lib_name,  # legacy
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

    path = _find_cuda_lib()
    if path is None:
        raise FileNotFoundError(
            "calimerge_cuda library not found. "
            "Build with: src\\cuda_pipeline\\build_cuda_win32.bat release"
        )

    _lib = ctypes.CDLL(str(path))
    _lib_path = path

    # pt_stream_create
    _lib.pt_stream_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_StreamConfig),
    ]
    _lib.pt_stream_create.restype = ctypes.c_int

    # pt_stream_process_frame
    _lib.pt_stream_process_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_StreamFrameSet),
        ctypes.POINTER(_StreamResult),
    ]
    _lib.pt_stream_process_frame.restype = ctypes.c_int

    # pt_stream_destroy
    _lib.pt_stream_destroy.argtypes = [ctypes.c_void_p]
    _lib.pt_stream_destroy.restype = None

    # pt_stream_get_stats
    _lib.pt_stream_get_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_StreamStats),
    ]
    _lib.pt_stream_get_stats.restype = None

    # pt_stream_reset_tracks
    _lib.pt_stream_reset_tracks.argtypes = [ctypes.c_void_p]
    _lib.pt_stream_reset_tracks.restype = None

    return _lib


def is_available() -> bool:
    """Return True if the CUDA streaming pipeline DLL is found."""
    return _find_cuda_lib() is not None


# ── High-level API ──

@dataclass
class StreamPersonResult:
    """One tracked person from a single frame."""
    person_id: int
    keypoints_3d: list  # list of np.ndarray(3,) or None, length 17
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
    upload_ms: float
    yolo_ms: float
    vitpose_ms: float
    matching_ms: float
    triangulation_ms: float
    tracking_ms: float
    total_ms: float
    frames_processed: int


class CudaStreamPipeline:
    """High-level wrapper for the CUDA streaming pose pipeline.

    Create once (expensive — builds TensorRT engines on first run),
    then call process_frame() per sync frame (~10ms).
    """

    def __init__(
        self,
        num_cameras: int,
        frame_width: int,
        frame_height: int,
        calibration_toml_path: str,
        yolo_onnx_path: str = "",
        vitpose_onnx_path: str = "",
        engine_cache_dir: str = "",
        max_persons: int = 2,
        use_fp16_yolo: bool = True,
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
        config.person_confidence = 0.1
        config.keypoint_confidence = 0.1
        config.epipolar_threshold = 10.0
        config.max_track_distance = 0.15
        config.track_patience = 30
        config.use_fp16_yolo = 1 if use_fp16_yolo else 0

        config.yolo_onnx_path = yolo_onnx_path.encode("utf-8")[:511]
        config.vitpose_onnx_path = vitpose_onnx_path.encode("utf-8")[:511]
        config.engine_cache_dir = engine_cache_dir.encode("utf-8")[:511]
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

        rc = lib.pt_stream_create(ctypes.byref(self._handle), ctypes.byref(config))
        if rc != PT_OK:
            raise RuntimeError(f"pt_stream_create failed with code {rc}")

        self._lib = lib

    def process_frame(
        self,
        frames: list[tuple[np.ndarray, int]],
        sync_index: int = 0,
    ) -> StreamFrameResult:
        """Process one synchronized frame set.

        Args:
            frames: list of (bgr_array, port) tuples
            sync_index: monotonic frame counter

        Returns:
            StreamFrameResult with tracked persons
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
        rc = self._lib.pt_stream_process_frame(
            self._handle, ctypes.byref(frameset), ctypes.byref(result)
        )
        if rc != PT_OK:
            raise RuntimeError(f"pt_stream_process_frame failed with code {rc}")

        # Convert C result to Python
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
        self._lib.pt_stream_get_stats(self._handle, ctypes.byref(stats))
        return StreamStatsResult(
            upload_ms=stats.upload_ms,
            yolo_ms=stats.yolo_ms,
            vitpose_ms=stats.vitpose_ms,
            matching_ms=stats.matching_ms,
            triangulation_ms=stats.triangulation_ms,
            tracking_ms=stats.tracking_ms,
            total_ms=stats.total_ms,
            frames_processed=stats.frames_processed,
        )

    def reset_tracks(self):
        self._lib.pt_stream_reset_tracks(self._handle)

    def close(self):
        if self._handle and self._handle.value:
            self._lib.pt_stream_destroy(self._handle)
            self._handle = ctypes.c_void_p(None)

    def __del__(self):
        self.close()
