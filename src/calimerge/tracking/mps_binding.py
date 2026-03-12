"""
MPS pose tracking pipeline binding for macOS Apple Silicon.

Wraps the C API from calimerge_mps.dylib via ctypes.
Falls back gracefully if the dylib is not available.

Same API pattern as cuda_binding.py.
"""

from __future__ import annotations

import ctypes
import logging
import platform
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# Constants (must match pt_common.h)
# ============================================================================

PT_MAX_CAMERAS = 16
PT_MAX_TRACKS = 32
PT_NUM_KEYPOINTS = 52
PT_OK = 0
PT_ERR_FILE_NOT_FOUND = -4
PT_ERR_INVALID_ARGS = -5
PT_ERR_ENGINE_BUILD = -7
PT_ERR_INFERENCE = -8

_ERROR_MESSAGES = {
    PT_ERR_FILE_NOT_FOUND: "File not found",
    PT_ERR_INVALID_ARGS: "Invalid arguments",
    PT_ERR_ENGINE_BUILD: "CoreML model load failed",
    PT_ERR_INFERENCE: "Inference error",
}

# ============================================================================
# DLL loading
# ============================================================================

_LIB = None
_AVAILABLE = False


def _load_library() -> None:
    global _LIB, _AVAILABLE

    if platform.system() != "Darwin":
        logger.debug("MPS pipeline only available on macOS")
        return

    search_dirs = [
        Path(__file__).resolve().parent.parent.parent / "mps_pipeline",
        Path(__file__).resolve().parent,
    ]

    for d in search_dirs:
        dylib_path = d / "calimerge_mps.dylib"
        if dylib_path.exists():
            try:
                _LIB = ctypes.CDLL(str(dylib_path))
                _setup_signatures()
                _AVAILABLE = True
                logger.info("Loaded MPS pipeline dylib: %s", dylib_path)
                return
            except OSError as e:
                logger.warning("Failed to load %s: %s", dylib_path, e)

    logger.debug("MPS pipeline dylib not found in search paths")


def is_available() -> bool:
    """Check if the MPS pipeline dylib is available."""
    if _LIB is None:
        _load_library()
    return _AVAILABLE


# ============================================================================
# ctypes struct definitions (must match pt_stream_mps.h)
# ============================================================================

LOG_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


class PT_MPS_StreamConfig(ctypes.Structure):
    _fields_ = [
        ("yolo_model_path", ctypes.c_char * 512),
        ("vitpose_model_path", ctypes.c_char * 512),
        ("calibration_toml_path", ctypes.c_char * 512),
        ("num_cameras", ctypes.c_int),
        ("frame_width", ctypes.c_int),
        ("frame_height", ctypes.c_int),
        ("max_persons", ctypes.c_int),
        ("person_confidence", ctypes.c_float),
        ("keypoint_confidence", ctypes.c_float),
        ("epipolar_threshold", ctypes.c_float),
        ("max_track_distance", ctypes.c_float),
        ("track_patience", ctypes.c_int),
        ("log_callback", LOG_FUNC),
        ("callback_user_data", ctypes.c_void_p),
    ]


class PT_MPS_StreamFrame(ctypes.Structure):
    _fields_ = [
        ("pixels", ctypes.POINTER(ctypes.c_uint8)),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("stride", ctypes.c_int),
        ("port", ctypes.c_int),
    ]


class PT_MPS_StreamFrameSet(ctypes.Structure):
    _fields_ = [
        ("frames", PT_MPS_StreamFrame * PT_MAX_CAMERAS),
        ("num_frames", ctypes.c_int),
        ("sync_index", ctypes.c_uint64),
    ]


class PT_MPS_StreamPerson(ctypes.Structure):
    _fields_ = [
        ("person_id", ctypes.c_int),
        ("keypoints_3d", (ctypes.c_double * 3) * PT_NUM_KEYPOINTS),
        ("keypoints_valid", ctypes.c_int * PT_NUM_KEYPOINTS),
        ("com_3d", ctypes.c_double * 3),
        ("com_valid", ctypes.c_int),
        ("num_views", ctypes.c_int),
    ]


class PT_MPS_StreamResult(ctypes.Structure):
    _fields_ = [
        ("persons", PT_MPS_StreamPerson * PT_MAX_TRACKS),
        ("num_persons", ctypes.c_int),
        ("sync_index", ctypes.c_uint64),
        ("processing_time_ms", ctypes.c_double),
    ]


class PT_MPS_StreamStats(ctypes.Structure):
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


# ============================================================================
# Function signatures
# ============================================================================

def _setup_signatures() -> None:
    _LIB.pt_mps_stream_create.restype = ctypes.c_int
    _LIB.pt_mps_stream_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(PT_MPS_StreamConfig),
    ]

    _LIB.pt_mps_stream_destroy.restype = None
    _LIB.pt_mps_stream_destroy.argtypes = [ctypes.c_void_p]

    _LIB.pt_mps_stream_process_frame.restype = ctypes.c_int
    _LIB.pt_mps_stream_process_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PT_MPS_StreamFrameSet),
        ctypes.POINTER(PT_MPS_StreamResult),
    ]

    _LIB.pt_mps_stream_get_stats.restype = None
    _LIB.pt_mps_stream_get_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PT_MPS_StreamStats),
    ]

    _LIB.pt_mps_stream_reset_tracks.restype = None
    _LIB.pt_mps_stream_reset_tracks.argtypes = [ctypes.c_void_p]

    _LIB.pt_mps_stream_export_csv.restype = ctypes.c_int
    _LIB.pt_mps_stream_export_csv.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]


def _check_rc(rc: int, func_name: str) -> None:
    if rc != PT_OK:
        msg = _ERROR_MESSAGES.get(rc, f"Unknown error ({rc})")
        raise RuntimeError(f"{func_name} failed: {msg}")


# ============================================================================
# Streaming API wrapper
# ============================================================================

class MPSStreamPipeline:
    """High-level wrapper for the MPS streaming pose tracking pipeline."""

    def __init__(
        self,
        calibration_toml: str | Path,
        yolo_model: str | Path,
        vitpose_model: str | Path,
        num_cameras: int,
        frame_width: int,
        frame_height: int,
        max_persons: int = 2,
        person_confidence: float = 0.1,
        keypoint_confidence: float = 0.1,
        epipolar_threshold: float = 10.0,
        max_track_distance: float = 0.15,
        track_patience: int = 30,
        log_callback=None,
    ):
        if not is_available():
            raise RuntimeError(
                "MPS pipeline dylib not available. "
                "Build it with build_mps_macos.sh first."
            )

        self._handle = ctypes.c_void_p(None)
        self._log_ref = None

        config = PT_MPS_StreamConfig()
        ctypes.memset(ctypes.byref(config), 0, ctypes.sizeof(config))

        config.yolo_model_path = str(yolo_model).encode("utf-8")
        config.vitpose_model_path = str(vitpose_model).encode("utf-8")
        config.calibration_toml_path = str(calibration_toml).encode("utf-8")
        config.num_cameras = num_cameras
        config.frame_width = frame_width
        config.frame_height = frame_height
        config.max_persons = max_persons
        config.person_confidence = person_confidence
        config.keypoint_confidence = keypoint_confidence
        config.epipolar_threshold = epipolar_threshold
        config.max_track_distance = max_track_distance
        config.track_patience = track_patience

        if log_callback is not None:
            def _log_trampoline(message, user_data):
                try:
                    msg_str = message.decode("utf-8") if message else ""
                    log_callback(msg_str)
                except Exception:
                    pass

            self._log_ref = LOG_FUNC(_log_trampoline)
            config.log_callback = self._log_ref

        rc = _LIB.pt_mps_stream_create(
            ctypes.byref(self._handle),
            ctypes.byref(config),
        )
        _check_rc(rc, "pt_mps_stream_create")

    def process_frame(
        self,
        frames: dict[int, np.ndarray],
        sync_index: int,
    ) -> list[dict]:
        """Process one synchronized frame set.

        Args:
            frames: port -> BGR numpy array (H, W, 3) uint8
            sync_index: monotonic frame counter

        Returns:
            List of person dicts with keys:
                person_id, keypoints_3d (52,3), keypoints_valid (52,),
                com_3d (3,), num_views, processing_time_ms
        """
        frame_set = PT_MPS_StreamFrameSet()
        ctypes.memset(ctypes.byref(frame_set), 0, ctypes.sizeof(frame_set))
        frame_set.sync_index = sync_index

        i = 0
        for port, bgr in frames.items():
            if i >= PT_MAX_CAMERAS:
                break
            h, w = bgr.shape[:2]
            frame_set.frames[i].pixels = bgr.ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint8)
            )
            frame_set.frames[i].width = w
            frame_set.frames[i].height = h
            frame_set.frames[i].stride = w * 3
            frame_set.frames[i].port = port
            i += 1
        frame_set.num_frames = i

        result = PT_MPS_StreamResult()
        rc = _LIB.pt_mps_stream_process_frame(
            self._handle,
            ctypes.byref(frame_set),
            ctypes.byref(result),
        )
        _check_rc(rc, "pt_mps_stream_process_frame")

        persons = []
        for p in range(result.num_persons):
            person = result.persons[p]
            kp3d = np.zeros((PT_NUM_KEYPOINTS, 3), dtype=np.float64)
            kp_valid = np.zeros(PT_NUM_KEYPOINTS, dtype=np.int32)
            for k in range(PT_NUM_KEYPOINTS):
                kp3d[k, 0] = person.keypoints_3d[k][0]
                kp3d[k, 1] = person.keypoints_3d[k][1]
                kp3d[k, 2] = person.keypoints_3d[k][2]
                kp_valid[k] = person.keypoints_valid[k]

            persons.append({
                "person_id": person.person_id,
                "keypoints_3d": kp3d,
                "keypoints_valid": kp_valid,
                "com_3d": np.array([person.com_3d[0], person.com_3d[1], person.com_3d[2]]),
                "num_views": person.num_views,
            })

        return persons

    def get_stats(self) -> dict:
        stats = PT_MPS_StreamStats()
        _LIB.pt_mps_stream_get_stats(self._handle, ctypes.byref(stats))
        return {
            "coreml_yolo_ms": stats.coreml_yolo_ms,
            "coreml_vitpose_ms": stats.coreml_vitpose_ms,
            "preprocess_ms": stats.preprocess_ms,
            "matching_ms": stats.matching_ms,
            "triangulation_ms": stats.triangulation_ms,
            "tracking_ms": stats.tracking_ms,
            "total_ms": stats.total_ms,
            "frames_processed": stats.frames_processed,
        }

    def reset_tracks(self) -> None:
        _LIB.pt_mps_stream_reset_tracks(self._handle)

    def export_csv(self, output_base_path: str | Path) -> None:
        rc = _LIB.pt_mps_stream_export_csv(
            self._handle,
            str(output_base_path).encode("utf-8"),
        )
        _check_rc(rc, "pt_mps_stream_export_csv")

    def destroy(self) -> None:
        if self._handle:
            _LIB.pt_mps_stream_destroy(self._handle)
            self._handle = ctypes.c_void_p(None)

    def __del__(self):
        self.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.destroy()
