"""
CUDA pose tracking pipeline binding.

Wraps the C API from calimerge_cuda.dll via ctypes.
Falls back gracefully if the DLL is not available.
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# ============================================================================
# Constants (must match pt_common.h)
# ============================================================================

PT_MAX_CAMERAS = 16
PT_OK = 0
PT_ERR_CUDA = -1
PT_ERR_TENSORRT = -2
PT_ERR_NVDEC = -3
PT_ERR_FILE_NOT_FOUND = -4
PT_ERR_INVALID_PARAM = -5
PT_ERR_OUT_OF_MEMORY = -6
PT_ERR_ENGINE_BUILD = -7
PT_ERR_INFERENCE = -8
PT_ERR_DECODE = -9
PT_ERR_NOT_INITIALIZED = -10

_ERROR_MESSAGES = {
    PT_ERR_CUDA: "CUDA error",
    PT_ERR_TENSORRT: "TensorRT error",
    PT_ERR_NVDEC: "NVDEC decode error",
    PT_ERR_FILE_NOT_FOUND: "File not found",
    PT_ERR_INVALID_PARAM: "Invalid parameter",
    PT_ERR_OUT_OF_MEMORY: "Out of memory",
    PT_ERR_ENGINE_BUILD: "TensorRT engine build failed",
    PT_ERR_INFERENCE: "Inference error",
    PT_ERR_DECODE: "Video decode error",
    PT_ERR_NOT_INITIALIZED: "Pipeline not initialized",
}

# ============================================================================
# DLL loading
# ============================================================================

_LIB = None
_AVAILABLE = False


def _load_library() -> None:
    global _LIB, _AVAILABLE

    # Search paths: cuda_pipeline dir (sibling of calimerge package),
    # next to this file, system PATH
    search_dirs = [
        Path(__file__).resolve().parent.parent.parent / "cuda_pipeline",
        Path(__file__).resolve().parent,
    ]

    for d in search_dirs:
        dll_path = d / "calimerge_cuda.dll"
        if dll_path.exists():
            try:
                _LIB = ctypes.CDLL(str(dll_path))
                _setup_signatures()
                _AVAILABLE = True
                logger.info("Loaded CUDA pipeline DLL: %s", dll_path)
                return
            except OSError as e:
                logger.warning("Failed to load %s: %s", dll_path, e)

    logger.debug("CUDA pipeline DLL not found in search paths")


def is_available() -> bool:
    """Check if the CUDA pipeline DLL is available."""
    if _LIB is None:
        _load_library()
    return _AVAILABLE


# ============================================================================
# ctypes struct definitions (must match pt_common.h)
# ============================================================================

# Callback function types
PROGRESS_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_float, ctypes.c_void_p
)
LOG_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


class PT_PipelineConfig(ctypes.Structure):
    """Maps to PT_PipelineConfig in pt_common.h."""
    _fields_ = [
        # Input paths
        ("video_paths", (ctypes.c_char * 512) * PT_MAX_CAMERAS),
        ("num_cameras", ctypes.c_int),
        ("yolo_onnx_path", ctypes.c_char * 512),
        ("vitpose_onnx_path", ctypes.c_char * 512),
        ("engine_cache_dir", ctypes.c_char * 512),
        ("frame_time_csv_path", ctypes.c_char * 512),
        ("output_dir", ctypes.c_char * 512),
        # Processing parameters
        ("batch_size", ctypes.c_int),
        ("skip_sync_indices", ctypes.c_int),
        ("max_persons", ctypes.c_int),
        ("person_confidence", ctypes.c_float),
        ("keypoint_confidence", ctypes.c_float),
        ("epipolar_threshold", ctypes.c_float),
        ("max_track_distance", ctypes.c_float),
        ("track_patience", ctypes.c_int),
        # Device
        ("use_fp16_yolo", ctypes.c_int),
        # Callbacks
        ("progress_callback", PROGRESS_FUNC),
        ("log_callback", LOG_FUNC),
        ("callback_user_data", ctypes.c_void_p),
    ]


class PT_Stats(ctypes.Structure):
    """Maps to PT_Stats in pt_common.h."""
    _fields_ = [
        ("total_seconds", ctypes.c_double),
        ("decode_seconds", ctypes.c_double),
        ("yolo_seconds", ctypes.c_double),
        ("vitpose_seconds", ctypes.c_double),
        ("matching_seconds", ctypes.c_double),
        ("triangulation_seconds", ctypes.c_double),
        ("export_seconds", ctypes.c_double),
        ("frames_processed", ctypes.c_int),
        ("persons_tracked", ctypes.c_int),
    ]


# ============================================================================
# Function signatures
# ============================================================================

def _setup_signatures() -> None:
    """Set up ctypes function signatures for the C API."""
    # pt_pipeline_create(PT_Pipeline **out, const PT_PipelineConfig *config) -> int
    _LIB.pt_pipeline_create.restype = ctypes.c_int
    _LIB.pt_pipeline_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(PT_PipelineConfig),
    ]

    # pt_pipeline_destroy(PT_Pipeline *p) -> void
    _LIB.pt_pipeline_destroy.restype = None
    _LIB.pt_pipeline_destroy.argtypes = [ctypes.c_void_p]

    # pt_pipeline_load_calibration(PT_Pipeline *p, const char *toml_path) -> int
    _LIB.pt_pipeline_load_calibration.restype = ctypes.c_int
    _LIB.pt_pipeline_load_calibration.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]

    # pt_pipeline_load_sync_table(PT_Pipeline *p, const char *csv_path) -> int
    _LIB.pt_pipeline_load_sync_table.restype = ctypes.c_int
    _LIB.pt_pipeline_load_sync_table.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]

    # pt_pipeline_run(PT_Pipeline *p) -> int
    _LIB.pt_pipeline_run.restype = ctypes.c_int
    _LIB.pt_pipeline_run.argtypes = [ctypes.c_void_p]

    # pt_pipeline_get_stats(const PT_Pipeline *p, PT_Stats *out) -> void
    _LIB.pt_pipeline_get_stats.restype = None
    _LIB.pt_pipeline_get_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PT_Stats),
    ]


# ============================================================================
# Helper: error code to message
# ============================================================================

def _check_rc(rc: int, func_name: str) -> None:
    """Raise RuntimeError if the return code indicates failure."""
    if rc != PT_OK:
        msg = _ERROR_MESSAGES.get(rc, f"Unknown error ({rc})")
        raise RuntimeError(f"{func_name} failed: {msg}")


# ============================================================================
# High-level Python API
# ============================================================================

def run_cuda_pipeline(
    video_paths: dict[int, Path],
    calibration_toml: Path,
    frame_time_csv: Path,
    output_path: Path,
    yolo_onnx: Path | None = None,
    vitpose_onnx: Path | None = None,
    batch_size: int = 8,
    skip_sync_indices: int = 1,
    max_persons: int = 2,
    person_confidence: float = 0.1,
    keypoint_confidence: float = 0.1,
    epipolar_threshold: float = 10.0,
    max_track_distance: float = 0.15,
    track_patience: int = 30,
    use_fp16_yolo: bool = True,
    progress_callback: Callable[[str, float], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> Path | None:
    """
    Run the CUDA pose tracking pipeline.

    This is the GPU-accelerated equivalent of
    ``calimerge.tracking.pipeline.run_pose_tracking()``.

    Args:
        video_paths: port -> Path to video file.
        calibration_toml: Path to calibration TOML file.
        frame_time_csv: Path to frame_time_history.csv.
        output_path: Directory to write results into.
        yolo_onnx: Path to YOLO v10s ONNX model. If None, looks in
            recording dir for yolo_v10s.onnx.
        vitpose_onnx: Path to VitPose ONNX model. If None, looks in
            recording dir for vitpose_base_coco_wholebody.onnx.
        batch_size: Sync indices per batch.
        skip_sync_indices: Process every Nth sync index (1 = all).
        max_persons: Maximum concurrent tracked persons.
        person_confidence: YOLO detection threshold.
        keypoint_confidence: Minimum keypoint confidence.
        epipolar_threshold: Max epipolar distance in pixels.
        max_track_distance: Max 3D COM distance for track matching.
        track_patience: Frames before losing a track.
        use_fp16_yolo: Use FP16 for YOLO inference.
        progress_callback: Called with (step_name, fraction).
        log_callback: Called with log messages.

    Returns:
        Path to output directory on success, None on failure.

    Raises:
        RuntimeError: If the CUDA pipeline DLL is not available or if
            any pipeline step fails.
    """
    if not is_available():
        raise RuntimeError(
            "CUDA pipeline DLL not available. "
            "Build it with build_cuda_win32.bat first."
        )

    # Sort ports to get consistent camera ordering
    sorted_ports = sorted(video_paths.keys())
    if len(sorted_ports) > PT_MAX_CAMERAS:
        raise ValueError(
            f"Too many cameras ({len(sorted_ports)}), max is {PT_MAX_CAMERAS}"
        )

    # Determine model paths from the recording directory if not specified
    recording_dir = frame_time_csv.parent
    if yolo_onnx is None:
        yolo_onnx = recording_dir / "yolo_v10s.onnx"
    if vitpose_onnx is None:
        vitpose_onnx = recording_dir / "vitpose_base_coco_wholebody.onnx"

    # Build engine cache dir next to the output
    engine_cache_dir = output_path / "engine_cache"

    # --- Fill config struct ---
    config = PT_PipelineConfig()
    ctypes.memset(ctypes.byref(config), 0, ctypes.sizeof(config))

    config.num_cameras = len(sorted_ports)

    for i, port in enumerate(sorted_ports):
        path_bytes = str(video_paths[port]).encode("utf-8")
        ctypes.memmove(config.video_paths[i], path_bytes, min(len(path_bytes), 511))

    config.yolo_onnx_path = str(yolo_onnx).encode("utf-8")
    config.vitpose_onnx_path = str(vitpose_onnx).encode("utf-8")
    config.engine_cache_dir = str(engine_cache_dir).encode("utf-8")
    config.frame_time_csv_path = str(frame_time_csv).encode("utf-8")
    config.output_dir = str(output_path).encode("utf-8")

    config.batch_size = batch_size
    config.skip_sync_indices = skip_sync_indices
    config.max_persons = max_persons
    config.person_confidence = person_confidence
    config.keypoint_confidence = keypoint_confidence
    config.epipolar_threshold = epipolar_threshold
    config.max_track_distance = max_track_distance
    config.track_patience = track_patience
    config.use_fp16_yolo = 1 if use_fp16_yolo else 0

    # --- Set up callbacks ---
    # We need to keep references to the ctypes callback objects so they
    # are not garbage collected while the pipeline is running.
    _progress_ref = None
    _log_ref = None

    if progress_callback is not None:
        def _progress_trampoline(step, fraction, user_data):
            try:
                step_str = step.decode("utf-8") if step else ""
                progress_callback(step_str, fraction)
            except Exception:
                pass

        _progress_ref = PROGRESS_FUNC(_progress_trampoline)
        config.progress_callback = _progress_ref

    if log_callback is not None:
        def _log_trampoline(message, user_data):
            try:
                msg_str = message.decode("utf-8") if message else ""
                log_callback(msg_str)
            except Exception:
                pass

        _log_ref = LOG_FUNC(_log_trampoline)
        config.log_callback = _log_ref

    config.callback_user_data = None

    # --- Create pipeline ---
    pipeline_ptr = ctypes.c_void_p(None)
    rc = _LIB.pt_pipeline_create(ctypes.byref(pipeline_ptr), ctypes.byref(config))
    _check_rc(rc, "pt_pipeline_create")

    try:
        # --- Load calibration ---
        calib_bytes = str(calibration_toml).encode("utf-8")
        rc = _LIB.pt_pipeline_load_calibration(pipeline_ptr, calib_bytes)
        _check_rc(rc, "pt_pipeline_load_calibration")

        # --- Load sync table ---
        csv_bytes = str(frame_time_csv).encode("utf-8")
        rc = _LIB.pt_pipeline_load_sync_table(pipeline_ptr, csv_bytes)
        _check_rc(rc, "pt_pipeline_load_sync_table")

        # --- Run ---
        rc = _LIB.pt_pipeline_run(pipeline_ptr)
        _check_rc(rc, "pt_pipeline_run")

        # --- Get stats ---
        stats = PT_Stats()
        _LIB.pt_pipeline_get_stats(pipeline_ptr, ctypes.byref(stats))

        if log_callback:
            log_callback(
                f"CUDA pipeline complete: {stats.frames_processed} frames, "
                f"{stats.persons_tracked} persons, "
                f"{stats.total_seconds:.1f}s total"
            )
            if stats.total_seconds > 0 and stats.frames_processed > 0:
                fps = stats.frames_processed / stats.total_seconds
                log_callback(f"Throughput: {fps:.1f} sync-frames/s")

        return output_path

    except RuntimeError:
        # Re-raise pipeline errors after cleanup
        raise

    finally:
        # --- Destroy pipeline ---
        _LIB.pt_pipeline_destroy(pipeline_ptr)

        # Keep callback references alive until after destroy
        del _progress_ref
        del _log_ref


def get_pipeline_stats_fields() -> list[str]:
    """Return the list of stat field names for display."""
    return [
        "total_seconds",
        "decode_seconds",
        "yolo_seconds",
        "vitpose_seconds",
        "matching_seconds",
        "triangulation_seconds",
        "export_seconds",
        "frames_processed",
        "persons_tracked",
    ]
