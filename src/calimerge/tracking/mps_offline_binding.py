"""
MPS offline (batch) pose tracking pipeline binding.

Wraps the C API ``pt_mps_offline_*`` from ``libcalimerge_mps.dylib`` via
ctypes, mirroring :mod:`calimerge.tracking.cuda_binding` (the CUDA / Windows
batch path) so the GUI's :class:`OfflineProcessingWorker` can dispatch to
either backend with the same call signature.

Architecture (kept intentionally aligned with the CUDA path):

  * The C side (``pt_offline_mps.m``) is a thin "decode + drive the
    streaming core" loop. It reuses the same ``pt_mps_stream_create`` /
    ``pt_mps_stream_process_frame`` / ``pt_mps_stream_export_csv`` calls
    that the live MPS path uses, so offline ≡ online + (decoded video frame
    source) + (optional larger batch).
  * Output filenames match the CUDA convention exactly:
    ``<output_dir>/output_3d_poses_tracked.csv_personN.csv`` — that is what
    ``OfflineProcessingWorker._convert_outputs`` globs for.

The dylib is only present on macOS. ``is_available()`` returns False on
Windows / Linux unconditionally, so import-time of this module is a no-op
on the user's Windows dev box.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# ============================================================================
# Constants (must match pt_common.h)
# ============================================================================

PT_MAX_CAMERAS = 16
PT_OK = 0
PT_ERR_FILE_NOT_FOUND = -4
PT_ERR_INVALID_PARAM = -5
PT_ERR_INVALID_ARGS = -5  # alias the C side uses
PT_ERR_OUT_OF_MEMORY = -6
PT_ERR_ENGINE_BUILD = -7
PT_ERR_INFERENCE = -8

_ERROR_MESSAGES = {
    PT_ERR_FILE_NOT_FOUND: "File not found",
    PT_ERR_INVALID_PARAM: "Invalid parameter",
    PT_ERR_OUT_OF_MEMORY: "Out of memory",
    PT_ERR_ENGINE_BUILD: "CoreML model load failed",
    PT_ERR_INFERENCE: "Inference error",
}

# ============================================================================
# Library loading
# ============================================================================

_LIB = None
_LIB_PATH: Path | None = None
_AVAILABLE = False


def _find_dylib() -> Path | None:
    """Search for libcalimerge_mps.dylib in standard locations.

    Returns None on non-Darwin platforms or if the dylib has not been built.
    """
    if sys.platform != "darwin":
        return None

    lib_name = "libcalimerge_mps.dylib"

    module_dir = Path(__file__).resolve().parent          # tracking/
    repo_root = module_dir.parent.parent.parent           # repo

    candidates = [
        repo_root / "build" / "mps" / lib_name,
        repo_root / "src" / "mps_pipeline" / lib_name,
        repo_root / "src" / "mps_pipeline" / "calimerge_mps.dylib",
        module_dir / lib_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_library() -> None:
    global _LIB, _LIB_PATH, _AVAILABLE

    if _LIB is not None:
        return

    if sys.platform != "darwin":
        logger.debug("MPS offline pipeline only available on macOS")
        return

    path = _find_dylib()
    if path is None:
        logger.debug("calimerge_mps dylib not found in search paths")
        return

    try:
        _LIB = ctypes.CDLL(str(path))
        _setup_signatures()
        _LIB_PATH = path
        _AVAILABLE = True
        logger.info("Loaded MPS offline pipeline dylib: %s", path)
    except OSError as e:
        logger.warning("Failed to load %s: %s", path, e)


def is_available() -> bool:
    """Return True iff the MPS offline pipeline is usable on this host.

    Conditions:
      1. We're on darwin.
      2. ``libcalimerge_mps.dylib`` is present and loadable.
      3. ``pt_mps_offline_*`` symbols resolve.

    Note: this does NOT verify that CoreML mlpackage files exist — that
    check happens in :func:`run_mps_pipeline` when the caller passes a path,
    so the user gets a clearer error message at runtime.
    """
    if _LIB is None:
        _load_library()
    return _AVAILABLE


# ============================================================================
# ctypes struct definitions (must match pt_offline_mps.h)
# ============================================================================

PROGRESS_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_float, ctypes.c_void_p
)
LOG_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


class PT_MPS_OfflineConfig(ctypes.Structure):
    """Maps to PT_MPS_OfflineConfig in pt_offline_mps.h."""
    _fields_ = [
        # Input paths (parallel arrays: video_paths[i] for ports[i])
        ("video_paths", (ctypes.c_char * 512) * PT_MAX_CAMERAS),
        ("ports", ctypes.c_int * PT_MAX_CAMERAS),
        ("num_cameras", ctypes.c_int),

        ("yolo_model_path", ctypes.c_char * 512),
        ("vitpose_model_path", ctypes.c_char * 512),
        ("calibration_toml_path", ctypes.c_char * 512),
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

        # Callbacks
        ("progress_callback", PROGRESS_FUNC),
        ("log_callback", LOG_FUNC),
        ("callback_user_data", ctypes.c_void_p),
    ]


class PT_MPS_OfflineStats(ctypes.Structure):
    """Maps to PT_MPS_OfflineStats in pt_offline_mps.h."""
    _fields_ = [
        ("total_seconds", ctypes.c_double),
        ("decode_seconds", ctypes.c_double),
        ("inference_seconds", ctypes.c_double),
        ("matching_seconds", ctypes.c_double),
        ("triangulation_seconds", ctypes.c_double),
        ("export_seconds", ctypes.c_double),
        ("frames_processed", ctypes.c_int),
        ("persons_tracked", ctypes.c_int),
    ]


# ============================================================================
# ctypes function signatures
# ============================================================================


def _setup_signatures() -> None:
    assert _LIB is not None

    _LIB.pt_mps_offline_create.restype = ctypes.c_int
    _LIB.pt_mps_offline_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(PT_MPS_OfflineConfig),
    ]

    _LIB.pt_mps_offline_destroy.restype = None
    _LIB.pt_mps_offline_destroy.argtypes = [ctypes.c_void_p]

    _LIB.pt_mps_offline_run.restype = ctypes.c_int
    _LIB.pt_mps_offline_run.argtypes = [ctypes.c_void_p]

    _LIB.pt_mps_offline_get_stats.restype = None
    _LIB.pt_mps_offline_get_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PT_MPS_OfflineStats),
    ]


# ============================================================================
# Helper
# ============================================================================


def _check_rc(rc: int, func_name: str) -> None:
    if rc != PT_OK:
        msg = _ERROR_MESSAGES.get(rc, f"Unknown error ({rc})")
        raise RuntimeError(f"{func_name} failed: {msg}")


# ============================================================================
# High-level API (mirrors run_cuda_pipeline exactly)
# ============================================================================


def run_mps_pipeline(
    video_paths: dict[int, Path],
    calibration_toml: Path,
    frame_time_csv: Path,
    output_path: Path,
    yolo_coreml: Path | None = None,
    vitpose_coreml: Path | None = None,
    batch_size: int = 8,
    skip_sync_indices: int = 1,
    max_persons: int = 2,
    person_confidence: float = 0.1,
    keypoint_confidence: float = 0.1,
    epipolar_threshold: float = 10.0,
    max_track_distance: float = 0.5,
    track_patience: int = 60,
    progress_callback: Callable[[str, float], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> Path | None:
    """
    Run the MPS / CoreML offline pose tracking pipeline.

    The macOS counterpart of :func:`calimerge.tracking.cuda_binding.run_cuda_pipeline`.
    The function signature and argument names match (with ``yolo_coreml`` /
    ``vitpose_coreml`` standing in for ``yolo_onnx`` / ``vitpose_onnx``), so
    :class:`calimerge.gui.workers.OfflineProcessingWorker` can dispatch to
    either backend via a single platform check.

    Args:
        video_paths: port -> Path to ``port_<n>.mp4`` (or ``port_<n>_<serial>.mp4``).
        calibration_toml: Per-session extrinsic calibration TOML file.
        frame_time_csv: ``frame_time_history.csv`` from the recording.
        output_path: Directory to write per-track CSVs into.
        yolo_coreml: Path to ``yolo_v10s.mlpackage``. If None, looks under
            the recording dir.
        vitpose_coreml: Path to ``vitpose_synthpose.mlpackage``. If None,
            looks under the recording dir.
        batch_size: Sync indices per CoreML batch.
        skip_sync_indices: Process every Nth sync index (1 = all).
        max_persons: Maximum concurrent tracked persons.
        person_confidence: YOLO detection threshold.
        keypoint_confidence: Minimum keypoint confidence.
        epipolar_threshold: Max epipolar distance in pixels.
        max_track_distance: Max 3D COM distance for track matching (default
            tuned to match the live tracker, looser than CUDA's 0.15).
        track_patience: Frames before losing a track (default tuned to
            match the live tracker, looser than CUDA's 30).
        progress_callback: ``(step_name, fraction)``.
        log_callback: ``(message,)``.

    Returns:
        ``output_path`` on success, None on failure (mirrors
        :func:`run_cuda_pipeline`'s shape).

    Raises:
        RuntimeError: dylib unavailable or pipeline error.
    """
    if not is_available():
        raise RuntimeError(
            "MPS offline pipeline dylib not available. "
            "Build it on macOS with: bash src/mps_pipeline/build_mps.sh release"
        )

    sorted_ports = sorted(video_paths.keys())
    if len(sorted_ports) > PT_MAX_CAMERAS:
        raise ValueError(
            f"Too many cameras ({len(sorted_ports)}), max is {PT_MAX_CAMERAS}"
        )

    # Default model locations (mirror cuda_binding's "look in recording dir" fallback).
    recording_dir = frame_time_csv.parent
    if yolo_coreml is None:
        yolo_coreml = recording_dir / "yolo_v10s.mlpackage"
    if vitpose_coreml is None:
        vitpose_coreml = recording_dir / "vitpose_synthpose.mlpackage"

    # --- Fill config struct ---
    config = PT_MPS_OfflineConfig()
    ctypes.memset(ctypes.byref(config), 0, ctypes.sizeof(config))

    config.num_cameras = len(sorted_ports)
    for i, port in enumerate(sorted_ports):
        path_bytes = str(video_paths[port]).encode("utf-8")
        ctypes.memmove(config.video_paths[i], path_bytes,
                       min(len(path_bytes), 511))
        config.ports[i] = port

    config.yolo_model_path = str(yolo_coreml).encode("utf-8")
    config.vitpose_model_path = str(vitpose_coreml).encode("utf-8")
    config.calibration_toml_path = str(calibration_toml).encode("utf-8")
    config.frame_time_csv_path = str(frame_time_csv).encode("utf-8")
    config.output_dir = str(output_path).encode("utf-8")

    config.batch_size = int(batch_size)
    config.skip_sync_indices = int(skip_sync_indices)
    config.max_persons = int(max_persons)
    config.person_confidence = float(person_confidence)
    config.keypoint_confidence = float(keypoint_confidence)
    config.epipolar_threshold = float(epipolar_threshold)
    config.max_track_distance = float(max_track_distance)
    config.track_patience = int(track_patience)

    # --- Callbacks (keep refs alive across the C call) ---
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

    # --- Create + run + destroy ---
    pipeline_ptr = ctypes.c_void_p(None)
    rc = _LIB.pt_mps_offline_create(ctypes.byref(pipeline_ptr),
                                     ctypes.byref(config))
    _check_rc(rc, "pt_mps_offline_create")

    try:
        rc = _LIB.pt_mps_offline_run(pipeline_ptr)
        _check_rc(rc, "pt_mps_offline_run")

        stats = PT_MPS_OfflineStats()
        _LIB.pt_mps_offline_get_stats(pipeline_ptr, ctypes.byref(stats))

        if log_callback:
            log_callback(
                f"MPS offline complete: {stats.frames_processed} frames, "
                f"{stats.total_seconds:.1f}s total"
            )
            if stats.total_seconds > 0 and stats.frames_processed > 0:
                fps = stats.frames_processed / stats.total_seconds
                log_callback(f"Throughput: {fps:.1f} sync-frames/s")

        return output_path

    finally:
        _LIB.pt_mps_offline_destroy(pipeline_ptr)
        del _progress_ref
        del _log_ref


def get_pipeline_stats_fields() -> list[str]:
    """Return the list of stat field names for display."""
    return [
        "total_seconds",
        "decode_seconds",
        "inference_seconds",
        "matching_seconds",
        "triangulation_seconds",
        "export_seconds",
        "frames_processed",
        "persons_tracked",
    ]
