"""
camera_binding.py - Python ctypes bindings for the calimerge C++ camera module.

Usage:
    from calimerge.camera_binding import (
        init, shutdown, enumerate_cameras,
        open_camera, close_camera, capture_frame
    )

    init()
    cameras = enumerate_cameras()
    for cam in cameras:
        print(f"{cam.display_name}: {cam.serial_number}")

    open_camera(cameras[0])
    frame = capture_frame(cameras[0])
    print(f"Frame: {frame.width}x{frame.height}")
    close_camera(cameras[0])
    shutdown()
"""

import ctypes
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# ============================================================================
# Constants (must match calimerge_platform.h)
# ============================================================================

CM_MAX_CAMERAS = 16
CM_SERIAL_LEN = 64
CM_NAME_LEN = 128
CM_RES_COUNT = 3

CM_OK = 0
CM_ERROR_INIT_FAILED = -1
CM_ERROR_NO_CAMERA = -2
CM_ERROR_OPEN_FAILED = -3
CM_ERROR_CAPTURE_FAILED = -4
CM_ERROR_INVALID_PARAM = -5
CM_ERROR_NOT_SUPPORTED = -6

# ============================================================================
# Load Library
# ============================================================================

def _find_library() -> Path:
    """Find the calimerge shared library."""
    # Look relative to this file
    module_dir = Path(__file__).parent

    # Try various locations
    candidates = [
        module_dir.parent.parent / "native" / "libcalimerge.dylib",  # src/native/
        module_dir.parent / "native" / "libcalimerge.dylib",
        module_dir / "libcalimerge.dylib",
        Path("/usr/local/lib/libcalimerge.dylib"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find libcalimerge.dylib. Searched: {[str(p) for p in candidates]}"
    )

_lib_path = _find_library()
_lib = ctypes.CDLL(str(_lib_path))

# ============================================================================
# C Struct Definitions
# ============================================================================

class CM_Resolution(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
    ]

class CM_Camera(ctypes.Structure):
    _fields_ = [
        ("serial_number", ctypes.c_char * CM_SERIAL_LEN),
        ("display_name", ctypes.c_char * CM_NAME_LEN),
        ("device_index", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("fps", ctypes.c_int),
        ("rotation", ctypes.c_int),
        ("exposure", ctypes.c_int),
        ("enabled", ctypes.c_bool),
        ("supported_resolutions", CM_Resolution * CM_RES_COUNT),
        ("supported_resolution_count", ctypes.c_int),
        ("platform_handle", ctypes.c_void_p),
    ]

class CM_Frame(ctypes.Structure):
    _fields_ = [
        ("pixels", ctypes.POINTER(ctypes.c_uint8)),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("stride", ctypes.c_int),
        ("timestamp_ns", ctypes.c_uint64),
        ("arrival_ns", ctypes.c_uint64),
        ("corrected_ns", ctypes.c_uint64),
        ("camera_index", ctypes.c_int),
    ]

class CM_SyncedFrameSet(ctypes.Structure):
    _fields_ = [
        ("frames", CM_Frame * CM_MAX_CAMERAS),
        ("frame_count", ctypes.c_int),
        ("dropped_mask", ctypes.c_int),
        ("sync_index", ctypes.c_uint64),
    ]

# ============================================================================
# Function Signatures
# ============================================================================

# Lifecycle
_lib.cm_init.argtypes = []
_lib.cm_init.restype = ctypes.c_int

_lib.cm_shutdown.argtypes = []
_lib.cm_shutdown.restype = None

# Enumeration
_lib.cm_enumerate_cameras.argtypes = [ctypes.POINTER(CM_Camera), ctypes.c_int]
_lib.cm_enumerate_cameras.restype = ctypes.c_int

_lib.cm_get_camera_serial.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
_lib.cm_get_camera_serial.restype = ctypes.c_int

# Camera control
_lib.cm_open_camera.argtypes = [ctypes.POINTER(CM_Camera)]
_lib.cm_open_camera.restype = ctypes.c_int

_lib.cm_close_camera.argtypes = [ctypes.POINTER(CM_Camera)]
_lib.cm_close_camera.restype = None

_lib.cm_set_resolution.argtypes = [ctypes.POINTER(CM_Camera), ctypes.c_int, ctypes.c_int]
_lib.cm_set_resolution.restype = ctypes.c_int

_lib.cm_set_fps.argtypes = [ctypes.POINTER(CM_Camera), ctypes.c_int]
_lib.cm_set_fps.restype = ctypes.c_int

_lib.cm_set_exposure.argtypes = [ctypes.POINTER(CM_Camera), ctypes.c_int]
_lib.cm_set_exposure.restype = ctypes.c_int

# Frame capture
_lib.cm_capture_frame.argtypes = [ctypes.POINTER(CM_Camera), ctypes.POINTER(CM_Frame)]
_lib.cm_capture_frame.restype = ctypes.c_int

_lib.cm_release_frame.argtypes = [ctypes.POINTER(CM_Frame)]
_lib.cm_release_frame.restype = None

_lib.cm_get_latest_timestamp.argtypes = [ctypes.POINTER(CM_Camera)]
_lib.cm_get_latest_timestamp.restype = ctypes.c_uint64

# Multi-camera
_lib.cm_capture_synced.argtypes = [ctypes.POINTER(CM_Camera), ctypes.c_int, ctypes.POINTER(CM_SyncedFrameSet)]
_lib.cm_capture_synced.restype = ctypes.c_int

_lib.cm_release_synced.argtypes = [ctypes.POINTER(CM_SyncedFrameSet)]
_lib.cm_release_synced.restype = None

# ============================================================================
# Python-Friendly Data Classes
# ============================================================================

@dataclass
class CameraInfo:
    """Python-friendly camera information."""
    serial_number: str
    display_name: str
    device_index: int
    width: int
    height: int
    fps: int
    rotation: int
    exposure: int
    enabled: bool
    supported_resolutions: list[tuple[int, int]]

    # Internal: reference to the C struct for API calls
    _c_camera: Optional[CM_Camera] = None

@dataclass
class Frame:
    """A captured frame with numpy array data."""
    pixels: np.ndarray  # BGR format, shape (height, width, 3)
    width: int
    height: int
    timestamp_ns: int       # Camera's native PTS (its own clock domain)
    arrival_ns: int         # Common clock arrival time (mach_absolute_time)
    corrected_ns: int       # PTS + clock_offset = common clock domain
    camera_index: int

    @property
    def timestamp_ms(self) -> float:
        """Camera's native PTS in milliseconds."""
        return self.timestamp_ns / 1e6

    @property
    def timestamp_s(self) -> float:
        """Camera's native PTS in seconds."""
        return self.timestamp_ns / 1e9

    @property
    def corrected_ms(self) -> float:
        """Offset-corrected timestamp in milliseconds (comparable across cameras)."""
        return self.corrected_ns / 1e6

@dataclass
class SyncedFrameSet:
    """A set of synchronized frames from multiple cameras."""
    frames: dict[int, Optional[Frame]]  # camera_index -> Frame or None if dropped
    sync_index: int
    dropped_mask: int

    @property
    def dropped_cameras(self) -> list[int]:
        """List of camera indices that dropped frames."""
        dropped = []
        for i in range(CM_MAX_CAMERAS):
            if self.dropped_mask & (1 << i):
                dropped.append(i)
        return dropped

# ============================================================================
# High-Level API
# ============================================================================

_initialized = False
_cameras: list[CM_Camera] = []

def init() -> None:
    """Initialize the camera subsystem."""
    global _initialized
    if _initialized:
        return

    result = _lib.cm_init()
    if result != CM_OK:
        raise RuntimeError(f"Failed to initialize camera subsystem: {result}")
    _initialized = True

def shutdown() -> None:
    """Shutdown the camera subsystem."""
    global _initialized, _cameras
    if not _initialized:
        return

    _lib.cm_shutdown()
    _cameras = []
    _initialized = False

def enumerate_cameras() -> list[CameraInfo]:
    """
    Enumerate available cameras.

    Returns a list of CameraInfo objects with camera details.
    """
    if not _initialized:
        init()

    global _cameras
    _cameras = (CM_Camera * CM_MAX_CAMERAS)()
    count = _lib.cm_enumerate_cameras(_cameras, CM_MAX_CAMERAS)

    if count < 0:
        raise RuntimeError(f"Camera enumeration failed: {count}")

    result = []
    for i in range(count):
        cam = _cameras[i]

        # Extract supported resolutions
        resolutions = []
        for r in range(cam.supported_resolution_count):
            res = cam.supported_resolutions[r]
            resolutions.append((res.width, res.height))

        info = CameraInfo(
            serial_number=cam.serial_number.decode('utf-8'),
            display_name=cam.display_name.decode('utf-8'),
            device_index=cam.device_index,
            width=cam.width,
            height=cam.height,
            fps=cam.fps,
            rotation=cam.rotation,
            exposure=cam.exposure,
            enabled=cam.enabled,
            supported_resolutions=resolutions,
            _c_camera=cam,
        )
        result.append(info)

    return result

def open_camera(camera: CameraInfo) -> None:
    """Open a camera for capture."""
    if camera._c_camera is None:
        raise ValueError("Invalid camera object")

    result = _lib.cm_open_camera(ctypes.byref(camera._c_camera))
    if result != CM_OK:
        raise RuntimeError(f"Failed to open camera: {result}")

def close_camera(camera: CameraInfo) -> None:
    """Close a camera."""
    if camera._c_camera is None:
        return
    _lib.cm_close_camera(ctypes.byref(camera._c_camera))

def set_resolution(camera: CameraInfo, width: int, height: int) -> None:
    """Set camera resolution."""
    if camera._c_camera is None:
        raise ValueError("Invalid camera object")

    result = _lib.cm_set_resolution(ctypes.byref(camera._c_camera), width, height)
    if result != CM_OK:
        raise RuntimeError(f"Failed to set resolution: {result}")

    camera.width = width
    camera.height = height

def set_fps(camera: CameraInfo, fps: int) -> None:
    """Set camera frame rate."""
    if camera._c_camera is None:
        raise ValueError("Invalid camera object")

    result = _lib.cm_set_fps(ctypes.byref(camera._c_camera), fps)
    if result != CM_OK:
        raise RuntimeError(f"Failed to set FPS: {result}")

    camera.fps = fps

def capture_frame(camera: CameraInfo) -> Frame:
    """
    Capture a single frame from a camera.

    Returns a Frame object with the image as a numpy array.
    """
    if camera._c_camera is None:
        raise ValueError("Invalid camera object")

    c_frame = CM_Frame()
    result = _lib.cm_capture_frame(ctypes.byref(camera._c_camera), ctypes.byref(c_frame))

    if result != CM_OK:
        raise RuntimeError(f"Failed to capture frame: {result}")

    # Convert to numpy array (copy data since C will free the buffer)
    size = c_frame.height * c_frame.width * 3
    buffer = ctypes.cast(c_frame.pixels, ctypes.POINTER(ctypes.c_uint8 * size))
    pixels = np.frombuffer(buffer.contents, dtype=np.uint8).reshape(
        (c_frame.height, c_frame.width, 3)
    ).copy()

    frame = Frame(
        pixels=pixels,
        width=c_frame.width,
        height=c_frame.height,
        timestamp_ns=c_frame.timestamp_ns,
        arrival_ns=c_frame.arrival_ns,
        corrected_ns=c_frame.corrected_ns,
        camera_index=c_frame.camera_index,
    )

    # Release C buffer
    _lib.cm_release_frame(ctypes.byref(c_frame))

    return frame

def capture_synced(cameras: list[CameraInfo]) -> SyncedFrameSet:
    """
    Capture synchronized frames from multiple cameras.

    Returns a SyncedFrameSet with frames from all cameras.
    Cameras that failed to produce a frame will have None in the frames dict.
    """
    if not cameras:
        raise ValueError("No cameras provided")

    # Build C array of cameras
    c_cameras = (CM_Camera * len(cameras))()
    for i, cam in enumerate(cameras):
        if cam._c_camera is None:
            raise ValueError(f"Invalid camera object at index {i}")
        c_cameras[i] = cam._c_camera

    c_frameset = CM_SyncedFrameSet()
    result = _lib.cm_capture_synced(c_cameras, len(cameras), ctypes.byref(c_frameset))

    if result != CM_OK:
        raise RuntimeError(f"Failed to capture synced frames: {result}")

    # Convert frames to Python
    frames = {}
    for i in range(len(cameras)):
        c_frame = c_frameset.frames[i]
        if c_frame.pixels:
            size = c_frame.height * c_frame.width * 3
            buffer = ctypes.cast(c_frame.pixels, ctypes.POINTER(ctypes.c_uint8 * size))
            pixels = np.frombuffer(buffer.contents, dtype=np.uint8).reshape(
                (c_frame.height, c_frame.width, 3)
            ).copy()

            frames[i] = Frame(
                pixels=pixels,
                width=c_frame.width,
                height=c_frame.height,
                timestamp_ns=c_frame.timestamp_ns,
                arrival_ns=c_frame.arrival_ns,
                corrected_ns=c_frame.corrected_ns,
                camera_index=c_frame.camera_index,
            )
        else:
            frames[i] = None

    frameset = SyncedFrameSet(
        frames=frames,
        sync_index=c_frameset.sync_index,
        dropped_mask=c_frameset.dropped_mask,
    )

    # Release C buffers
    _lib.cm_release_synced(ctypes.byref(c_frameset))

    return frameset

# ============================================================================
# Context Manager
# ============================================================================

class Camera:
    """Context manager for camera capture."""

    def __init__(self, camera_info: CameraInfo):
        self.info = camera_info
        self._opened = False

    def __enter__(self):
        open_camera(self.info)
        self._opened = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._opened:
            close_camera(self.info)
            self._opened = False
        return False

    def capture(self) -> Frame:
        """Capture a frame."""
        return capture_frame(self.info)

    @property
    def serial(self) -> str:
        return self.info.serial_number

    @property
    def name(self) -> str:
        return self.info.display_name
