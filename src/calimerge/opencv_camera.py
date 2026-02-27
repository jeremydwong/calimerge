"""
opencv_camera.py - OpenCV-based camera capture with native serial number support.

This module uses:
- Native library (AVFoundation/MSMF) for camera enumeration and serial numbers
- OpenCV for all capture and camera control (resolution, FPS, exposure)

Why this hybrid approach:
- OpenCV exposes exposure control that works on USB webcams
- Native library provides unique serial numbers (not available in OpenCV)
- OpenCV capture is well-tested and cross-platform
"""

import sys
import platform
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

# ============================================================================
# Native Library for Serial Numbers Only
# ============================================================================

def _get_native_camera_serials() -> dict[int, tuple[str, str]]:
    """
    Get camera serial numbers from native library.

    Returns:
        dict mapping device_index -> (serial_number, display_name)
    """
    try:
        from . import camera_binding
        camera_binding.init()
        cameras = camera_binding.enumerate_cameras()
        result = {}
        for cam in cameras:
            result[cam.device_index] = (cam.serial_number, cam.display_name)
        return result
    except Exception as e:
        print(f"Warning: Native enumeration failed ({e}), using port-based IDs")
        return {}


# ============================================================================
# OpenCV Backend Selection
# ============================================================================

def _get_opencv_backend() -> int:
    """Get the appropriate OpenCV backend for the platform.

    Note: MWC uses CAP_ANY on macOS (not CAP_AVFOUNDATION) and CAP_DSHOW on Windows.
    """
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    else:
        # MWC uses CAP_ANY on UNIX/macOS
        return cv2.CAP_ANY


# ============================================================================
# Camera Data Classes
# ============================================================================

@dataclass
class CameraInfo:
    """Camera information - compatible with existing camera_binding.CameraInfo."""
    serial_number: str
    display_name: str
    device_index: int
    width: int = 0
    height: int = 0
    fps: int = 30
    rotation: int = 0
    exposure: int = -4
    enabled: bool = True
    supported_formats: list[tuple[int, int, int]] = field(default_factory=list)

    # Internal: OpenCV capture object
    _capture: Optional[cv2.VideoCapture] = field(default=None, repr=False)

    # For compatibility with code expecting _c_camera
    _c_camera: Optional[object] = field(default=None, repr=False)

    @property
    def supported_resolutions(self) -> list[tuple[int, int]]:
        """Unique (width, height) pairs, sorted largest first."""
        seen = set()
        result = []
        for w, h, _ in self.supported_formats:
            if (w, h) not in seen:
                seen.add((w, h))
                result.append((w, h))
        result.sort(key=lambda r: r[0] * r[1], reverse=True)
        return result

    def fps_for_resolution(self, width: int, height: int) -> list[int]:
        """FPS values available for a given resolution."""
        return sorted({fps for w, h, fps in self.supported_formats if w == width and h == height})


@dataclass
class Frame:
    """A captured frame with numpy array data."""
    pixels: np.ndarray  # BGR format, shape (height, width, 3)
    width: int
    height: int
    timestamp_ns: int = 0
    arrival_ns: int = 0
    corrected_ns: int = 0
    camera_index: int = 0

    @property
    def timestamp_ms(self) -> float:
        return self.timestamp_ns / 1e6

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ns / 1e9


@dataclass
class SyncedFrameSet:
    """A set of synchronized frames from multiple cameras."""
    frames: dict[int, Optional[Frame]]
    sync_index: int
    dropped_mask: int

    @property
    def dropped_cameras(self) -> list[int]:
        dropped = []
        for i in range(16):
            if self.dropped_mask & (1 << i):
                dropped.append(i)
        return dropped


# ============================================================================
# Standard Resolutions to Probe
# ============================================================================

STANDARD_RESOLUTIONS = [
    (1920, 1080),
    (1280, 720),
    (640, 480),
    (640, 360),
    (320, 240),
]

STANDARD_FPS = [30, 60, 15, 24]


# ============================================================================
# Module State
# ============================================================================

_initialized = False
_backend = cv2.CAP_ANY
_native_serials: dict[int, tuple[str, str]] = {}


def init() -> None:
    """Initialize the camera subsystem."""
    global _initialized, _backend, _native_serials
    if _initialized:
        return

    _backend = _get_opencv_backend()
    _native_serials = _get_native_camera_serials()
    _initialized = True


def shutdown() -> None:
    """Shutdown the camera subsystem."""
    global _initialized, _native_serials
    _initialized = False
    _native_serials = {}


def enumerate_cameras(max_cameras: int = 8) -> list[CameraInfo]:
    """
    Enumerate available cameras.

    Uses native library for serial numbers, OpenCV for probing.
    Only probes ports that exist in native enumeration to avoid
    "out of bound" errors.
    """
    if not _initialized:
        init()

    cameras = []

    # Only check ports that native library found (avoids OpenCV errors)
    ports_to_check = list(_native_serials.keys()) if _native_serials else range(max_cameras)

    for port in ports_to_check:
        # Try to open camera
        cap = cv2.VideoCapture(port, _backend)
        if not cap.isOpened():
            continue

        # Try to read a frame to verify it's real
        success = False
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                success = True
                break

        if not success:
            cap.release()
            continue

        # Get serial from native library or generate from port
        if port in _native_serials:
            serial, name = _native_serials[port]
        else:
            serial = f"port_{port}"
            name = f"Camera {port}"

        # Get current resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        # Probe supported formats
        formats = _probe_formats(cap)
        if not formats:
            # At minimum, current format works
            formats = [(width, height, fps)]

        cap.release()

        cameras.append(CameraInfo(
            serial_number=serial,
            display_name=name,
            device_index=port,
            width=width,
            height=height,
            fps=fps,
            supported_formats=formats,
        ))

    return cameras


def _probe_formats(cap: cv2.VideoCapture) -> list[tuple[int, int, int]]:
    """Probe supported resolutions and FPS for a camera."""
    formats = []
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for w, h in STANDARD_RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_w == w and actual_h == h:
            # Resolution supported - check FPS options
            for fps in STANDARD_FPS:
                cap.set(cv2.CAP_PROP_FPS, fps)
                actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
                if actual_fps > 0 and (w, h, actual_fps) not in formats:
                    formats.append((w, h, actual_fps))

    # Restore original resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_h)

    return formats


def open_camera(camera: CameraInfo) -> None:
    """Open a camera for capture."""
    if camera._capture is not None:
        return  # Already open

    cap = cv2.VideoCapture(camera.device_index, _backend)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera.device_index}")

    # Set buffer size to 1 for minimal latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Apply current settings
    if camera.width > 0 and camera.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)

    if camera.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, camera.fps)

    # Apply exposure
    _apply_exposure(cap, camera.exposure)

    camera._capture = cap
    camera._c_camera = cap  # For compatibility

    # Update actual values
    camera.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera.fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30


def close_camera(camera: CameraInfo) -> None:
    """Close a camera."""
    if camera._capture is not None:
        camera._capture.release()
        camera._capture = None
        camera._c_camera = None


def _apply_exposure(cap: cv2.VideoCapture, exposure: int) -> bool:
    """
    Apply exposure setting to camera.

    Copied from MWC multiwebcam/cameras/camera.py lines 96-106:
        if platform.system()=="Windows":
            self.capture.set(cv2.CAP_PROP_EXPOSURE, value)
        else:
            self.capture.set(cv2.CAP_PROP_IOS_DEVICE_EXPOSURE, value)

    Returns the result from cv2.VideoCapture.set()
    """
    if platform.system() == "Windows":
        result = cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    else:
        # macOS: CAP_PROP_IOS_DEVICE_EXPOSURE - may not work for all cameras
        result = cap.set(cv2.CAP_PROP_IOS_DEVICE_EXPOSURE, exposure)

    return result


def set_exposure(camera: CameraInfo, exposure: int) -> bool:
    """Set camera exposure. Returns True if successful."""
    camera.exposure = exposure
    if camera._capture is not None:
        return _apply_exposure(camera._capture, exposure)
    return False


def set_resolution(camera: CameraInfo, width: int, height: int) -> None:
    """Set camera resolution."""
    if camera._capture is not None:
        camera._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.width = int(camera._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        camera.height = int(camera._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        camera.width = width
        camera.height = height


def set_fps(camera: CameraInfo, fps: int) -> None:
    """Set camera frame rate."""
    if camera._capture is not None:
        camera._capture.set(cv2.CAP_PROP_FPS, fps)
        camera.fps = int(camera._capture.get(cv2.CAP_PROP_FPS)) or fps
    else:
        camera.fps = fps


def set_format(camera: CameraInfo, width: int, height: int, fps: int) -> None:
    """Set camera resolution and FPS."""
    set_resolution(camera, width, height)
    set_fps(camera, fps)


def capture_frame(camera: CameraInfo) -> Frame:
    """Capture a single frame."""
    if camera._capture is None:
        raise RuntimeError("Camera not open")

    ret, frame = camera._capture.read()
    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame")

    timestamp_ns = int(time.perf_counter_ns())

    return Frame(
        pixels=frame,
        width=frame.shape[1],
        height=frame.shape[0],
        timestamp_ns=timestamp_ns,
        arrival_ns=timestamp_ns,
        corrected_ns=timestamp_ns,
        camera_index=camera.device_index,
    )


def capture_synced(cameras: list[CameraInfo]) -> SyncedFrameSet:
    """
    Capture frames from multiple cameras as close together as possible.

    Note: This is software sync, not hardware sync. Frames are captured
    sequentially but as fast as possible.
    """
    frames = {}
    dropped_mask = 0
    sync_time = int(time.perf_counter_ns())

    for i, cam in enumerate(cameras):
        if cam._capture is None:
            dropped_mask |= (1 << i)
            frames[i] = None
            continue

        ret, pixels = cam._capture.read()
        if ret and pixels is not None:
            timestamp_ns = int(time.perf_counter_ns())
            frames[i] = Frame(
                pixels=pixels,
                width=pixels.shape[1],
                height=pixels.shape[0],
                timestamp_ns=timestamp_ns,
                arrival_ns=timestamp_ns,
                corrected_ns=timestamp_ns,
                camera_index=cam.device_index,
            )
        else:
            dropped_mask |= (1 << i)
            frames[i] = None

    # Use a module-level sync counter
    global _sync_index
    if '_sync_index' not in globals():
        _sync_index = 0
    _sync_index += 1

    return SyncedFrameSet(
        frames=frames,
        sync_index=_sync_index,
        dropped_mask=dropped_mask,
    )


_sync_index = 0


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
        return capture_frame(self.info)

    @property
    def serial(self) -> str:
        return self.info.serial_number

    @property
    def name(self) -> str:
        return self.info.display_name
