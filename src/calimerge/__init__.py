# calimerge - Unified multi-camera motion capture

__version__ = "0.1.0"

# Camera bindings (C++ module)
from calimerge.camera_binding import (
    init,
    shutdown,
    enumerate_cameras,
    open_camera,
    close_camera,
    capture_frame,
    capture_synced,
    CameraInfo,
    Frame,
    SyncedFrameSet,
    Camera,
)

# Recording
from calimerge.video_recorder import (
    MultiCameraRecorder,
    record_session,
)

# Core types
from calimerge.types import (
    CameraConfig,
    CameraIntrinsics,
    CameraExtrinsics,
    CalibratedCamera,
    CharucoConfig,
    PointPacket,
    FramePoints,
    SyncedPoints,
    XYZPoints,
    ProjectConfig,
)

# Configuration
from calimerge.config import (
    load_project_config,
    save_project_config,
    save_intrinsics,
    load_intrinsics,
)

# Triangulation
from calimerge.triangulation import (
    triangulate_frame,
    triangulate_points,
    triangulate_all,
    undistort_points,
)

__all__ = [
    # Lifecycle
    "init",
    "shutdown",
    # Camera management
    "enumerate_cameras",
    "open_camera",
    "close_camera",
    # Capture
    "capture_frame",
    "capture_synced",
    # Camera data classes
    "CameraInfo",
    "Frame",
    "SyncedFrameSet",
    "Camera",
    # Recording
    "MultiCameraRecorder",
    "record_session",
    # Core types
    "CameraConfig",
    "CameraIntrinsics",
    "CameraExtrinsics",
    "CalibratedCamera",
    "CharucoConfig",
    "PointPacket",
    "FramePoints",
    "SyncedPoints",
    "XYZPoints",
    "ProjectConfig",
    # Configuration
    "load_project_config",
    "save_project_config",
    "save_intrinsics",
    "load_intrinsics",
    # Triangulation
    "triangulate_frame",
    "triangulate_points",
    "triangulate_all",
    "undistort_points",
]
