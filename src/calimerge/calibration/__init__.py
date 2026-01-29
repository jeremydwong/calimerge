"""
Calibration module for calimerge.

All functions are pure - they take dataclasses and return dataclasses.
No threading, no state management. Caller handles concurrency.
"""

from .charuco import (
    ARUCO_DICTIONARIES,
    create_charuco_board,
    generate_board_image,
    get_charuco_object_points,
    get_connected_corners,
)

from .intrinsic import (
    detect_charuco_points,
    calibrate_intrinsics,
)

from .extrinsic import (
    stereo_calibrate_pair,
    run_bundle_adjustment,
    compute_initial_extrinsics,
)

__all__ = [
    # Charuco
    "ARUCO_DICTIONARIES",
    "create_charuco_board",
    "generate_board_image",
    "get_charuco_object_points",
    "get_connected_corners",
    # Intrinsic
    "detect_charuco_points",
    "calibrate_intrinsics",
    # Extrinsic
    "stereo_calibrate_pair",
    "run_bundle_adjustment",
    "compute_initial_extrinsics",
]
