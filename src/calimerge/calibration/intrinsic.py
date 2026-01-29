"""
Intrinsic camera calibration.

Pure functions - no threading, no state. Caller manages frame collection.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..types import CameraIntrinsics, CharucoConfig, PointPacket
from .charuco import ARUCO_DICTIONARIES, create_charuco_board


# ============================================================================
# ChArUco Detection
# ============================================================================


def detect_charuco_points(
    frame: np.ndarray,
    config: CharucoConfig,
    board: cv2.aruco.CharucoBoard | None = None,
) -> PointPacket:
    """
    Detect ChArUco corners in a single frame.

    If corners aren't found, tries detecting in mirrored frame.

    Args:
        frame: BGR image (h, w, 3)
        config: CharucoConfig for the board
        board: Optional pre-created board (created from config if None)

    Returns:
        PointPacket with detected corners
    """
    if board is None:
        board = create_charuco_board(config)

    # Get dictionary
    dict_int = ARUCO_DICTIONARIES.get(config.dictionary, cv2.aruco.DICT_4X4_50)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_int)

    # Convert to grayscale and optionally invert
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if config.inverted:
        gray = ~gray

    # Try to detect corners
    ids, img_loc = _find_corners(gray, dictionary, board, mirror=False)

    # If no corners found, try mirrored image
    if ids.size == 0:
        gray_mirror = cv2.flip(gray, 1)
        ids, img_loc = _find_corners(gray_mirror, dictionary, board, mirror=True)
        if ids.size > 0:
            # Flip x coordinates back
            frame_width = gray.shape[1]
            img_loc[:, 0] = frame_width - img_loc[:, 0]

    # Get object locations for calibration
    if ids.size > 0:
        obj_loc = board.getChessboardCorners()[ids, :]
    else:
        obj_loc = np.array([], dtype=np.float32).reshape(0, 3)
        img_loc = np.array([], dtype=np.float32).reshape(0, 2)

    return PointPacket(
        point_id=ids,
        img_loc=img_loc,
        obj_loc=obj_loc,
    )


def _find_corners(
    gray: np.ndarray,
    dictionary: cv2.aruco.Dictionary,
    board: cv2.aruco.CharucoBoard,
    mirror: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Internal helper to detect ChArUco corners in a grayscale image.

    Args:
        gray: Grayscale image
        dictionary: ArUco dictionary
        board: CharucoBoard
        mirror: Whether image was mirrored

    Returns:
        (ids, img_loc) arrays
    """
    # Detect ArUco markers
    aruco_corners, aruco_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

    if len(aruco_corners) < 2:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32).reshape(0, 2)

    # Interpolate ChArUco corners
    success, img_loc, ids = cv2.aruco.interpolateCornersCharuco(
        aruco_corners, aruco_ids, gray, board
    )

    if not success or ids is None or img_loc is None:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32).reshape(0, 2)

    # Sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
    try:
        img_loc = cv2.cornerSubPix(gray, img_loc, (11, 11), (-1, -1), criteria)
    except cv2.error:
        pass  # Sub-pixel refinement failed, use raw corners

    # Flatten arrays
    ids_flat = ids[:, 0]
    img_loc_flat = img_loc[:, 0, :]

    return ids_flat, img_loc_flat


# ============================================================================
# Calibration
# ============================================================================


def calibrate_intrinsics(
    point_packets: list[PointPacket],
    resolution: tuple[int, int],
    serial_number: str,
    min_corners: int = 4,
) -> CameraIntrinsics:
    """
    Calibrate camera intrinsics from collected ChArUco corners.

    Args:
        point_packets: List of PointPacket from different frames
        resolution: (width, height) of captured frames
        serial_number: Camera serial for identification
        min_corners: Minimum corners required per frame

    Returns:
        CameraIntrinsics with calibration results

    Raises:
        ValueError: If insufficient data for calibration
    """
    # Filter frames with enough corners and valid object locations
    valid_obj = []
    valid_img = []

    for packet in point_packets:
        if packet.point_id is None or packet.point_id.size == 0:
            continue
        if len(packet.point_id) < min_corners:
            continue
        if packet.obj_loc is None or packet.obj_loc.size == 0:
            continue

        valid_obj.append(packet.obj_loc.astype(np.float32))
        valid_img.append(packet.img_loc.astype(np.float32))

    if len(valid_obj) < 3:
        raise ValueError(
            f"Insufficient frames for calibration: {len(valid_obj)} (need at least 3)"
        )

    width, height = resolution

    # Run OpenCV calibration
    error, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        valid_obj,
        valid_img,
        (width, height),
        None,
        None,
    )

    return CameraIntrinsics(
        serial_number=serial_number,
        resolution=resolution,
        matrix=matrix,
        distortion=dist[0],  # Remove extra dimension from cv2.calibrateCamera output
        error=round(error, 4),
        grid_count=len(valid_obj),
    )


def filter_frames_for_calibration(
    point_packets: list[PointPacket],
    target_count: int = 20,
    min_corners: int = 6,
    spacing_frames: int = 10,
) -> list[PointPacket]:
    """
    Select a good subset of frames for calibration.

    Attempts to select frames that are well-distributed across the video
    with sufficient corner detection.

    Args:
        point_packets: All detected point packets (ordered by frame)
        target_count: Target number of frames to select
        min_corners: Minimum corners required per frame
        spacing_frames: Minimum frames between selected frames

    Returns:
        Filtered list of PointPackets
    """
    import random

    # Filter to only frames with enough corners
    valid_frames = [
        (i, p)
        for i, p in enumerate(point_packets)
        if p.point_id is not None
        and p.point_id.size >= min_corners
        and p.obj_loc is not None
    ]

    if len(valid_frames) <= target_count:
        return [p for _, p in valid_frames]

    # Try to select well-spaced frames
    selected = []
    selected_indices = set()

    # First pass: select evenly spaced frames
    step = len(valid_frames) // target_count
    for i in range(0, len(valid_frames), max(1, step)):
        if len(selected) >= target_count:
            break
        frame_idx, packet = valid_frames[i]
        selected.append(packet)
        selected_indices.add(frame_idx)

    # Backfill if needed with random selection
    remaining = [
        (i, p)
        for i, p in valid_frames
        if i not in selected_indices
    ]

    if remaining and len(selected) < target_count:
        needed = target_count - len(selected)
        random_picks = random.sample(remaining, min(needed, len(remaining)))
        selected.extend([p for _, p in random_picks])

    return selected


def compute_reprojection_error(
    point_packet: PointPacket,
    intrinsics: CameraIntrinsics,
) -> float | None:
    """
    Compute reprojection error for a single frame.

    Args:
        point_packet: Detected points with obj_loc
        intrinsics: Camera intrinsics

    Returns:
        RMS reprojection error in pixels, or None if can't compute
    """
    if point_packet.obj_loc is None or point_packet.obj_loc.size == 0:
        return None
    if point_packet.img_loc is None or point_packet.img_loc.size == 0:
        return None

    # Use solvePnP to get pose
    success, rvec, tvec = cv2.solvePnP(
        point_packet.obj_loc,
        point_packet.img_loc,
        intrinsics.matrix,
        intrinsics.distortion,
    )

    if not success:
        return None

    # Project points back
    projected, _ = cv2.projectPoints(
        point_packet.obj_loc,
        rvec,
        tvec,
        intrinsics.matrix,
        intrinsics.distortion,
    )
    projected = projected[:, 0, :]

    # Compute error
    error = np.sqrt(np.mean((point_packet.img_loc - projected) ** 2))
    return float(error)
