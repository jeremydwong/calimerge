"""
ChArUco board creation and utilities.

Pure functions - no classes, no state.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import cv2
import numpy as np

from ..types import CharucoConfig


# ============================================================================
# ArUco Dictionary Reference
# ============================================================================

ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


# ============================================================================
# Board Creation
# ============================================================================


def create_charuco_board(config: CharucoConfig) -> cv2.aruco.CharucoBoard:
    """
    Create an OpenCV CharucoBoard from configuration.

    Args:
        config: CharucoConfig with board parameters

    Returns:
        cv2.aruco.CharucoBoard object
    """
    # Get dictionary
    dict_int = ARUCO_DICTIONARIES.get(config.dictionary, cv2.aruco.DICT_4X4_50)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_int)

    # Create board
    board = cv2.aruco.CharucoBoard(
        size=(config.columns, config.rows),
        squareLength=config.square_size_m,
        markerLength=config.marker_size_m,
        dictionary=dictionary,
    )

    board.setLegacyPattern(config.legacy_pattern)

    return board


def generate_board_image(
    config: CharucoConfig,
    width: int = 1000,
    height: int = 1000,
) -> np.ndarray:
    """
    Generate an image of the ChArUco board.

    Args:
        config: CharucoConfig with board parameters
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        BGR image as numpy array
    """
    board = create_charuco_board(config)
    img = board.generateImage((width, height))

    if config.inverted:
        img = ~img

    # Convert to BGR if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def get_charuco_object_points(board: cv2.aruco.CharucoBoard) -> np.ndarray:
    """
    Get the 3D object points for all corners on the board.

    Args:
        board: OpenCV CharucoBoard

    Returns:
        (n, 3) array of corner positions in board frame
    """
    return board.getChessboardCorners()


def get_connected_corners(board: cv2.aruco.CharucoBoard) -> set[tuple[int, int]]:
    """
    Get pairs of corner IDs that should be connected to form a grid.

    Useful for visualization and for computing distance constraints.

    Args:
        board: OpenCV CharucoBoard

    Returns:
        Set of (corner_id_a, corner_id_b) tuples
    """
    corners = board.getChessboardCorners()
    corners_x = corners[:, 0]
    corners_y = corners[:, 1]

    x_set = set(corners_x)
    y_set = set(corners_y)

    lines = defaultdict(list)

    # Group corners by their x position (vertical lines)
    for x_line in x_set:
        for corner_id, (x, y) in enumerate(zip(corners_x, corners_y)):
            if x == x_line:
                lines[f"x_{x_line}"].append(corner_id)

    # Group corners by their y position (horizontal lines)
    for y_line in y_set:
        for corner_id, (x, y) in enumerate(zip(corners_x, corners_y)):
            if y == y_line:
                lines[f"y_{y_line}"].append(corner_id)

    # Create pairs of adjacent corners
    connected = set()
    for line_corners in lines.values():
        for pair in combinations(line_corners, 2):
            connected.add(pair)

    return connected


def get_corner_distances(board: cv2.aruco.CharucoBoard) -> dict[tuple[int, int], float]:
    """
    Get expected distances between all pairs of connected corners.

    Useful for quality control during calibration.

    Args:
        board: OpenCV CharucoBoard

    Returns:
        Dict mapping (corner_a, corner_b) to expected distance in meters
    """
    corners = board.getChessboardCorners()
    connected = get_connected_corners(board)

    distances = {}
    for a, b in connected:
        dist = np.linalg.norm(corners[a] - corners[b])
        distances[(a, b)] = float(dist)

    return distances
