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


def create_charuco_pdf(config: CharucoConfig, filename: str | Path) -> None:
    """
    Generate a PDF of the ChArUco board at exact physical dimensions.

    The board is rendered at its true size so that when printed at 100% scale,
    each square measures exactly ``config.square_size_cm`` cm.

    Args:
        config: CharucoConfig with board parameters
        filename: Output PDF path
    """
    from pathlib import Path

    filename = Path(filename)

    # Board physical dimensions in cm
    board_width_cm = config.columns * config.square_size_cm
    board_height_cm = config.rows * config.square_size_cm

    # Convert cm to points (1 inch = 72 pt, 1 inch = 2.54 cm)
    cm_to_pt = 72.0 / 2.54
    board_width_pt = board_width_cm * cm_to_pt
    board_height_pt = board_height_cm * cm_to_pt

    # Add margin (1 cm each side)
    margin_pt = 1.0 * cm_to_pt
    page_width_pt = board_width_pt + 2 * margin_pt
    page_height_pt = board_height_pt + 2 * margin_pt

    # Render at high DPI (300 DPI)
    dpi = 300
    px_per_pt = dpi / 72.0
    img_w = int(board_width_pt * px_per_pt)
    img_h = int(board_height_pt * px_per_pt)

    img = generate_board_image(config, width=img_w, height=img_h)
    # Convert BGR to RGB for PDF
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Encode as JPEG for smaller file size
    _, jpeg_buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    jpeg_bytes = jpeg_buf.tobytes()

    # Build a minimal single-page PDF with the image placed at exact dimensions
    # Using raw PDF construction to avoid extra dependencies
    _write_pdf_with_image(
        filename, jpeg_bytes, img_w, img_h,
        page_width_pt, page_height_pt,
        board_width_pt, board_height_pt,
        margin_pt,
        config,
    )


def _write_pdf_with_image(
    filename,
    jpeg_bytes: bytes,
    img_w: int, img_h: int,
    page_w: float, page_h: float,
    board_w: float, board_h: float,
    margin: float,
    config,
) -> None:
    """Write a minimal PDF containing the board image at exact dimensions."""
    from pathlib import Path

    offsets = []
    parts = []

    def obj(content: str) -> bytes:
        offsets.append(len(b"".join(parts)))
        return content.encode("latin-1")

    # Header
    parts.append(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    # 1: Catalog
    parts.append(obj("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"))

    # 2: Pages
    parts.append(obj("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"))

    # 3: Page
    label_text = (
        f"ChArUco {config.columns}x{config.rows}, "
        f"square={config.square_size_cm}cm, "
        f"{config.dictionary}"
        f"{', inverted' if config.inverted else ''}"
    )
    # Total page height with extra space for label
    label_space = 14  # points
    total_page_h = page_h + label_space
    parts.append(obj(
        f"3 0 obj\n"
        f"<< /Type /Page /Parent 2 0 R "
        f"/MediaBox [0 0 {page_w:.2f} {total_page_h:.2f}] "
        f"/Contents 4 0 R /Resources << /XObject << /Img 5 0 R >> "
        f"/Font << /F1 6 0 R >> >> >>\n"
        f"endobj\n"
    ))

    # 4: Content stream — place image and label
    # Image placed at (margin, margin + label_space) with exact board dimensions
    stream = (
        f"BT /F1 8 Tf {margin:.2f} {label_space - 2:.2f} Td ({label_text}) Tj ET\n"
        f"q {board_w:.4f} 0 0 {board_h:.4f} {margin:.4f} {margin + label_space:.4f} cm "
        f"/Img Do Q\n"
    )
    stream_bytes = stream.encode("latin-1")
    parts.append(obj(
        f"4 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n"
    ))
    parts.append(stream_bytes)
    parts.append(b"\nendstream\nendobj\n")

    # 5: Image XObject
    parts.append(obj(
        f"5 0 obj\n"
        f"<< /Type /XObject /Subtype /Image /Width {img_w} /Height {img_h} "
        f"/ColorSpace /DeviceRGB /BitsPerComponent 8 "
        f"/Filter /DCTDecode /Length {len(jpeg_bytes)} >>\n"
        f"stream\n"
    ))
    parts.append(jpeg_bytes)
    parts.append(b"\nendstream\nendobj\n")

    # 6: Font (Helvetica)
    parts.append(obj(
        "6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    ))

    # Cross-reference table
    xref_offset = len(b"".join(parts))
    xref_lines = [f"xref\n0 {len(offsets) + 1}\n0000000000 65535 f \n"]
    for off in offsets:
        xref_lines.append(f"{off:010d} 00000 n \n")
    parts.append("".join(xref_lines).encode("latin-1"))

    parts.append(
        f"trailer\n<< /Size {len(offsets) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
    )

    Path(filename).write_bytes(b"".join(parts))


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
