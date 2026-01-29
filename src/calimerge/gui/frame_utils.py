"""
Frame conversion utilities for PySide6 display.
"""

import numpy as np
from PySide6.QtGui import QImage, QPixmap


def bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    """
    Convert BGR numpy array to QPixmap for display.

    Args:
        frame: (H, W, 3) BGR uint8 array from OpenCV/camera

    Returns:
        QPixmap ready for QLabel.setPixmap()
    """
    if frame is None or frame.size == 0:
        return QPixmap()

    height, width = frame.shape[:2]

    # BGR to RGB
    rgb = frame[:, :, ::-1].copy()

    # Create QImage (must use contiguous array)
    image = QImage(
        rgb.data,
        width,
        height,
        width * 3,  # bytes per line
        QImage.Format.Format_RGB888,
    )

    return QPixmap.fromImage(image)


def gray_to_pixmap(frame: np.ndarray) -> QPixmap:
    """
    Convert grayscale numpy array to QPixmap.

    Args:
        frame: (H, W) uint8 array

    Returns:
        QPixmap ready for QLabel.setPixmap()
    """
    if frame is None or frame.size == 0:
        return QPixmap()

    height, width = frame.shape[:2]

    image = QImage(
        frame.data,
        width,
        height,
        width,  # bytes per line
        QImage.Format.Format_Grayscale8,
    )

    return QPixmap.fromImage(image)


def scale_pixmap_to_fit(pixmap: QPixmap, max_width: int, max_height: int) -> QPixmap:
    """
    Scale pixmap to fit within bounds while preserving aspect ratio.

    Args:
        pixmap: Source pixmap
        max_width: Maximum width
        max_height: Maximum height

    Returns:
        Scaled QPixmap
    """
    if pixmap.isNull():
        return pixmap

    return pixmap.scaled(
        max_width,
        max_height,
        aspectMode=Qt.AspectRatioMode.KeepAspectRatio,
        transformMode=Qt.TransformationMode.SmoothTransformation,
    )


# Import Qt after function definitions to avoid circular import issues
from PySide6.QtCore import Qt
