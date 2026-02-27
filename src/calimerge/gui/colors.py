"""
Global camera color palette.

Each camera port index maps to a consistent color used across the entire GUI:
camera grid borders, FPS graph lines, labels, etc.

Edit CAMERA_COLORS to change the palette. Up to MAX_CAMERAS are supported.
"""

from PySide6.QtGui import QColor

MAX_CAMERAS = 8

# Ordered list of camera colors — index 0 = camera 0, etc.
# Chosen for readability on dark backgrounds and mutual distinguishability.
CAMERA_COLORS: list[QColor] = [
    QColor(80, 200, 120),    # 0: green
    QColor(100, 160, 255),   # 1: blue
    QColor(255, 180, 80),    # 2: orange
    QColor(220, 100, 220),   # 3: purple
    QColor(255, 100, 100),   # 4: red
    QColor(100, 220, 220),   # 5: cyan
    QColor(255, 220, 80),    # 6: yellow
    QColor(180, 140, 255),   # 7: lavender
]


def camera_color(port: int) -> QColor:
    """Get the color for a camera port, cycling if port >= MAX_CAMERAS."""
    return CAMERA_COLORS[port % len(CAMERA_COLORS)]


def camera_color_hex(port: int) -> str:
    """Get the hex color string (e.g. '#50c878') for a camera port."""
    return camera_color(port).name()
