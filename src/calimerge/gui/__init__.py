"""
Calimerge unified GUI.

Tabbed interface for multi-camera motion capture:
- Cameras: Detection, preview, configuration
- Record: Synchronized video capture
- Intrinsic: Per-camera lens calibration
- Extrinsic: Multi-camera spatial calibration
- Process: Tracking and triangulation
"""

from .main import MainWindow

__all__ = ["MainWindow"]
