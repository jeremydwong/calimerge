"""
SkeletonViewWidget - oblique three-quarter projection of a 3D skeleton.

Coordinate conventions
----------------------
World space  : camera-vision Y-down (Y increases downward in the scene).
View space   : set by _view_transform (4x4 world→view).
               After "Rotate to Human":
                 view X = foot-to-foot (rightward)
                 view Y = forward/depth
                 view Z = body-up (head is at large +Z)
Screen space : standard Qt (sy increases downward).

Projection
----------
Oblique three-quarter projection: view X + Y-shear maps to screen X,
view Z + Y-shear maps to screen Y (inverted for Y-up display).
"""

from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen, QFont


# SynthPose 52-keypoint skeleton connections (draws whatever keypoints are available)
_SKELETON = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Neck / shoulders
    (0, 17), (17, 5), (17, 6), (17, 48),
    # Arms
    (5, 7), (7, 9), (6, 8), (8, 10),
    (7, 20), (7, 22), (8, 21), (8, 23),
    (9, 24), (9, 26), (10, 25), (10, 27),
    # Torso
    (5, 11), (6, 12), (11, 12),
    (48, 51), (51, 50), (50, 49),
    (49, 28), (49, 29), (28, 30), (29, 31),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16),
    (13, 32), (13, 34), (14, 33), (14, 35),
    (15, 36), (15, 38), (16, 37), (16, 39),
    # Feet
    (15, 46), (16, 47), (15, 40), (16, 41),
    (40, 42), (41, 43), (42, 44), (43, 45),
    # Fallback COCO-only connections (used when SynthPose kps 17+ are absent)
    (5, 6),
]

# Per-person colors (limbs + keypoints use the same color per person)
_PERSON_COLORS = [
    QColor(220, 220, 220),  # 0  white
    QColor(80,  220, 120),  # 1  green
    QColor(80,  160, 255),  # 2  blue
    QColor(255, 180,  60),  # 3  orange
    QColor(220,  80, 220),  # 4  purple
]

# Show ±_VIEW_HALF in the narrower widget dimension.
_VIEW_HALF = 2.5   # metres

# Oblique projection coefficients for view Y (depth) axis.
# Positive view-Y (behind person) shifts screen LEFT and UP.
_OBLIQUE_X = 0.45   # depth shifts screen X  (positive = left when depth is positive)
_OBLIQUE_Z = 0.30   # depth shifts screen Z  (positive = upward when depth is positive)

# When has_origin=True, centre the static view at this height above the floor (view Z).
_FIXED_CZ = 1.0    # metres
_FIXED_CX = 0.0    # metres

# Default transform: world X → view X, world Z (depth) → view Y, world -Y (up) → view Z.
# Gives a sensible three-quarter display before "Rotate to Human" is pressed.
_DEFAULT_TRANSFORM = np.array([
    [1,  0,  0,  0],
    [0,  0,  1,  0],
    [0, -1,  0,  0],
    [0,  0,  0,  1],
], dtype=float)


class SkeletonViewWidget(QWidget):
    """Displays a 3D skeleton with an oblique three-quarter projection."""

    # Maximum age (in frames) that a held keypoint survives before being dropped.
    # After this many frames without an update, the keypoint is considered lost.
    HOLD_MAX_AGE = 10

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(200, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # list of persons; each person is list[np.ndarray(3,) | None] length 17
        self._persons: list[list] = []
        # Parallel structure: per-person, per-keypoint "age" (frames since last seen).
        # 0 = fresh, ≥1 = held from previous frame, <0 unused.
        self._persons_age: list[list[int]] = []
        self._message: str = "No detection"
        self._view_transform: np.ndarray = _DEFAULT_TRANSFORM.copy()
        # True once "Set Origin at L_Ankle" has been applied → static view + floor grid.
        self._has_origin: bool = False

    # ── Public API ──

    def update_keypoints(self, persons: list) -> None:
        """Set raw world-space keypoints for one or more persons and repaint.

        Missing keypoints (None) are held from the previous frame at increasing
        age so the display dims rather than flickers. After HOLD_MAX_AGE frames
        without an update the held value is dropped.

        Args:
            persons: list[list[np.ndarray(3,) | None]]
                     Outer list = persons; inner list = 17 COCO-17 keypoints.
        """
        new_persons: list[list] = []
        new_ages: list[list[int]] = []

        for p_idx, incoming in enumerate(persons):
            # Recover previous person's buffer (may not exist yet)
            prev_person = self._persons[p_idx] if p_idx < len(self._persons) else []
            prev_age = self._persons_age[p_idx] if p_idx < len(self._persons_age) else []

            held_person: list = []
            held_age: list[int] = []
            for k_idx, kp in enumerate(incoming):
                if kp is not None:
                    held_person.append(kp)
                    held_age.append(0)
                else:
                    # Try to hold the previous value if still fresh enough
                    if k_idx < len(prev_person) and prev_person[k_idx] is not None:
                        age = prev_age[k_idx] + 1 if k_idx < len(prev_age) else 1
                        if age <= self.HOLD_MAX_AGE:
                            held_person.append(prev_person[k_idx])
                            held_age.append(age)
                            continue
                    held_person.append(None)
                    held_age.append(-1)
            new_persons.append(held_person)
            new_ages.append(held_age)

        self._persons = new_persons
        self._persons_age = new_ages
        self._message = ""
        self.update()

    def set_view_transform(self, mat: np.ndarray, has_origin: bool = False) -> None:
        """Set 4x4 world→view transform and repaint.

        Args:
            mat: 4x4 matrix; after "Rotate to Human" columns are
                 [X_foot-to-foot, Y_forward, Z_body-up].
            has_origin: True when a floor origin has been zeroed.
                        Switches to static centre + enables floor grid.
        """
        self._view_transform = np.array(mat, dtype=float)
        self._has_origin = has_origin
        self.update()

    def set_message(self, msg: str) -> None:
        self._persons = []
        self._persons_age = []
        self._message = msg
        self.update()

    def clear(self) -> None:
        self._persons = []
        self._persons_age = []
        self._message = "No detection"
        self._has_origin = False
        self._view_transform = _DEFAULT_TRANSFORM.copy()
        self.update()

    def get_keypoints(self) -> list:
        """Return person 0's RAW world-space keypoints (for rotate/zero buttons)."""
        if self._persons:
            return list(self._persons[0])
        return []

    # ── Painting ──

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        margin = 20
        title_h = 18   # pixels reserved for title at top

        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))

        has_kps = bool(self._persons) and any(
            any(k is not None for k in p) for p in self._persons
        )

        if not has_kps:
            msg = self._message if self._message else "No detection"
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont("monospace", 10))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, msg)
            painter.end()
            return

        # ── Project all persons: world → view space ──
        def project_kps(kps: list) -> list:
            result = []
            for kp in kps:
                if kp is None:
                    result.append(None)
                    continue
                pt = np.array(kp, dtype=float)
                if pt.shape != (3,):
                    result.append(None)
                    continue
                v = (self._view_transform @ np.append(pt, 1.0))[:3]
                result.append((float(v[0]), float(v[1]), float(v[2])))
            return result

        all_view_pts = [project_kps(p) for p in self._persons]

        # Collect all valid 3D view points for centering
        all_valid = [vp for pvp in all_view_pts for vp in pvp if vp is not None]
        if not all_valid:
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont("monospace", 10))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No detection")
            painter.end()
            return

        # ── Scale ──
        avail = min(w - 2 * margin, h - 2 * margin - title_h)
        scale = avail / (2 * _VIEW_HALF)

        # ── Centre of view ──
        # Static (has_origin): 1 m above zeroed floor, centred on X=0.
        # Tracking (no origin): centroid over all persons in XZ.
        if self._has_origin:
            cx, cz = _FIXED_CX, _FIXED_CZ
        else:
            cx = sum(p[0] for p in all_valid) / len(all_valid)
            cz = sum(p[2] for p in all_valid) / len(all_valid)

        # Screen centre (slightly below mid to leave title room)
        scx = w / 2
        scy = h / 2 + title_h // 2

        def to_screen(vx: float, vy: float, vz: float) -> tuple[int, int]:
            # Oblique: view X + depth-shear → screen X
            #          view Z + depth-shear → screen Y (inverted: Z-up)
            sx = int(scx + ((vx - vy * _OBLIQUE_X) - cx) * scale)
            sy = int(scy - ((vz + vy * _OBLIQUE_Z) - cz) * scale)
            return sx, sy

        # ── Floor grid (only when origin is zeroed) ──
        if self._has_origin:
            # Height markers: Z = 0, 0.5, 1.0, 1.5, 2.0 m (at vy=0)
            for gh in (0.0, 0.5, 1.0, 1.5, 2.0):
                # Front edge of grid line (vy = 0)
                sx0, sy0 = to_screen(-_VIEW_HALF, 0.0, gh)
                sx1, sy1 = to_screen(+_VIEW_HALF, 0.0, gh)
                if gh == 0.0:
                    painter.setPen(QPen(QColor(110, 110, 110), 1))
                else:
                    pen = QPen(QColor(60, 60, 60), 1)
                    pen.setStyle(Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                painter.drawLine(sx0, sy0, sx1, sy1)
                if gh > 0:
                    painter.setPen(QColor(65, 65, 65))
                    painter.setFont(QFont("monospace", 7))
                    painter.drawText(sx1 + 3, sy1 + 4, f"{gh:.1f}m")

            # Depth lines on floor (Z=0): draw X-axis and Y-axis at floor
            # X axis (foot-to-foot at floor level)
            ax0, ay0 = to_screen(0.0, 0.0, 0.0)
            ax1, ay1 = to_screen(1.0, 0.0, 0.0)
            painter.setPen(QPen(QColor(180, 60, 60), 2))   # red = X
            painter.drawLine(ax0, ay0, ax1, ay1)

            # Z axis (body up)
            az1, az2 = to_screen(0.0, 0.0, 1.0)
            painter.setPen(QPen(QColor(60, 60, 180), 2))   # blue = Z (up)
            painter.drawLine(ax0, ay0, az1, az2)

            # Y axis (depth / forward)
            ay_1, ay_2 = to_screen(0.0, 1.0, 0.0)
            painter.setPen(QPen(QColor(60, 160, 60), 2))   # green = Y (forward)
            painter.drawLine(ax0, ay0, ay_1, ay_2)

        # ── Skeletons — one per person, each in a distinct colour ──
        def alpha_for_age(age: int) -> int:
            """Linearly fade from 255 (fresh) to ~60 (max held age)."""
            if age <= 0:
                return 255
            # Fade from 255 to 60 across HOLD_MAX_AGE frames
            max_age = max(1, self.HOLD_MAX_AGE)
            frac = min(1.0, age / max_age)
            return int(255 - frac * (255 - 60))

        for person_idx, view_pts in enumerate(all_view_pts):
            base_color = _PERSON_COLORS[person_idx % len(_PERSON_COLORS)]
            n = len(view_pts)
            ages = self._persons_age[person_idx] if person_idx < len(self._persons_age) else []

            def age_of(k: int) -> int:
                return ages[k] if k < len(ages) else 0

            # Limbs — dim to the older of the two endpoints
            for i, j in _SKELETON:
                if i >= n or j >= n:
                    continue
                if view_pts[i] is None or view_pts[j] is None:
                    continue
                limb_age = max(age_of(i), age_of(j))
                a = alpha_for_age(limb_age)
                limb_color = QColor(base_color.red(), base_color.green(), base_color.blue(), a)
                painter.setPen(QPen(limb_color, 2))
                painter.drawLine(*to_screen(*view_pts[i]), *to_screen(*view_pts[j]))

            # Keypoints — dim each by its own age
            for k_idx, vp in enumerate(view_pts):
                if vp is None:
                    continue
                a = alpha_for_age(age_of(k_idx))
                dot_color = QColor(base_color.red(), base_color.green(), base_color.blue(), a)
                outline_color = QColor(30, 30, 30, a)
                painter.setBrush(dot_color)
                painter.setPen(QPen(outline_color, 1))
                sp = to_screen(*vp)
                painter.drawEllipse(sp[0] - 4, sp[1] - 4, 8, 8)

        # ── Title: person 0's L_Ankle ──
        p0_view = all_view_pts[0] if all_view_pts else []
        p0_raw  = self._persons[0] if self._persons else []
        l_ankle_vp  = p0_view[15] if len(p0_view) > 15 else None
        l_ankle_raw = p0_raw[15]  if len(p0_raw) > 15 else None

        n_str = f"  [{len(self._persons)} person{'s' if len(self._persons) != 1 else ''}]"
        if self._has_origin and l_ankle_vp is not None:
            vx, vy, vz = l_ankle_vp
            title = f"L_Ankle  x={vx:.2f}  y={vy:.2f}  z={vz:.2f} m{n_str}"
        elif l_ankle_raw is not None:
            rx, ry, rz = l_ankle_raw
            title = f"L_Ankle (world)  x={rx:.2f}  y={ry:.2f}  z={rz:.2f} m{n_str}"
        else:
            title = f"L_Ankle: not detected{n_str}"

        painter.setPen(QColor(160, 160, 160))
        painter.setFont(QFont("monospace", 8))
        painter.drawText(margin, margin + 12, title)

        painter.end()
