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

import time
from collections import deque

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from calimerge.tracking.footstep_detector import FootPlacement, FootstepDetector

# SynthPose 52-keypoint skeleton connections.
# Convention: r_ = right side body, l_ = left side body.
# 20=r_lelbow (right lateral elbow) → connects to 8 (R_Elbow), NOT 7.
_SKELETON = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Neck / shoulders
    (0, 17), (17, 5), (17, 6), (17, 48),
    (5, 19), (6, 18),                         # shoulder landmarks (l/r)
    # Left arm: 5→7→9 with landmarks
    (5, 7), (7, 9),
    (7, 21), (7, 23),                         # l_lelbow, l_melbow
    (9, 25), (9, 27),                         # l_lwrist, l_mwrist
    # Right arm: 6→8→10 with landmarks
    (6, 8), (8, 10),
    (8, 20), (8, 22),                         # r_lelbow, r_melbow
    (10, 24), (10, 26),                       # r_lwrist, r_mwrist
    # Torso
    (5, 11), (6, 12), (11, 12),
    (48, 51), (51, 50), (50, 49),             # spine: C7→T6→T11→L2
    (49, 29), (49, 28),                       # L2→ASIS (l/r)
    (29, 31), (28, 30),                       # ASIS→PSIS (l/r)
    # Left leg: 11→13→15 with landmarks
    (11, 13), (13, 15),
    (13, 33), (13, 35),                       # l_knee, l_mknee
    (15, 37), (15, 39),                       # l_ankle, l_mankle
    # Right leg: 12→14→16 with landmarks
    (12, 14), (14, 16),
    (14, 32), (14, 34),                       # r_knee, r_mknee
    (16, 36), (16, 38),                       # r_ankle, r_mankle
    # Left foot
    (15, 46), (15, 41),                       # l_calc, l_5meta
    (41, 43), (43, 45),                       # l_5meta→l_toe→l_big_toe
    # Right foot
    (16, 47), (16, 40),                       # r_calc, r_5meta
    (40, 42), (42, 44),                       # r_5meta→r_toe→r_big_toe
    # Fallback COCO (used when SynthPose kps 17+ are absent)
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


# ── Footstep overlay constants ─────────────────────────────────────────
# How many recent placements to keep per foot.
_FOOTSTEP_HISTORY = 10
# Body keypoint indices (SynthPose-52) for the ankle joints.
_LEFT_ANKLE_IDX = 15
_RIGHT_ANKLE_IDX = 16
# Footstep marker styling. Match the body's left/right palette so the dots
# read as "this is where the left foot landed".
_LEFT_FOOT_COLOR = QColor(80, 160, 255)    # blue  — left
_RIGHT_FOOT_COLOR = QColor(255, 80, 80)    # red   — right
# Squares are larger than the old discs but transparent enough to overlay
# without obscuring the floor grid behind them.
_FOOTSTEP_MAX_ALPHA = 140
_FOOTSTEP_MIN_ALPHA = 25
# Half-side of the square in screen pixels (so total side = 2*this).
_FOOTSTEP_HALF_PX = 9


# ── Body laterality (SynthPose-52) ─────────────────────────────────────
# Each body keypoint is classified as Left ("L"), Right ("R"), or Center
# ("C"). Used to colour the skeleton so the right side draws red and the
# left side blue regardless of which person is in frame.
_BODY_SIDE_LEFT = "L"
_BODY_SIDE_RIGHT = "R"
_BODY_SIDE_CENTER = "C"


def _build_laterality_table() -> dict[int, str]:
    """Classify each SynthPose-52 index by its anatomical side.

    Driven off the marker name string: "L_*" / "l_*" → left, "R_*" / "r_*"
    → right, otherwise center (spine, head midline, sternum). This is built
    once at import time so paintEvent stays cheap.
    """
    from ...tracking.markers import SYNTHPOSE_MARKERS
    out: dict[int, str] = {}
    for idx, name in SYNTHPOSE_MARKERS.items():
        # Names use a mix of "L_Hip" and "l_5meta" — both forms classify
        # left, the cap-R / lowercase-r forms classify right, anything else
        # is center.
        if name.startswith(("L_", "l_")):
            out[idx] = _BODY_SIDE_LEFT
        elif name.startswith(("R_", "r_")):
            out[idx] = _BODY_SIDE_RIGHT
        else:
            out[idx] = _BODY_SIDE_CENTER
    return out


_LATERALITY = _build_laterality_table()

# Body colours per side. Keep the names distinct from the foot constants
# so the drawing code reads explicitly even though they happen to match.
_BODY_LEFT_COLOR = QColor(80, 160, 255)    # blue
_BODY_RIGHT_COLOR = QColor(255, 80, 80)    # red
_BODY_CENTER_COLOR = QColor(180, 180, 180) # gray


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

        # Footstep overlay state — only meaningful once has_origin=True (so
        # ankle z is in body-up coordinates and the floor is at z=0).
        self._left_detector = FootstepDetector()
        self._right_detector = FootstepDetector()
        self._left_steps: deque[FootPlacement] = deque(maxlen=_FOOTSTEP_HISTORY)
        self._right_steps: deque[FootPlacement] = deque(maxlen=_FOOTSTEP_HISTORY)
        # Wall-clock anchor — the detector wants strictly increasing timestamps;
        # we use perf_counter so it survives clock changes.
        self._t0 = time.perf_counter()

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
        # Update footstep history (in body coords). Skipped when origin
        # has not been zeroed because z=0 is not a meaningful floor yet.
        if self._has_origin:
            self._update_footsteps_from_persons(new_persons)
        self.update()

    def set_view_transform(self, mat: np.ndarray, has_origin: bool = False) -> None:
        """Set 4x4 world→view transform and repaint.

        Args:
            mat: 4x4 matrix; after "Rotate to Human" columns are
                 [X_foot-to-foot, Y_forward, Z_body-up].
            has_origin: True when a floor origin has been zeroed.
                        Switches to static centre + enables floor grid.
        """
        prev_has_origin = self._has_origin
        self._view_transform = np.array(mat, dtype=float)
        self._has_origin = has_origin
        # Re-zeroing or losing the origin should drop stale footsteps —
        # they're tied to the previous body frame and would be misplaced.
        if has_origin != prev_has_origin or has_origin:
            self.clear_footsteps()
        self.update()

    def set_message(self, msg: str) -> None:
        self._persons = []
        self._persons_age = []
        self._message = msg
        self.update()

    def clear(self) -> None:
        # Only clears the displayed data — NOT the view transform. The
        # transform is per-model state owned by the parent page; it
        # survives detection stop/start and is updated explicitly via
        # set_view_transform when the user picks a different model. (We
        # used to reset _view_transform here, which silently wiped the
        # rotate-to-human + zero settings every time the user switched
        # backends, so e.g. flipping to CUDA looked like the transform
        # had never been engaged.)
        self._persons = []
        self._persons_age = []
        self._message = "No detection"
        self.clear_footsteps()
        self.update()

    def reset_view_transform(self) -> None:
        """Explicitly reset the view transform to identity. Use this when
        the user wants to drop the saved orientation, not when stopping
        detection."""
        self._has_origin = False
        self._view_transform = _DEFAULT_TRANSFORM.copy()
        self.update()

    def clear_footsteps(self) -> None:
        """Drop both feet's placement history and reset detector state."""
        self._left_detector.reset()
        self._right_detector.reset()
        self._left_steps.clear()
        self._right_steps.clear()
        self.update()

    def get_keypoints(self) -> list:
        """Return person 0's RAW world-space keypoints (for rotate/zero buttons)."""
        if self._persons:
            return list(self._persons[0])
        return []

    # ── Footstep detection ──

    def _world_to_body(self, kp_world) -> tuple[float, float, float] | None:
        """Apply the current 4x4 world→view transform; returns body coords."""
        if kp_world is None:
            return None
        try:
            pt = np.asarray(kp_world, dtype=float).reshape(3)
        except Exception:
            return None
        if not np.all(np.isfinite(pt)):
            return None
        v = (self._view_transform @ np.append(pt, 1.0))[:3]
        return float(v[0]), float(v[1]), float(v[2])

    def _update_footsteps_from_persons(self, persons: list) -> None:
        """Run footstep detection on person 0's ankles in body coords.

        Only runs when has_origin=True. Newly emitted placements are
        appended to the rolling history deques.
        """
        if not persons:
            return
        person0 = persons[0]
        n = len(person0)
        l_world = person0[_LEFT_ANKLE_IDX] if n > _LEFT_ANKLE_IDX else None
        r_world = person0[_RIGHT_ANKLE_IDX] if n > _RIGHT_ANKLE_IDX else None

        t = time.perf_counter() - self._t0

        l_body = self._world_to_body(l_world)
        if l_body is not None:
            ev = self._left_detector.update(t, l_body[0], l_body[1], l_body[2])
            if ev is not None:
                self._left_steps.append(ev)

        r_body = self._world_to_body(r_world)
        if r_body is not None:
            ev = self._right_detector.update(t, r_body[0], r_body[1], r_body[2])
            if ev is not None:
                self._right_steps.append(ev)

    # ── Painting ──

    def _draw_footsteps(self, painter: QPainter, to_screen) -> None:
        """Render each foot's placement history as a fading-alpha disc.

        Placements are stored in body coords (post view-transform); we
        force them onto the floor plane (z=0) at render time so they
        appear pinned to the floor regardless of vertical noise in the
        original landing.
        """
        def draw_history(history, base_color: QColor) -> None:
            n = len(history)
            if n == 0:
                return
            # Index 0 = oldest, n-1 = newest. Newest gets full alpha.
            for i, step in enumerate(history):
                age_frac = (n - 1 - i) / max(1, n - 1)  # 0=newest, 1=oldest
                alpha = int(_FOOTSTEP_MAX_ALPHA - age_frac *
                            (_FOOTSTEP_MAX_ALPHA - _FOOTSTEP_MIN_ALPHA))
                fill = QColor(base_color.red(), base_color.green(),
                              base_color.blue(), alpha)
                outline = QColor(20, 20, 20, alpha)
                # Pin to floor (z=0): the marker represents where the foot
                # touched down on the floor plane.
                sx, sy = to_screen(step.x, step.y, 0.0)
                painter.setBrush(fill)
                painter.setPen(QPen(outline, 1))
                h = _FOOTSTEP_HALF_PX
                painter.drawRect(sx - h, sy - h, 2 * h, 2 * h)

        draw_history(self._left_steps, _LEFT_FOOT_COLOR)
        draw_history(self._right_steps, _RIGHT_FOOT_COLOR)

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

            # ── Footstep history ──
            # Drawn just above the floor grid so step markers sit on the
            # floor, beneath the skeleton. Newest = full alpha, oldest = faint.
            self._draw_footsteps(painter, to_screen)

        # ── Skeletons — one per person, each in a distinct colour ──
        def alpha_for_age(age: int) -> int:
            """Linearly fade from 255 (fresh) to ~60 (max held age)."""
            if age <= 0:
                return 255
            # Fade from 255 to 60 across HOLD_MAX_AGE frames
            max_age = max(1, self.HOLD_MAX_AGE)
            frac = min(1.0, age / max_age)
            return int(255 - frac * (255 - 60))

        # Map keypoint index -> body-side colour. Persons all share the same
        # left=blue / right=red palette so the side reads at a glance; if you
        # need to disambiguate two persons, do it spatially (they're 3D-
        # separated already) rather than by overall hue.
        def color_for_kp(k_idx: int) -> QColor:
            side = _LATERALITY.get(k_idx, _BODY_SIDE_CENTER)
            if side == _BODY_SIDE_LEFT:
                return _BODY_LEFT_COLOR
            if side == _BODY_SIDE_RIGHT:
                return _BODY_RIGHT_COLOR
            return _BODY_CENTER_COLOR

        def color_for_limb(i: int, j: int) -> QColor:
            si = _LATERALITY.get(i, _BODY_SIDE_CENTER)
            sj = _LATERALITY.get(j, _BODY_SIDE_CENTER)
            # Both endpoints same side → that side's colour.
            if si == sj:
                return color_for_kp(i)
            # Limb crossing midline (e.g. shoulder-to-shoulder, hip-to-hip):
            # use the centre tone so it doesn't read as belonging to one side.
            if _BODY_SIDE_CENTER in (si, sj):
                # Connecting a side to centre — colour by the side endpoint.
                return color_for_kp(i if si != _BODY_SIDE_CENTER else j)
            return _BODY_CENTER_COLOR

        for person_idx, view_pts in enumerate(all_view_pts):
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
                base = color_for_limb(i, j)
                limb_color = QColor(base.red(), base.green(), base.blue(), a)
                painter.setPen(QPen(limb_color, 2))
                painter.drawLine(*to_screen(*view_pts[i]), *to_screen(*view_pts[j]))

            # Keypoints — dim each by its own age. Radius is 3 px (was 4),
            # i.e. 75% of the previous size — quieter dots that crowd less.
            for k_idx, vp in enumerate(view_pts):
                if vp is None:
                    continue
                a = alpha_for_age(age_of(k_idx))
                base = color_for_kp(k_idx)
                dot_color = QColor(base.red(), base.green(), base.blue(), a)
                outline_color = QColor(30, 30, 30, a)
                painter.setBrush(dot_color)
                painter.setPen(QPen(outline_color, 1))
                sp = to_screen(*vp)
                painter.drawEllipse(sp[0] - 3, sp[1] - 3, 6, 6)

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
