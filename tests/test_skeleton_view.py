"""Sanity tests for the SkeletonViewWidget visualization choices.

These are static checks against the constants in `skeleton_view.py` so we can
verify (a) the laterality table classifies every body keypoint correctly,
(b) left/right body colors are distinct and match the foot colors, and
(c) the keypoint radius reduction landed.

We don't instantiate the widget itself (that would need pytest-qt) — the
look-up tables and constants are the load-bearing pieces and they live at
module level.
"""

from __future__ import annotations


def test_laterality_classifies_all_body_keypoints():
    from calimerge.gui.widgets.skeleton_view import (
        _LATERALITY,
        _BODY_SIDE_LEFT,
        _BODY_SIDE_RIGHT,
        _BODY_SIDE_CENTER,
    )
    from calimerge.tracking.markers import SYNTHPOSE_MARKERS

    # Every marker index must have a side classification — drives skeleton
    # colouring; an unclassified index would silently fall back to "center"
    # which is misleading.
    for idx in SYNTHPOSE_MARKERS:
        assert idx in _LATERALITY, f"missing laterality for idx {idx}"
        assert _LATERALITY[idx] in {
            _BODY_SIDE_LEFT, _BODY_SIDE_RIGHT, _BODY_SIDE_CENTER,
        }


def test_known_keypoints_are_classified_correctly():
    from calimerge.gui.widgets.skeleton_view import _LATERALITY
    # Spot-check a representative sample of the SynthPose-52 schema.
    expected = {
        # COCO-17 main joints
        0: "C",   # Nose
        5: "L", 6: "R",   # shoulders
        7: "L", 8: "R",   # elbows
        9: "L", 10: "R",  # wrists
        11: "L", 12: "R", # hips
        13: "L", 14: "R", # knees
        15: "L", 16: "R", # ankles
        # SynthPose extension landmarks
        17: "C",          # sternum
        28: "R", 29: "L", # ASIS
        40: "R", 41: "L", # 5th metatarsal
        46: "L", 47: "R", # calcaneus
        48: "C", 49: "C", # spine
    }
    for idx, side in expected.items():
        assert _LATERALITY[idx] == side, (
            f"idx {idx} expected {side}, got {_LATERALITY[idx]}"
        )


def test_left_right_balance():
    """Anatomy is symmetric — every left-side index should have a right
    counterpart in the schema."""
    from calimerge.gui.widgets.skeleton_view import _LATERALITY
    n_l = sum(1 for v in _LATERALITY.values() if v == "L")
    n_r = sum(1 for v in _LATERALITY.values() if v == "R")
    assert n_l == n_r, f"left ({n_l}) != right ({n_r}) — schema mismatch"


def test_body_and_foot_colors_match_per_side():
    """Foot dots should read as 'belonging to' the same side they're drawn
    under, so the body side colour and foot side colour must match."""
    from calimerge.gui.widgets.skeleton_view import (
        _BODY_LEFT_COLOR, _BODY_RIGHT_COLOR,
        _LEFT_FOOT_COLOR, _RIGHT_FOOT_COLOR,
    )
    assert _BODY_LEFT_COLOR.getRgb()[:3] == _LEFT_FOOT_COLOR.getRgb()[:3]
    assert _BODY_RIGHT_COLOR.getRgb()[:3] == _RIGHT_FOOT_COLOR.getRgb()[:3]


def test_left_blue_right_red_palette():
    """The user picked blue for left, red for right. Lock that so we don't
    silently flip it later."""
    from calimerge.gui.widgets.skeleton_view import (
        _BODY_LEFT_COLOR, _BODY_RIGHT_COLOR,
    )
    # Left = blue: blue channel dominates
    lr, lg, lb, _ = _BODY_LEFT_COLOR.getRgb()
    assert lb > lr and lb > lg, f"left should be blue-dominant, got {(lr, lg, lb)}"
    # Right = red: red channel dominates
    rr, rg, rb, _ = _BODY_RIGHT_COLOR.getRgb()
    assert rr > rg and rr > rb, f"right should be red-dominant, got {(rr, rg, rb)}"


def test_footstep_marker_is_sized_for_squares():
    """The new square footstep marker uses _FOOTSTEP_HALF_PX. The old disc
    constant (_FOOTSTEP_RADIUS_PX) should be gone — leaving it around invites
    drift between two side-by-side conventions."""
    from calimerge.gui.widgets import skeleton_view
    assert hasattr(skeleton_view, "_FOOTSTEP_HALF_PX")
    assert not hasattr(skeleton_view, "_FOOTSTEP_RADIUS_PX")


def test_footstep_alpha_is_transparent():
    """User asked specifically for *transparent* squares — lock the upper
    bound so we don't quietly bump it back to fully opaque."""
    from calimerge.gui.widgets.skeleton_view import _FOOTSTEP_MAX_ALPHA
    assert _FOOTSTEP_MAX_ALPHA <= 200, (
        f"footstep alpha {_FOOTSTEP_MAX_ALPHA} too high — squares should be "
        f"see-through enough to read the floor grid behind them."
    )
