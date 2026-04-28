"""
Tests for ``calimerge.session_naming``.

Covers Task 2:

* The session-dir basename is ``<username>_<timestamp>_<workout_type>``.
* Forbidden / unsafe filesystem characters in the username are
  sanitized.
* Empty / dot-only usernames fall back to a sane default.
"""

from __future__ import annotations

from pathlib import Path

from calimerge.session_naming import (
    build_session_dir_name,
    sanitize_username,
)


# ---------------------------------------------------------------------------
# sanitize_username
# ---------------------------------------------------------------------------


def test_simple_username_pass_through():
    assert sanitize_username("alice") == "alice"
    assert sanitize_username("Alice_42") == "Alice_42"
    assert sanitize_username("jeremy.wong") == "jeremy.wong"


def test_forbidden_characters_replaced():
    # Unix path sep
    assert sanitize_username("a/b") == "a_b"
    # Windows path sep
    assert sanitize_username("a\\b") == "a_b"
    # Drive-letter style colon
    assert sanitize_username("C:user") == "C_user"
    # Quote / wildcards / pipe / question — all Windows-reserved
    for raw, expected in [
        ('a"b', "a_b"),
        ("a*b", "a_b"),
        ("a?b", "a_b"),
        ("a|b", "a_b"),
        ("a<b>c", "a_b_c"),
    ]:
        assert sanitize_username(raw) == expected, raw


def test_control_characters_replaced():
    raw = "alice\x00\x01\x1f"
    assert sanitize_username(raw) == "alice"


def test_leading_dots_stripped():
    assert sanitize_username("..hidden") == "hidden"
    assert sanitize_username(".alice") == "alice"


def test_trailing_whitespace_and_dots_stripped():
    assert sanitize_username("alice ") == "alice"
    assert sanitize_username("alice.") == "alice"
    assert sanitize_username("alice. ") == "alice"


def test_empty_falls_back_to_default():
    assert sanitize_username("") == "user"
    # Pure-forbidden -> after collapse / strip becomes empty -> default
    assert sanitize_username("///") == "user"
    assert sanitize_username("...") == "user"


def test_collapses_repeated_underscores():
    # "a/b/c" produced 3 underscores in a row before — verify they collapse.
    assert sanitize_username("a/b/c") == "a_b_c"
    assert sanitize_username("a///b") == "a_b"


# ---------------------------------------------------------------------------
# build_session_dir_name
# ---------------------------------------------------------------------------


def test_session_dir_name_format():
    name = build_session_dir_name("alice", "20260427_153022", "fga_horizontal_head_turns")
    assert name == "alice_20260427_153022_fga_horizontal_head_turns"


def test_session_dir_name_sanitizes_username():
    name = build_session_dir_name("a/b:c", "20260427_153022", "sit_to_stand")
    assert name == "a_b_c_20260427_153022_sit_to_stand"
    # No path-separator characters should leak through.
    assert "/" not in name and "\\" not in name and ":" not in name


def test_session_dir_under_workout_dir(tmp_path: Path):
    """End-to-end: build a real subdir using the helper, mirroring workout_page.py."""
    workout_dir = tmp_path / "workouts"
    workout_dir.mkdir()

    name = build_session_dir_name("alice", "20260427_153022", "stretch")
    session_dir = workout_dir / name
    session_dir.mkdir(parents=True, exist_ok=True)

    assert session_dir.exists()
    assert session_dir.parent == workout_dir
    assert session_dir.name.startswith("alice_")


def test_session_dir_uses_default_for_empty_username():
    name = build_session_dir_name("", "20260427_153022", "stretch")
    assert name == "user_20260427_153022_stretch"
