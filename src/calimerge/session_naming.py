"""
Session-directory naming utilities.

A workout session subdirectory is named::

    <username>_<timestamp>_<workout_type>/

so that multiple users sharing the same `workouts/` parent folder do not
collide on identical timestamps.  The username portion is sanitized for
filesystem safety -- slashes, colons, NUL, and Windows-reserved
characters are replaced with ``_``; leading dots are stripped.

Public helpers
--------------

``sanitize_username(username)`` -- collision-safe filesystem token.
``build_session_dir_name(username, timestamp, workout_type)`` -- the
canonical directory basename used at recording time.
"""

from __future__ import annotations

# Characters never allowed in any cross-platform filename.
# Includes Windows-reserved set plus path separators and control chars.
_FORBIDDEN = set('<>:"/\\|?*\0')


def sanitize_username(username: str) -> str:
    """
    Return a filesystem-safe token derived from ``username``.

    * Replaces forbidden characters (``< > : " / \\ | ? *`` and NUL plus
      ASCII control bytes) with ``_``.
    * Strips leading dots so the resulting basename never starts with
      ``.`` (avoids hidden directories on POSIX).
    * Collapses runs of ``_`` and trims trailing whitespace / dots.
    * Falls back to ``"user"`` when the result would be empty.

    Examples
    --------
    >>> sanitize_username("alice")
    'alice'
    >>> sanitize_username("a/b:c")
    'a_b_c'
    >>> sanitize_username("..hidden")
    'hidden'
    >>> sanitize_username("")
    'user'
    """
    if not username:
        return "user"

    out_chars: list[str] = []
    for ch in username:
        if ch in _FORBIDDEN or ord(ch) < 0x20:
            out_chars.append("_")
        else:
            out_chars.append(ch)

    cleaned = "".join(out_chars).lstrip(".").rstrip(" .")

    # Collapse repeated underscores so "a/b/c" -> "a_b_c", not "a___c".
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    cleaned = cleaned.strip("_")
    return cleaned or "user"


def build_session_dir_name(
    username: str,
    timestamp: str,
    workout_type: str,
) -> str:
    """
    Return the basename used for a workout session directory.

    Format: ``<sanitized_username>_<timestamp>_<workout_type>``.
    Example::

        >>> build_session_dir_name("alice", "20260427_153022", "fga_horizontal_head_turns")
        'alice_20260427_153022_fga_horizontal_head_turns'
    """
    return f"{sanitize_username(username)}_{timestamp}_{workout_type}"
