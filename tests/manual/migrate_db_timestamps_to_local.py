"""
One-shot migration: shift `created_at` from UTC to local time in
extrinsics.db and workouts.db.

Background. SQLite's `CURRENT_TIMESTAMP` is UTC. The previous schema
defaulted `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`, so every
calibration session got a UTC timestamp. Recording filenames are local
time (`<user>_YYYYMMDD_HHMMSS_<rest>`), so the offline pipeline runner's
"newest extrinsic_session before the recording" logic was comparing
local-time filename to UTC stored timestamp — picking the wrong session
when the rig was re-calibrated minutes before recording (UTC > local
makes the right session look "after" the recording).

Schema is now `DEFAULT (datetime('now', 'localtime'))` and explicit
INSERTs pass local time. This script converts existing rows.

Idempotency. A `migrations` table records which (db, table) pairs have
been migrated; running `--apply` twice is a no-op the second time.

Usage:
    uv run python3 tests/manual/migrate_db_timestamps_to_local.py
        # DRY RUN — print what would change, no writes

    uv run python3 tests/manual/migrate_db_timestamps_to_local.py --apply
        # Actually shift the rows.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sqlite3
import sys
from pathlib import Path


# (db_resolver_callable, table_name, column_name)
TARGETS = [
    ("extrinsics", "extrinsic_sessions", "created_at"),
    ("workouts", "sessions", "created_at"),
    ("workouts", "users", "created_at"),
    ("intrinsics", "intrinsics", "created_at"),
    ("view_transforms", "view_transforms", "updated_at"),
]


def _local_offset_seconds() -> int:
    """How many seconds local time is ahead of UTC (negative for west-of-UTC)."""
    now_local = _dt.datetime.now().astimezone()
    return int(now_local.utcoffset().total_seconds())


def _resolve_db(name: str) -> Path:
    from calimerge.config import (
        extrinsics_db_path, workouts_db_path, intrinsics_db_path,
        view_transforms_db_path,
    )
    return {
        "extrinsics": extrinsics_db_path(),
        "workouts": workouts_db_path(),
        "intrinsics": intrinsics_db_path(),
        "view_transforms": view_transforms_db_path(),
    }[name]


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS migrations ("
        " name TEXT PRIMARY KEY, applied_at TIMESTAMP NOT NULL"
        ")"
    )


def _is_already_migrated(conn: sqlite3.Connection, key: str) -> bool:
    _ensure_migrations_table(conn)
    row = conn.execute(
        "SELECT applied_at FROM migrations WHERE name = ?", (key,),
    ).fetchone()
    return row is not None


def _record_migration(conn: sqlite3.Connection, key: str) -> None:
    _ensure_migrations_table(conn)
    conn.execute(
        "INSERT OR IGNORE INTO migrations (name, applied_at) "
        "VALUES (?, datetime('now', 'localtime'))",
        (key,),
    )


def _shift_table(
    db_path: Path,
    table: str,
    col: str,
    offset_seconds: int,
    apply: bool,
) -> tuple[int, int]:
    """Returns (rows_seen, rows_changed)."""
    if not db_path.exists():
        print(f"  {db_path}: missing — skip")
        return (0, 0)

    conn = sqlite3.connect(str(db_path))
    try:
        # Verify table + column exist
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        if not cols:
            print(f"  {db_path}::{table}: table missing — skip")
            return (0, 0)
        if col not in cols:
            print(f"  {db_path}::{table}: column {col!r} missing — skip")
            return (0, 0)

        key = f"utc-to-local::{table}::{col}"
        if _is_already_migrated(conn, key):
            print(f"  {db_path}::{table}.{col}: already migrated — skip")
            return (0, 0)

        rows = list(conn.execute(f"SELECT rowid, {col} FROM {table}"))
        seen = len(rows)
        if seen == 0:
            print(f"  {db_path}::{table}.{col}: 0 rows")
            if apply:
                _record_migration(conn, key)
                conn.commit()
            return (0, 0)

        delta = _dt.timedelta(seconds=offset_seconds)
        changed = 0
        print(f"  {db_path}::{table}.{col}: {seen} rows, applying delta {delta}")
        for rowid, val in rows:
            if val is None:
                continue
            try:
                # SQLite returns the string as-is (e.g. "2026-04-28 21:10:45")
                v = str(val)
                # Try common formats
                dt = None
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        dt = _dt.datetime.strptime(v, fmt)
                        break
                    except ValueError:
                        continue
                if dt is None:
                    print(f"    rowid={rowid}: unparseable {val!r} — skip")
                    continue
                new_dt = dt + delta
                new_val = new_dt.strftime("%Y-%m-%d %H:%M:%S")
                print(f"    rowid={rowid}: {v!s:<20}  ->  {new_val}")
                if apply:
                    conn.execute(
                        f"UPDATE {table} SET {col} = ? WHERE rowid = ?",
                        (new_val, rowid),
                    )
                changed += 1
            except Exception as e:
                print(f"    rowid={rowid}: error {e}")

        if apply and changed:
            _record_migration(conn, key)
            conn.commit()
            print(f"    committed; changed={changed}")
        return (seen, changed)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually write changes. Default is DRY RUN.",
    )
    args = parser.parse_args(argv)

    offset = _local_offset_seconds()
    print(f"local-vs-UTC offset: {offset} seconds "
          f"({_dt.timedelta(seconds=offset)})")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN'}\n")

    total_seen = 0
    total_changed = 0
    for db_name, table, col in TARGETS:
        try:
            db = _resolve_db(db_name)
        except Exception as e:
            print(f"could not resolve {db_name}: {e}")
            continue
        print(f"--- {db_name} ({db}) ---")
        seen, changed = _shift_table(db, table, col, offset, args.apply)
        total_seen += seen
        total_changed += changed
        print()

    print(f"summary: {total_seen} rows seen, "
          f"{total_changed} rows {'changed' if args.apply else 'would change'}")
    if not args.apply and total_changed:
        print("\nRe-run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
