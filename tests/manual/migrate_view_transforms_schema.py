"""
One-shot migration: view_transforms.db schema upgrade.

Old schema (single preset per model, upsert):
    view_transforms(model_key TEXT PK, rotation BLOB, translation BLOB,
                    has_origin INT, updated_at TEXT)

New schema (append-only, session-tagged):
    view_transforms(id INTEGER PK AUTOINCREMENT,
                    extrinsic_session_id INTEGER,
                    model_key TEXT NOT NULL,
                    rotation BLOB, translation BLOB,
                    has_origin INTEGER, created_at TEXT,
                    notes TEXT)

Also renames the canonical model_key for the body model:
    "vitpose"  ->  "synthpose"

Rationale: the model we ship is the SynthPose extension of VitPose
(52 anatomical keypoints), not vanilla VitPose-Base (17 COCO keypoints).
Calling the row "vitpose" in the DB was misleading — every code path
already uses the SynthPose weights.

Idempotent: a `migrations` table records that this was applied, so
re-running `--apply` is a no-op the second time.

Run:
    uv run python3 tests/manual/migrate_view_transforms_schema.py        # DRY RUN
    uv run python3 tests/manual/migrate_view_transforms_schema.py --apply
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


MIGRATION_KEY = "view_transforms-schema-v2-and-vitpose-rename"
RENAME_MAP = {"vitpose": "synthpose"}


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS migrations ("
        " name TEXT PRIMARY KEY, applied_at TIMESTAMP NOT NULL"
        ")"
    )


def _is_migrated(conn: sqlite3.Connection, key: str) -> bool:
    _ensure_migrations_table(conn)
    return conn.execute(
        "SELECT 1 FROM migrations WHERE name = ?", (key,)
    ).fetchone() is not None


def _record_migration(conn: sqlite3.Connection, key: str) -> None:
    _ensure_migrations_table(conn)
    conn.execute(
        "INSERT OR IGNORE INTO migrations (name, applied_at) "
        "VALUES (?, datetime('now', 'localtime'))",
        (key,),
    )


def _has_old_schema(conn: sqlite3.Connection) -> bool:
    """Old schema: model_key is the PRIMARY KEY (no `id` column)."""
    cols = list(conn.execute("PRAGMA table_info(view_transforms)"))
    if not cols:
        return False
    names = {c[1] for c in cols}
    has_id = "id" in names
    has_session = "extrinsic_session_id" in names
    # New schema must have both. Old has neither.
    return not (has_id and has_session)


def _migrate(db_path: Path, apply: bool) -> int:
    if not db_path.exists():
        print(f"  {db_path}: missing — nothing to migrate")
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        if _is_migrated(conn, MIGRATION_KEY):
            print(f"  {db_path}: already migrated — skip")
            return 0

        cols = [c[1] for c in conn.execute("PRAGMA table_info(view_transforms)")]
        if not cols:
            print(f"  {db_path}: no view_transforms table — skip")
            if apply:
                _record_migration(conn, MIGRATION_KEY)
                conn.commit()
            return 0
        print(f"  current columns: {cols}")

        if not _has_old_schema(conn):
            print(f"  {db_path}: already has new schema — recording migration sentinel only")
            if apply:
                _record_migration(conn, MIGRATION_KEY)
                conn.commit()
            return 0

        rows = list(conn.execute(
            "SELECT model_key, rotation, translation, has_origin, updated_at "
            "FROM view_transforms"
        ))
        print(f"  legacy rows to migrate: {len(rows)}")
        for r in rows:
            new_key = RENAME_MAP.get(r[0], r[0])
            tag = " (rename)" if new_key != r[0] else ""
            print(f"    {r[0]!r:>16} -> {new_key!r:<16}{tag}  has_origin={r[3]}  updated={r[4]}")

        if not apply:
            return 0

        # Restructure: rename old, build new, copy, drop legacy.
        conn.execute("ALTER TABLE view_transforms RENAME TO view_transforms_legacy")
        conn.execute(
            """
            CREATE TABLE view_transforms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                extrinsic_session_id INTEGER,
                model_key TEXT NOT NULL,
                rotation BLOB NOT NULL,
                translation BLOB NOT NULL,
                has_origin INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                notes TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vt_lookup "
            "ON view_transforms(model_key, extrinsic_session_id, created_at)"
        )
        for r in rows:
            new_key = RENAME_MAP.get(r[0], r[0])
            conn.execute(
                "INSERT INTO view_transforms ("
                " extrinsic_session_id, model_key, rotation, translation,"
                " has_origin, created_at, notes"
                ") VALUES (NULL, ?, ?, ?, ?, ?, ?)",
                (new_key, r[1], r[2], int(r[3]), r[4],
                 f"migrated from view_transforms_legacy (was model_key={r[0]!r})"),
            )
        conn.execute("DROP TABLE view_transforms_legacy")
        _record_migration(conn, MIGRATION_KEY)
        conn.commit()
        print(f"  migrated {len(rows)} rows; dropped view_transforms_legacy")
        return len(rows)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually write changes. Default is DRY RUN.",
    )
    args = parser.parse_args(argv)

    from calimerge.config import view_transforms_db_path
    db = view_transforms_db_path()
    print(f"db: {db}")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN'}\n")
    n = _migrate(db, args.apply)
    print(f"\n{n} rows {'migrated' if args.apply else 'would migrate'}")
    if not args.apply and n:
        print("Re-run with --apply to commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
