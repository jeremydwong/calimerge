"""Show the workouts.db sessions table schema so we know what's recoverable."""
from __future__ import annotations
import sqlite3
import sys
from pathlib import Path


def main() -> int:
    from calimerge.config import workouts_db_path
    db = workouts_db_path()
    if not db.exists():
        print(f"missing: {db}")
        return 1
    conn = sqlite3.connect(str(db))
    print(f"workouts.db: {db}\n")

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )]
    print(f"tables: {tables}\n")

    for t in tables:
        cols = list(conn.execute(f"PRAGMA table_info({t})"))
        print(f"=== {t} ===")
        for c in cols:
            # cid, name, type, notnull, dflt_value, pk
            print(f"  {c[1]:<24} {c[2]}")
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  ({n} rows)\n")

    if "sessions" in tables:
        print("sessions table sample (last 5):")
        rows = list(conn.execute(
            "SELECT * FROM sessions ORDER BY rowid DESC LIMIT 5"
        ))
        cols = [c[1] for c in conn.execute("PRAGMA table_info(sessions)")]
        for row in rows:
            d = dict(zip(cols, row))
            print(f"  {d}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
