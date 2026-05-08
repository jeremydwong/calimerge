"""
Audit which extrinsic-calibration session the offline pipeline picks
for the zelda recording, and confirm it pre-dates the recording.

This is the same selection logic as run_offline_pipeline_on_test_data.py:
   (a) workouts.db sessions.extrinsic_session_id (preferred)
   (b) timestamp-before-recording fallback (chronological)
   (c) load_latest_extrinsic_session (LAST RESORT — risky)

Run:
    uv run python3 tests/manual/check_calibration_choice.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECORDING_NAME = "zelda_20260428_151934_fga_horizontal_head_turns"
# Recording timestamp parsed out of the folder name:
#   <user>_<YYYYMMDD>_<HHMMSS>_<rest>
# zelda_20260428_151934 → 2026-04-28 15:19:34
RECORDING_ISO = "2026-04-28 15:19:34"


def main() -> int:
    from calimerge.config import (
        extrinsics_db_path,
        list_extrinsic_sessions,
        load_extrinsic_session,
        workouts_db_path,
    )

    ext_db = extrinsics_db_path()
    wkt_db = workouts_db_path()
    print(f"extrinsics db: {ext_db}  exists={ext_db.exists()}")
    print(f"workouts db:   {wkt_db}  exists={wkt_db.exists()}")
    print(f"recording:     {RECORDING_NAME}")
    print(f"recording iso: {RECORDING_ISO}")

    if not ext_db.exists():
        print("\nFATAL: no extrinsics.db on this machine.")
        return 1

    sessions = list_extrinsic_sessions()
    print(f"\nall extrinsic sessions in db ({len(sessions)} total, newest-first):")
    for s in sessions:
        marker = " <-- chosen" if str(s["created_at"]) <= RECORDING_ISO else ""
        delta = "before" if str(s["created_at"]) <= RECORDING_ISO else "after "
        print(
            f"  id={s['id']:>3}  created={s['created_at']}  ({delta} recording){marker}"
        )

    # Path (a): workouts.db lookup
    chosen = None
    via = None
    if wkt_db.exists():
        try:
            conn = sqlite3.connect(str(wkt_db))
            row = conn.execute(
                "SELECT extrinsic_session_id "
                "FROM sessions WHERE recording_path LIKE ? "
                "ORDER BY created_at DESC LIMIT 1",
                (f"%{RECORDING_NAME}",),
            ).fetchone()
            conn.close()
            if row and row[0]:
                chosen = int(row[0])
                via = "workouts.db sessions.extrinsic_session_id"
        except Exception as e:
            print(f"\n  (workouts.db lookup raised: {e})")

    # Path (b): timestamp-before-recording fallback
    if chosen is None:
        for s in sessions:
            if str(s["created_at"]) <= RECORDING_ISO:
                chosen = int(s["id"])
                via = f"newest extrinsic_session before {RECORDING_ISO}"
                break

    # Path (c): latest fallback (BAD if (b) would have produced a different answer)
    if chosen is None and sessions:
        chosen = int(sessions[0]["id"])
        via = "load_latest_extrinsic_session (LAST RESORT — possibly post-dates recording)"

    if chosen is None:
        print("\nFATAL: no extrinsic session would be selected.")
        return 2

    print(f"\nselected: id={chosen}")
    print(f"selected via: {via}")
    loaded = load_extrinsic_session(chosen)
    if loaded is None:
        print("FATAL: load_extrinsic_session returned None for chosen id")
        return 3
    created_at, cams = loaded
    print(f"created_at: {created_at}  cameras: {sorted(cams.keys())}")
    print(f"recording predates calibration: {RECORDING_ISO > str(created_at)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
