"""Check what view transforms are in the DB near the zelda recording time."""
import sqlite3
from pathlib import Path

from calimerge.config import extrinsics_db_path

# view_transforms.db is next to extrinsics.db
vt_db = extrinsics_db_path().parent / "view_transforms.db"
print(f"view_transforms.db: {vt_db}")

conn = sqlite3.connect(str(vt_db))
conn.row_factory = sqlite3.Row

print("\nAll view transforms (newest first):")
print(f"  {'id':>4} {'model':>12} {'sess_id':>8} {'created_at':>22}")
print(f"  {'----':>4} {'------------':>12} {'--------':>8} {'----------------------':>22}")
for row in conn.execute(
    "SELECT id, model_name, extrinsic_session_id, created_at "
    "FROM view_transforms ORDER BY created_at DESC"
):
    print(f"  {row['id']:>4} {row['model_name']:>12} "
          f"{row['extrinsic_session_id'] or '':>8} "
          f"{row['created_at']:>22}")

print("\nRecording timestamp: 2026-04-28 15:19:34")
print("Expected: view transform created ~15:09-15:10")

# Also check extrinsic sessions
print("\nExtrinsic sessions near recording time:")
ext_db = extrinsics_db_path()
conn2 = sqlite3.connect(str(ext_db))
conn2.row_factory = sqlite3.Row
for row in conn2.execute(
    "SELECT id, created_at FROM extrinsic_sessions "
    "WHERE created_at BETWEEN '2026-04-28 14:00:00' AND '2026-04-28 16:00:00' "
    "ORDER BY created_at"
):
    print(f"  id={row['id']}  created_at={row['created_at']}")

conn.close()
conn2.close()
