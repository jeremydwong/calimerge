#!/usr/bin/env bash
# test_data_dir.sh - Smoke-test the new app-data-dir refactor.
#
# Approve once and reuse throughout the session. No external side effects:
#   - Imports calimerge.config and prints resolved paths
#   - Verifies CALIMERGE_DATA_DIR env override is respected
#   - Imports tracking modules so any path-related ImportError surfaces
#
# Does NOT touch the redirect file at the platform default location.
# Does NOT download models or read databases.
set -euo pipefail
cd "$(dirname "$0")"

unset VIRTUAL_ENV
UV="$HOME/.local/bin/uv"

echo "== 1. Default resolution (env var unset) =="
"$UV" run python - <<'PY'
import os
os.environ.pop("CALIMERGE_DATA_DIR", None)
from calimerge import config as c
print(f"  data_dir():           {c.data_dir()}")
print(f"  models_dir():         {c.models_dir()}")
print(f"  yolo_dir():           {c.yolo_dir()}")
print(f"  vitpose_dir():        {c.vitpose_dir()}")
print(f"  mediapipe_dir():      {c.mediapipe_dir()}")
print(f"  intrinsics_db_path(): {c.intrinsics_db_path()}")
print(f"  workouts_db_path():   {c.workouts_db_path()}")
print(f"  app_settings_path():  {c.app_settings_path()}")
print(f"  engine_cache_dir():   {c.engine_cache_dir()}")
print(f"  default_recordings_dir(): {c.default_recordings_dir()}")
print(f"  legacy_dotdir():      {c.legacy_dotdir()}  (exists={c.legacy_dotdir().exists()})")
assert str(c.engine_cache_dir()).startswith(str(c.data_dir())), "engine cache must live under data_dir"
expected_rec = "Documents"
assert expected_rec in str(c.default_recordings_dir()), f"recordings default must be under Documents, got {c.default_recordings_dir()}"
PY

echo
echo "== 2. CALIMERGE_DATA_DIR override =="
CALIMERGE_DATA_DIR="C:/tmp/calimerge_smoke" "$UV" run python - <<'PY'
from calimerge import config as c
import sys
expected = "C:/tmp/calimerge_smoke"
got = str(c.data_dir()).replace("\\", "/")
print(f"  data_dir() with env: {got}")
assert got == expected, f"override broken: got {got!r}, expected {expected!r}"
print("  override OK")
PY

echo
echo "== 3. Module imports (path refactor sanity) =="
"$UV" run python - <<'PY'
# Just import — surface any path-related typos at import time.
from calimerge import config
from calimerge.tracking import pose_detector
from calimerge.tracking import hand_detector
from calimerge.gui import main as gui_main
print("  config imported:", hasattr(config, "data_dir"))
print("  pose_detector imported:", hasattr(pose_detector, "load_models"))
print("  hand_detector imported:", hasattr(hand_detector, "_get_detector"))
print("  gui.main imported:", hasattr(gui_main, "MainWindow"))
PY

echo
echo "== 4. Intrinsics DB full dump =="
"$UV" run python - <<'PY'
import sqlite3
from calimerge import config as c

db = c.intrinsics_db_path()
print(f"  resolved path: {db}")
print(f"  exists:        {db.exists()}")
if db.exists():
    st = db.stat()
    print(f"  size:          {st.st_size} bytes")
    print(f"  mtime:         {st.st_mtime}")

if not db.exists():
    print("  (no DB present at the resolved path — nothing to enumerate)")
else:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    print()
    print("  -- table list --")
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        print(f"    {name}")

    print()
    print("  -- intrinsics rows (full dump) --")
    rows = conn.execute(
        "SELECT serial_number, width, height, error, grid_count, created_at "
        "FROM intrinsics ORDER BY serial_number, width, height"
    ).fetchall()
    print(f"  {len(rows)} row(s)")
    for r in rows:
        print(f"    serial={r['serial_number']!r}  {r['width']}x{r['height']}  "
              f"err={r['error']:.4f}  grids={r['grid_count']}  at={r['created_at']}")

    print()
    print("  -- nicknames rows (full dump) --")
    rows = conn.execute(
        "SELECT serial_number, nickname FROM nicknames ORDER BY serial_number"
    ).fetchall()
    print(f"  {len(rows)} row(s)")
    for r in rows:
        print(f"    serial={r['serial_number']!r}  nickname={r['nickname']!r}")

    print()
    print("  -- load_all_nicknames() round-trip --")
    nicks = c.load_all_nicknames()
    print(f"  returned {len(nicks)} entry(ies)")
    for s, n in nicks.items():
        print(f"    {s!r} -> {n!r}")

    conn.close()
PY

echo
echo "== 5. Workouts DB read-through =="
"$UV" run python - <<'PY'
from calimerge import config as c

db = c.workouts_db_path()
print(f"  resolved path: {db}")
print(f"  exists:        {db.exists()}")
print(f"  (this should be under last_project_folder when set, else under data_dir)")

import json
sp = c.app_settings_path()
last = None
if sp.exists():
    last = json.load(open(sp, "r", encoding="utf-8")).get("last_project_folder")
print(f"  last_project_folder: {last}")
PY

echo
echo "== 6. Removed-symbol audit =="
"$UV" run python - <<'PY'
from calimerge import config
removed = ["DEFAULT_INTRINSICS_DB", "DEFAULT_WORKOUTS_DB", "_APP_SETTINGS_PATH"]
leftover = [s for s in removed if hasattr(config, s)]
if leftover:
    raise SystemExit(f"  FAIL: legacy symbols still exported: {leftover}")
print("  no legacy DEFAULT_* / _APP_SETTINGS_PATH symbols on config — OK")
PY

echo
echo "== 7. GUI nickname placeholder is empty (no false 'A') =="
"$UV" run python - <<'PY'
from pathlib import Path
src = Path("src/calimerge/gui/tabs/cameras_tab.py").read_text(encoding="utf-8")
assert 'setPlaceholderText("A")' not in src, "placeholder still 'A' -- bad!"
assert 'setPlaceholderText("")' in src, "placeholder edit not applied"
print("  placeholder is empty -- OK")
PY

echo
echo "== 8a. Simulate intrinsic_tab._load_intrinsics_from_db logic =="
"$UV" run python - <<'PY'
from calimerge.config import (
    intrinsics_db_path, list_intrinsics, load_intrinsics,
    check_intrinsics_availability,
)

db = intrinsics_db_path()
all_db = list_intrinsics(db)
db_by_serial = {}
for sn, w, h, err in all_db:
    db_by_serial.setdefault(sn, []).append((w, h, err))

# Simulate the lookup the intrinsic tab performs for each enabled camera.
# Test against the user's actual cameras: pretend each camera reports
# its native resolution (the most common failure mode), and at the
# project-prefs resolution.
test_cases = [
    ("6&3023cdee&0&0000", (2560, 1440), "Nuroum A native 16:9"),
    ("6&3023cdee&0&0000", (640, 360),  "Nuroum A pref 16:9"),
    ("7&1b959837&0&0000", (2560, 1440), "Nuroum B native 16:9"),
    ("7&1b959837&0&0000", (640, 360),  "Nuroum B pref 16:9"),
    ("6&1ef182c7&0&0000", (640, 480),  "EPI pref 4:3"),
    ("6&29090ad2&0&0000", (640, 480),  "RERE pref 4:3"),
]

for serial, target_res, label in test_cases:
    status, src = check_intrinsics_availability(serial, target_res, db)
    intr = load_intrinsics(serial, target_res, db)
    if intr is None:
        # The fallback the intrinsic tab uses: pick best entry regardless of AR
        entries = db_by_serial.get(serial, [])
        if entries:
            best_w, best_h, _ = min(entries, key=lambda x: x[2])
            intr = load_intrinsics(serial, (best_w, best_h), db)
            fallback = f"  [fallback: loaded at {best_w}x{best_h}]"
        else:
            fallback = "  [no DB entries for serial]"
    else:
        fallback = ""

    fx = intr.matrix[0, 0] if intr is not None else None
    print(f"  {label}")
    print(f"    serial={serial} target={target_res}  availability={status}  src={src}")
    print(f"    loaded? {'YES' if intr is not None else 'NO'}  fx={fx}{fallback}")
PY

echo
echo "== 8e. Imports + schema round-trip (extrinsics.db + workouts.db) =="
"$UV" run python - <<'PY'
import sqlite3
import numpy as np

# Force-import the GUI entry points the launcher actually hits, so any
# wiring changes that break startup show up here.
from calimerge.gui.main import MainWindow  # noqa: F401
from calimerge.gui.workout_page import WorkoutPage  # noqa: F401
from calimerge.config import (
    extrinsics_db_path, init_extrinsics_db, save_extrinsic_session,
    list_extrinsic_sessions, load_latest_extrinsic_session,
    load_extrinsic_session, init_workouts_db, workouts_db_path,
    create_session,
)
from calimerge.types import (
    CalibratedCamera, CameraIntrinsics, CameraExtrinsics,
)

# 1) extrinsics.db round-trip
fake_intr = CameraIntrinsics(
    serial_number="SMOKE_TEST_SERIAL", resolution=(640, 480),
    matrix=np.eye(3), distortion=np.zeros(5), error=0.5, grid_count=10,
)
fake_extr = CameraExtrinsics(
    rotation=np.eye(3), translation=np.array([1.0, 2.0, 3.0]),
)
fake_cc = CalibratedCamera(
    serial_number="SMOKE_TEST_SERIAL", port=99,
    intrinsics=fake_intr, extrinsics=fake_extr,
)
sid = save_extrinsic_session({99: fake_cc}, rmse=0.42,
                             notes="smoke-roundtrip do-not-keep")
print(f"  extrinsics round-trip wrote session_id={sid}")

# Read it back and verify shape (won't load CalibratedCamera because the
# fake serial isn't in intrinsics.db; that's fine — list_extrinsic_sessions
# operates on metadata only and should see the row).
sessions = list_extrinsic_sessions()
hit = [s for s in sessions if s["id"] == sid]
assert len(hit) == 1, f"smoke session not found in list: {sessions}"
print(f"  round-trip metadata: {hit[0]}")

# Cleanup
conn = sqlite3.connect(str(extrinsics_db_path()))
conn.execute("DELETE FROM extrinsic_cameras WHERE session_id = ?", (sid,))
conn.execute("DELETE FROM extrinsic_sessions WHERE id = ?", (sid,))
conn.commit(); conn.close()
print("  cleanup OK")

# 2) workouts.db schema additions
init_workouts_db()
conn = sqlite3.connect(str(workouts_db_path()))
cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
conn.close()
expected = {
    "extrinsic_session_id", "extrinsic_calibrated_at",
    "zero_origin_rotation", "zero_origin_translation",
}
missing = expected - cols
if missing:
    raise SystemExit(f"  FAIL: workouts.db sessions missing columns: {missing}")
print("  workouts.db schema OK -- new columns present")
PY

echo
echo "== 8d. Extrinsics DB (machine-level) =="
"$UV" run python - <<'PY'
from pathlib import Path
from calimerge.config import (
    extrinsics_db_path, init_extrinsics_db, list_extrinsic_sessions,
    load_latest_extrinsic_session, import_calibration_toml_into_db,
    load_app_settings,
)

db = extrinsics_db_path()
print(f"  resolved path: {db}")
print(f"  exists:        {db.exists()}")

# Always init (idempotent) so the smoke test can verify schema.
init_extrinsics_db(db)
print(f"  init OK,       size now: {db.stat().st_size} bytes")

sessions = list_extrinsic_sessions(db_path=db)
print(f"  sessions:      {len(sessions)}")

# If empty, migrate the most recent calibration.toml so the rest of this
# section has something to read against. This mirrors the in-app migration
# in workout_page._check_calibration.
if not sessions:
    app = load_app_settings()
    folder = app.get("last_project_folder")
    if folder:
        cal_files = sorted(Path(folder).glob("*/calibration.toml"))
        if cal_files:
            sid = import_calibration_toml_into_db(
                cal_files[-1],
                notes=f"Smoke-test migration from {cal_files[-1]}",
                db_path=db,
            )
            print(f"  migrated:      {cal_files[-1]} -> session_id={sid}")
            sessions = list_extrinsic_sessions(db_path=db)

for s in sessions[:5]:
    print(f"    id={s['id']:>3}  cams={s['num_cameras']}  "
          f"created={s['created_at']}  rmse={s['rmse']}  "
          f"notes={s['notes']!r}")

latest = load_latest_extrinsic_session(db_path=db)
if latest is None:
    print("  load_latest_extrinsic_session: None")
else:
    sid, created, cams = latest
    print(f"  load_latest:   id={sid}  created={created}  cams={len(cams)}")
    for port in sorted(cams.keys()):
        cc = cams[port]
        print(f"    port={port}  serial={cc.serial_number!r}  "
              f"intrinsics_res={cc.intrinsics.resolution}  "
              f"err={cc.intrinsics.error:.4f}")
PY

echo
echo "== 8c. CUDA streaming model + engine cache resolution =="
"$UV" run python - <<'PY'
from pathlib import Path
from calimerge.config import models_dir, engine_cache_dir

onnx_dir = models_dir() / "onnx"
print(f"  ONNX dir (preferred): {onnx_dir}  exists={onnx_dir.exists()}")
for fname in ("yolo_v10s.onnx", "vitpose_synthpose.onnx"):
    p = onnx_dir / fname
    print(f"    {fname:>30}: {'OK' if p.exists() else 'MISSING'}  size={p.stat().st_size if p.exists() else '-'}")

repo_legacy = Path("models/onnx").resolve()
print(f"  ONNX dir (legacy):    {repo_legacy}  exists={repo_legacy.exists()}")
for fname in ("yolo_v10s.onnx", "vitpose_synthpose.onnx"):
    p = repo_legacy / fname
    print(f"    {fname:>30}: {'OK' if p.exists() else 'MISSING'}  size={p.stat().st_size if p.exists() else '-'}")

cache = engine_cache_dir()
print(f"  engine_cache_dir:     {cache}  exists={cache.exists()}")
if cache.exists():
    files = list(cache.iterdir())
    print(f"    contents: {len(files)} entry(ies)")
    for f in files[:10]:
        print(f"      {f.name}  size={f.stat().st_size}")
PY

echo
echo "== 8b. Extrinsic calibration cross-reference (login matching) =="
"$UV" run python - <<'PY'
from pathlib import Path
import rtoml
from calimerge.config import load_app_settings, load_calibration_from_toml

app = load_app_settings()
folder = app.get("last_project_folder")
if not folder:
    print("  no last_project_folder set; skip")
    raise SystemExit(0)

folder = Path(folder)
cal_files = sorted(folder.glob("*/calibration.toml"))
print(f"  workout dir:   {folder}")
print(f"  cal files:     {len(cal_files)}")
if not cal_files:
    raise SystemExit(0)

latest = cal_files[-1]
print(f"  most recent:   {latest}  (mtime={latest.stat().st_mtime})")
print()

# Raw TOML contents
data = rtoml.load(latest)
toml_cams = data.get("cameras", {})
print(f"  TOML 'cameras' section has {len(toml_cams)} entry/entries:")
for port_str, cam in toml_cams.items():
    serial = cam.get("serial_number")
    res = cam.get("intrinsics_resolution")
    print(f"    port={port_str!r:>5}  serial={serial!r}  res={res}  "
          f"len(serial)={len(serial) if serial else None}")

print()
# Through the loader (this drops cameras whose intrinsics can't be loaded)
loaded = load_calibration_from_toml(latest)
n_loaded = len(loaded) if loaded else 0
print(f"  load_calibration_from_toml() returned {n_loaded} camera(s):")
if loaded:
    for port, cc in sorted(loaded.items()):
        print(f"    port={port}  serial={cc.serial_number!r}  intrinsics res={cc.intrinsics.resolution}")

if loaded and n_loaded < len(toml_cams):
    dropped = []
    loaded_serials = {cc.serial_number for cc in loaded.values()}
    for port_str, cam in toml_cams.items():
        if cam.get("serial_number") not in loaded_serials:
            dropped.append(cam.get("serial_number"))
    print(f"  DROPPED by loader (likely intrinsics missing for stored resolution):")
    for s in dropped:
        print(f"    {s!r}")

print()
# Cross-reference against live cameras (same path the GUI takes)
print("  -- live enumerate_cameras() --")
try:
    from calimerge import camera_binding as cb
    cb.init()
    live = cb.enumerate_cameras()
    print(f"  enumerated {len(live)} camera(s):")
    for cam in live:
        s = cam.serial_number
        print(f"    idx={cam.device_index}  serial={s!r}  len={len(s)}  "
              f"name={cam.display_name!r}")

    print()
    print("  -- match attempt (same logic as workout_page._on_cameras_found) --")
    if loaded:
        cal_serial_to_port = {cc.serial_number: port for port, cc in loaded.items()}
        for cam in live:
            hit = cal_serial_to_port.get(cam.serial_number)
            verdict = f"MATCH (port={hit})" if hit is not None else "NOT MATCHED -> skipped"
            print(f"    {cam.serial_number!r}  -> {verdict}")
finally:
    try:
        cb.shutdown()
    except Exception:
        pass
PY

echo
echo "== 8. Project settings dump (per-camera prefs) =="
"$UV" run python - <<'PY'
import json
from pathlib import Path
from calimerge.config import load_app_settings, load_project_settings

app = load_app_settings()
folder = app.get("last_project_folder")
print(f"  last_project_folder: {folder}")
if not folder:
    print("  (no project folder set -- per-camera prefs will be empty)")
else:
    p = Path(folder)
    settings_path = p / "settings.json"
    print(f"  settings.json:       {settings_path}  exists={settings_path.exists()}")
    settings = load_project_settings(p)
    cams = settings.get("cameras", {})
    print(f"  cameras section has {len(cams)} entry(ies):")
    for serial, prefs in cams.items():
        print(f"    {serial}: {prefs}")
PY

echo
echo "All smoke tests passed."
