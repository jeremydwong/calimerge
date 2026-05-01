"""
Headless repro of the GUI 'Log In' camera-init path.

Loads the most recent extrinsic calibration (calibration.toml) under the
last project folder, matches enumerated cameras by serial number against
the calibrated set (same logic as workout_page._on_cameras_found), and
captures one synced frame from each matched camera.

Quits with non-zero status (and a clear reason) if:
  - no calibration is found,
  - any calibrated serial is missing from intrinsics.db,
  - any calibrated serial is missing from the live enumeration,
  - opening or capturing fails.

Run:  uv run python tests/manual/check_extrinsic_login.py
"""

from __future__ import annotations

from pathlib import Path

from calimerge.camera_binding import (
    capture_synced,
    close_camera,
    enumerate_cameras,
    init,
    open_camera,
    shutdown,
)
from calimerge.config import (
    intrinsics_db_path,
    load_app_settings,
    load_calibration_from_toml,
)
import rtoml


def _find_latest_calibration() -> Path | None:
    app = load_app_settings()
    folder = app.get("last_project_folder")
    if not folder:
        return None
    cal_files = sorted(Path(folder).glob("*/calibration.toml"))
    return cal_files[-1] if cal_files else None


def _calibrated_serials_from_toml(path: Path) -> dict[int, tuple[str, tuple[int, int]]]:
    """Read raw (port -> (serial, resolution)) from the toml without DB filtering."""
    data = rtoml.load(path)
    out: dict[int, tuple[str, tuple[int, int]]] = {}
    for port_str, cam in data.get("cameras", {}).items():
        port = int(port_str)
        serial = cam["serial_number"]
        res = tuple(cam["intrinsics_resolution"])
        out[port] = (serial, res)
    return out


def main() -> int:
    cal_path = _find_latest_calibration()
    if cal_path is None:
        print("FAIL: no calibration.toml found under last_project_folder")
        return 2
    print(f"[cal] {cal_path}")

    raw = _calibrated_serials_from_toml(cal_path)
    print(f"[cal] {len(raw)} camera(s) in calibration.toml:")
    for port, (serial, res) in sorted(raw.items()):
        print(f"  port={port} serial={serial!r} intrinsics_resolution={res}")

    db = intrinsics_db_path()
    print(f"[db]  {db} (exists={db.exists()})")
    calibrated = load_calibration_from_toml(cal_path) or {}
    if len(calibrated) < len(raw):
        missing = [
            (p, s, r) for p, (s, r) in raw.items()
            if p not in calibrated
        ]
        print("FAIL: calibration.toml has cameras whose intrinsics are missing in DB:")
        for p, s, r in missing:
            print(f"  port={p} serial={s!r} resolution={r}")
        print("  (load_calibration_from_toml silently drops these — that's likely "
              "why only 1 camera shows after Log In)")
        return 3

    cal_serial_to_port = {c.serial_number: p for p, c in calibrated.items()}

    init()
    try:
        enumerated = enumerate_cameras()
        print(f"[enum] {len(enumerated)} camera(s) discovered:")
        for c in enumerated:
            print(f"  serial={c.serial_number!r} name={c.display_name!r}")

        matched = []
        unmatched_cal = set(cal_serial_to_port.keys())
        for c in enumerated:
            if c.serial_number in cal_serial_to_port:
                port = cal_serial_to_port[c.serial_number]
                matched.append((port, c))
                unmatched_cal.discard(c.serial_number)

        if unmatched_cal:
            print("FAIL: calibrated serial(s) not present in live enumeration:")
            for s in sorted(unmatched_cal):
                print(f"  {s!r}")
            return 4

        print(f"[match] {len(matched)} camera(s) matched calibration:")
        for port, c in sorted(matched):
            print(f"  port={port} <- serial={c.serial_number!r}")

        opened = []
        for port, c in sorted(matched):
            try:
                open_camera(c)
                opened.append(c)
                print(f"[open] port={port} ok")
            except Exception as e:
                print(f"FAIL: open port={port} serial={c.serial_number!r}: {e}")
                for o in opened:
                    close_camera(o)
                return 5

        try:
            fs = capture_synced(opened)
        except Exception as e:
            print(f"FAIL: capture_synced raised {type(e).__name__}: {e}")
            return 6
        finally:
            for c in opened:
                close_camera(c)

        for port, frame in sorted(fs.frames.items()):
            if frame is None:
                print(f"  port={port}: NONE (dropped)")
            else:
                print(f"  port={port}: shape={frame.pixels.shape}")

        if any(f is None for f in fs.frames.values()):
            print("FAIL: at least one camera dropped its frame")
            return 7

        print("OK: all calibrated cameras matched, opened, and delivered a frame")
        return 0
    finally:
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
