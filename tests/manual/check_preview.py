"""
Headless repro of the GUI preview path: enumerate → open → capture_synced loop.

Prints one line per captured sync-frameset so you can see at a glance
whether every port is delivering BGR pixels. Mirrors what
CameraPreviewWorker.run() does, minus the Qt signals.

Run:  uv run python3 tests/manual/check_preview.py
"""

from __future__ import annotations

import time

from calimerge.camera_binding import (
    capture_synced,
    close_camera,
    enumerate_cameras,
    init,
    open_camera,
    shutdown,
)


ITERATIONS = 10
TARGET_FPS = 30


def main() -> int:
    init()
    try:
        cams = enumerate_cameras()
        print(f"[enum] {len(cams)} cameras")
        for c in cams:
            print(
                f"  port={c.device_index} serial={c.serial_number!r} "
                f"name={c.display_name!r} {c.width}x{c.height}@{c.fps}"
            )

        opened = []
        for c in cams:
            try:
                open_camera(c)
                opened.append(c)
                print(f"  opened port={c.device_index}")
            except Exception as e:
                print(f"  FAILED port={c.device_index}: {e}")

        if not opened:
            print("no cameras opened; aborting")
            return 1

        print(f"\n[capture] capture_synced x {ITERATIONS}")
        for i in range(ITERATIONS):
            t0 = time.perf_counter()
            try:
                fs = capture_synced(opened)
            except Exception as e:
                print(f"  iter {i}: EXCEPTION {type(e).__name__}: {e}")
                continue
            dt_ms = (time.perf_counter() - t0) * 1000
            parts = []
            for port, frame in fs.frames.items():
                if frame is None:
                    parts.append(f"port{port}=NONE")
                else:
                    parts.append(f"port{port}={frame.pixels.shape}")
            print(f"  iter {i}: {dt_ms:6.1f}ms dropped={fs.dropped_mask:#06b}  " + " ".join(parts))
            time.sleep(1.0 / TARGET_FPS)

        for c in opened:
            close_camera(c)
        return 0
    finally:
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
