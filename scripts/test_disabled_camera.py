"""Reproduce the disabled-camera preview bug.

Enumerates all cameras, opens only a subset (skipping one), and checks
whether capture_synced returns frames keyed by the correct port/device_index.
"""
import sys
import time

from calimerge.camera_binding import (
    init, enumerate_cameras, open_camera, close_camera,
    set_resolution, capture_synced,
)

init()
all_cameras = enumerate_cameras()
print(f"Found {len(all_cameras)} cameras:")
for cam in all_cameras:
    print(f"  device_index={cam.device_index}  {cam.display_name}  serial={cam.serial_number}")

if len(all_cameras) < 2:
    print("Need at least 2 cameras to test. Exiting.")
    sys.exit(1)

# Simulate disabling the FHD Camera (or whichever is at index 1)
# Open only cameras at device_index 0 and 2 (skip index 1)
skip_index = 1
enabled = [cam for cam in all_cameras if cam.device_index != skip_index]
print(f"\nSkipping device_index={skip_index}, opening {len(enabled)} cameras:")
for cam in enabled:
    print(f"  device_index={cam.device_index}  {cam.display_name}")

for cam in enabled:
    open_camera(cam)
    set_resolution(cam, 640, 480)

time.sleep(0.5)

# Capture and check what keys come back
print("\n--- capture_synced with subset ---")
for i in range(5):
    fs = capture_synced(enabled)
    keys = sorted(fs.frames.keys())
    got_frames = []
    for k, frame in sorted(fs.frames.items()):
        if frame:
            got_frames.append(f"key={k} ({frame.width}x{frame.height})")
        else:
            got_frames.append(f"key={k} (None)")
    print(f"  sync#{i}: dict keys={keys}  frames: {', '.join(got_frames)}")

# capture_synced should key frames by device_index (camera identity)
expected_keys = sorted(cam.device_index for cam in enabled)
actual_keys = sorted(fs.frames.keys())
print(f"\n  Expected keys (device_index): {expected_keys}")
print(f"  Actual keys:                  {actual_keys}")
if expected_keys == actual_keys:
    print("  PASS - keys match device_index")
else:
    print("  FAIL - keys are array positions, not device_index!")

# Simulate what the GUI does: assign ports by enumeration order
print("\n--- GUI port mapping simulation ---")
all_ports = {port: cam for port, cam in enumerate(all_cameras)}
enabled_ports = [(port, cam) for port, cam in all_ports.items() if cam.device_index != skip_index]
print(f"  Enabled ports: {[p for p, _ in enabled_ports]}")
print(f"  capture_synced keys: {actual_keys}")
print(f"  GUI would map: array idx 0 -> port {enabled_ports[0][0]}, array idx 1 -> port {enabled_ports[1][0]}")

for cam in enabled:
    close_camera(cam)
print("\nDone.")
