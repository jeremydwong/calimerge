"""Test cameras using the same code path as the GUI (capture_synced)."""
from calimerge.camera_binding import (
    init, enumerate_cameras, open_camera, close_camera,
    set_resolution, set_exposure, capture_synced,
)
import time

init()
cameras = enumerate_cameras()

print(f"Found {len(cameras)} cameras")
for cam in cameras:
    print(f"  {cam.display_name} (serial={cam.serial_number}, {cam.width}x{cam.height})")

# Open all
print("\n--- Opening ---")
opened = []
for cam in cameras:
    try:
        open_camera(cam)
        print(f"  Opened: {cam.display_name} -> {cam.width}x{cam.height}")
        opened.append(cam)
    except Exception as e:
        print(f"  FAILED: {cam.display_name}: {e}")

# Set low resolution (matches GUI default)
print("\n--- Set 640x480 ---")
for cam in opened:
    set_resolution(cam, 640, 480)

time.sleep(2)

# Test capture_synced (same as GUI's CameraPreviewWorker)
print("\n--- capture_synced ---")
for attempt in range(5):
    frameset = capture_synced(opened)
    results = []
    for port, frame in frameset.frames.items():
        if frame is not None:
            results.append(f"cam{port}: {frame.width}x{frame.height}")
        else:
            results.append(f"cam{port}: dropped")
    print(f"  sync#{frameset.sync_index}: {' | '.join(results)} (dropped_mask={frameset.dropped_mask})")
    time.sleep(0.1)

# Test exposure
print("\n--- Exposure ---")
for exp in [-4, -6, -8]:
    for cam in opened:
        set_exposure(cam, exp)
    time.sleep(0.5)
    frameset = capture_synced(opened)
    results = []
    for port, frame in frameset.frames.items():
        results.append(f"cam{port}:{'OK' if frame else 'drop'}")
    print(f"  exp={exp}: {' '.join(results)}")

# Cleanup
print("\n--- Close ---")
for cam in opened:
    close_camera(cam)
print("Done")
