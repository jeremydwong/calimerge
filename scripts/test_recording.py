"""Test recording functionality using capture_synced (same as GUI)."""
import sys
import os
import time

# Check what's available in video_recorder
from calimerge import video_recorder
print("video_recorder exports:", [x for x in dir(video_recorder) if not x.startswith('_')])

from calimerge.camera_binding import (
    init, enumerate_cameras, open_camera, close_camera,
    set_resolution, capture_synced,
)

init()
cameras = enumerate_cameras()
print(f"\nFound {len(cameras)} cameras")

# Open all
for cam in cameras:
    open_camera(cam)
    set_resolution(cam, 640, 480)
    print(f"  Opened: {cam.display_name} -> {cam.width}x{cam.height}")

time.sleep(0.5)

# Test basic capture first
print("\n--- Testing capture_synced ---")
for i in range(3):
    fs = capture_synced(cameras)
    if fs:
        cams_ok = []
        for port, frame in fs.frames.items():
            if frame:
                cams_ok.append(f"cam{port}:{frame.width}x{frame.height}")
        print(f"  sync#{i}: {' | '.join(cams_ok)} (dropped={fs.dropped_mask})")

# Record 2 seconds using direct cv2 writers
import cv2
import numpy as np

out_dir = os.path.join("recordings", "test_record")
os.makedirs(out_dir, exist_ok=True)

try:
    writers = {}
    for i in range(len(cameras)):
        path = os.path.join(out_dir, f"port_{i}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writers[i] = cv2.VideoWriter(path, fourcc, 30, (640, 480))
        print(f"  Writer for port {i}: {path}")

    print(f"\nRecording 2 seconds to {out_dir}...")
    frame_counts = {i: 0 for i in range(len(cameras))}
    for _ in range(60):
        fs = capture_synced(cameras)
        if fs:
            for port, frame in fs.frames.items():
                if frame and port in writers:
                    img = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(
                        frame.height, frame.width, 3)
                    writers[port].write(img)
                    frame_counts[port] += 1
        time.sleep(1.0/30)

    for w in writers.values():
        w.release()

    print("Recording stopped.")
    for i, count in frame_counts.items():
        path = os.path.join(out_dir, f"port_{i}.mp4")
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"  port_{i}.mp4: {size:,} bytes ({count} frames)")

except Exception as e:
    print(f"\nRecording error: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
for cam in cameras:
    close_camera(cam)
print("\nDone.")
