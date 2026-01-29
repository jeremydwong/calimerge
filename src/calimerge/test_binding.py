#!/usr/bin/env python3
"""
Test script for calimerge camera bindings.
"""

import time
from camera_binding import (
    init, shutdown, enumerate_cameras,
    open_camera, close_camera, capture_frame,
    capture_synced, Camera
)

def main():
    print("Calimerge Python Binding Test")
    print("=" * 40)
    print()

    # Initialize
    init()

    # Enumerate cameras
    cameras = enumerate_cameras()
    print(f"Found {len(cameras)} camera(s):\n")

    for cam in cameras:
        print(f"  {cam.device_index}: {cam.display_name}")
        print(f"      Serial: {cam.serial_number}")
        print(f"      Resolution: {cam.width}x{cam.height} @ {cam.fps} fps")
        print(f"      Supported: {cam.supported_resolutions}")
        print()

    if not cameras:
        print("No cameras found!")
        shutdown()
        return

    # Test single camera capture with context manager
    print("Testing single camera capture...")
    with Camera(cameras[0]) as cam:
        print(f"  Opened: {cam.name}")

        # Warm up
        time.sleep(0.5)

        # Capture a few frames
        for i in range(5):
            frame = cam.capture()
            print(f"  Frame {i}: {frame.width}x{frame.height}, "
                  f"ts={frame.timestamp_ms:.2f} ms, "
                  f"center BGR={tuple(frame.pixels[frame.height//2, frame.width//2])}")

    print()

    # Test multi-camera capture if we have multiple cameras
    if len(cameras) >= 2:
        print("Testing multi-camera capture...")

        # Open all cameras
        for cam in cameras:
            open_camera(cam)
            print(f"  Opened: {cam.display_name}")

        time.sleep(1)  # Warm up

        # Capture synced frames
        for i in range(5):
            frameset = capture_synced(cameras)
            print(f"  Sync {frameset.sync_index}:", end="")

            timestamps = []
            for idx, frame in frameset.frames.items():
                if frame:
                    print(f" [cam{idx}: {frame.timestamp_ms:.1f}ms]", end="")
                    timestamps.append(frame.timestamp_ns)
                else:
                    print(f" [cam{idx}: DROPPED]", end="")

            if len(timestamps) >= 2:
                spread_ms = (max(timestamps) - min(timestamps)) / 1e6
                print(f" (spread: {spread_ms:.2f}ms)")
            else:
                print()

        # Close all cameras
        for cam in cameras:
            close_camera(cam)

    print()
    shutdown()
    print("Done!")

if __name__ == "__main__":
    main()
