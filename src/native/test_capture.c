/*
 * test_capture.c - Test frame capture from camera
 *
 * Compile: clang -o test_capture test_capture.c -L. -lcalimerge -Wl,-rpath,.
 * Run: ./test_capture
 */

#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#define sleep(s) Sleep((s) * 1000)
#define usleep(us) Sleep((us) / 1000)
#else
#include <unistd.h>
#endif
#include "calimerge_platform.h"

int main(int argc, char *argv[]) {
    int camera_index = 0;
    int num_frames = 10;

    if (argc > 1) {
        camera_index = atoi(argv[1]);
    }
    if (argc > 2) {
        num_frames = atoi(argv[2]);
    }

    printf("Calimerge Frame Capture Test\n");
    printf("============================\n");
    printf("Camera index: %d, Frames to capture: %d\n\n", camera_index, num_frames);

    int result = cm_init();
    if (result != CM_OK) {
        printf("Failed to initialize: %d\n", result);
        return 1;
    }

    /* Enumerate cameras */
    CM_Camera cameras[CM_MAX_CAMERAS];
    int count = cm_enumerate_cameras(cameras, CM_MAX_CAMERAS);

    if (count <= camera_index) {
        printf("Camera %d not found (only %d cameras available)\n", camera_index, count);
        cm_shutdown();
        return 1;
    }

    CM_Camera *cam = &cameras[camera_index];
    printf("Opening camera: %s (%s)\n", cam->display_name, cam->serial_number);
    printf("Resolution: %dx%d @ %d fps\n\n", cam->width, cam->height, cam->fps);

    result = cm_open_camera(cam);
    if (result != CM_OK) {
        printf("Failed to open camera: %d\n", result);
        cm_shutdown();
        return 1;
    }

    /* Let camera warm up */
    printf("Warming up camera...\n");
    sleep(1);

    /* Capture frames */
    printf("Capturing %d frames...\n\n", num_frames);

    uint64_t first_ts = 0;
    uint64_t last_ts = 0;

    for (int i = 0; i < num_frames; i++) {
        CM_Frame frame;
        result = cm_capture_frame(cam, &frame);

        if (result == CM_OK) {
            if (first_ts == 0) first_ts = frame.timestamp_ns;
            last_ts = frame.timestamp_ns;

            double elapsed_ms = (frame.timestamp_ns - first_ts) / 1e6;
            printf("Frame %2d: %dx%d, stride=%d, ts=%.2f ms",
                   i, frame.width, frame.height, frame.stride, elapsed_ms);

            /* Print first few pixel values (BGR) */
            if (frame.pixels) {
                int mid = (frame.height / 2) * frame.stride + (frame.width / 2) * 3;
                printf(", center pixel BGR=(%d,%d,%d)",
                       frame.pixels[mid], frame.pixels[mid+1], frame.pixels[mid+2]);
            }
            printf("\n");

            cm_release_frame(&frame);
        } else {
            printf("Frame %2d: FAILED (%d)\n", i, result);
        }
    }

    /* Calculate average FPS */
    if (num_frames > 1 && last_ts > first_ts) {
        double duration_s = (last_ts - first_ts) / 1e9;
        double avg_fps = (num_frames - 1) / duration_s;
        printf("\nAverage FPS: %.2f\n", avg_fps);
    }

    cm_close_camera(cam);
    cm_shutdown();

    printf("\nDone.\n");
    return 0;
}
