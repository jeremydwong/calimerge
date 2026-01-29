/*
 * test_multi.c - Test simultaneous capture from multiple cameras
 *
 * Compile: clang -o test_multi test_multi.c -L. -lcalimerge -Wl,-rpath,.
 * Run: ./test_multi
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

int main(void) {
    printf("Calimerge Multi-Camera Capture Test\n");
    printf("====================================\n\n");

    int result = cm_init();
    if (result != CM_OK) {
        printf("Failed to initialize: %d\n", result);
        return 1;
    }

    /* Enumerate cameras */
    CM_Camera cameras[CM_MAX_CAMERAS];
    int count = cm_enumerate_cameras(cameras, CM_MAX_CAMERAS);

    printf("Found %d camera(s)\n\n", count);

    if (count < 2) {
        printf("Need at least 2 cameras for multi-camera test\n");
        cm_shutdown();
        return 1;
    }

    /* Open all cameras */
    printf("Opening cameras...\n");
    for (int i = 0; i < count; i++) {
        printf("  %d: %s (%s)... ", i, cameras[i].display_name, cameras[i].serial_number);
        result = cm_open_camera(&cameras[i]);
        if (result == CM_OK) {
            printf("OK\n");
        } else {
            printf("FAILED (%d)\n", result);
        }
    }

    /* Warm up */
    printf("\nWarming up...\n");
    sleep(2);

    /* Capture synced frames */
    printf("\nCapturing 20 synced frame sets...\n\n");
    printf("Note: 'corrected spread' shows sync quality (should be <10ms at 30fps)\n");
    printf("      'pts spread' is raw camera timestamps (different clock domains)\n\n");

    for (int s = 0; s < 20; s++) {
        CM_SyncedFrameSet frameset;
        result = cm_capture_synced(cameras, count, &frameset);

        printf("Sync %2llu: ", frameset.sync_index);

        for (int i = 0; i < count; i++) {
            CM_Frame *f = &frameset.frames[i];
            if (f->pixels) {
                printf("[cam%d: pts=%.1f corr=%.1f] ",
                       i, f->timestamp_ns / 1e6, f->corrected_ns / 1e6);
            } else {
                printf("[cam%d: DROPPED] ", i);
            }
        }

        /* Calculate CORRECTED timestamp spread (meaningful across cameras) */
        if (frameset.frame_count >= 2) {
            uint64_t min_corr = UINT64_MAX, max_corr = 0;
            for (int i = 0; i < count; i++) {
                if (frameset.frames[i].pixels) {
                    if (frameset.frames[i].corrected_ns < min_corr)
                        min_corr = frameset.frames[i].corrected_ns;
                    if (frameset.frames[i].corrected_ns > max_corr)
                        max_corr = frameset.frames[i].corrected_ns;
                }
            }
            printf("(spread: %.2f ms)", (max_corr - min_corr) / 1e6);
        }
        printf("\n");

        cm_release_synced(&frameset);

        /* Wait ~1 frame period before next capture */
        usleep(33000);  /* ~30 fps */
    }

    /* Close cameras */
    printf("\nClosing cameras...\n");
    for (int i = 0; i < count; i++) {
        cm_close_camera(&cameras[i]);
    }

    cm_shutdown();
    printf("Done.\n");
    return 0;
}
