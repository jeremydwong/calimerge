/*
 * test_sync_log.c - Test synchronized capture with timestamp CSV logging
 *
 * This test captures frames from multiple cameras and logs all timestamps
 * to a CSV file for analysis of clock synchronization quality.
 *
 * Compile: clang -o test_sync_log test_sync_log.c -L. -lcalimerge -Wl,-rpath,.
 * Run: ./test_sync_log
 *
 * Output: sync_timestamps.csv with columns:
 *   sync_index, camera, pts_ns, arrival_ns, corrected_ns, offset_from_mean_ns
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "calimerge_platform.h"

#define NUM_CAPTURES 100

int main(void) {
    printf("Calimerge Synchronization Logging Test\n");
    printf("======================================\n\n");

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
        printf("Need at least 2 cameras for sync test\n");
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
    printf("\nWarming up (2 seconds)...\n");
    sleep(2);

    /* Open CSV file */
    FILE *csv = fopen("sync_timestamps.csv", "w");
    if (!csv) {
        printf("Failed to open CSV file\n");
        for (int i = 0; i < count; i++) cm_close_camera(&cameras[i]);
        cm_shutdown();
        return 1;
    }

    /* Write CSV header */
    fprintf(csv, "sync_index,camera,pts_ns,arrival_ns,corrected_ns,offset_from_mean_ns,spread_ns\n");

    printf("\nCapturing %d synced frame sets...\n\n", NUM_CAPTURES);

    uint64_t total_spread_ns = 0;
    uint64_t min_spread_ns = UINT64_MAX;
    uint64_t max_spread_ns = 0;

    for (int s = 0; s < NUM_CAPTURES; s++) {
        CM_SyncedFrameSet frameset;
        result = cm_capture_synced(cameras, count, &frameset);

        if (result != CM_OK) {
            printf("Sync %d: CAPTURE FAILED (%d)\n", s, result);
            continue;
        }

        /* Calculate mean corrected timestamp */
        uint64_t sum_corrected = 0;
        int valid = 0;
        for (int i = 0; i < count; i++) {
            CM_Frame *f = &frameset.frames[i];
            if (f->pixels) {
                sum_corrected += f->corrected_ns;
                valid++;
            }
        }
        uint64_t mean_corrected = (valid > 0) ? (sum_corrected / valid) : 0;

        /* Calculate spread (max - min of corrected timestamps) */
        uint64_t min_corrected = UINT64_MAX, max_corrected = 0;
        for (int i = 0; i < count; i++) {
            CM_Frame *f = &frameset.frames[i];
            if (f->pixels) {
                if (f->corrected_ns < min_corrected) min_corrected = f->corrected_ns;
                if (f->corrected_ns > max_corrected) max_corrected = f->corrected_ns;
            }
        }
        uint64_t spread_ns = (valid >= 2) ? (max_corrected - min_corrected) : 0;

        /* Log to CSV */
        for (int i = 0; i < count; i++) {
            CM_Frame *f = &frameset.frames[i];
            if (f->pixels) {
                int64_t offset = (int64_t)f->corrected_ns - (int64_t)mean_corrected;
                fprintf(csv, "%llu,%d,%llu,%llu,%llu,%lld,%llu\n",
                        frameset.sync_index, i,
                        f->timestamp_ns, f->arrival_ns, f->corrected_ns,
                        offset, spread_ns);
            } else {
                fprintf(csv, "%llu,%d,DROPPED,DROPPED,DROPPED,0,%llu\n",
                        frameset.sync_index, i, spread_ns);
            }
        }

        /* Print summary */
        printf("Sync %3llu: ", frameset.sync_index);
        for (int i = 0; i < count; i++) {
            CM_Frame *f = &frameset.frames[i];
            if (f->pixels) {
                printf("[cam%d: pts=%.2f corr=%.2f] ",
                       i, f->timestamp_ns / 1e6, f->corrected_ns / 1e6);
            } else {
                printf("[cam%d: DROPPED] ", i);
            }
        }
        printf("(spread: %.2f ms)\n", spread_ns / 1e6);

        /* Track statistics */
        total_spread_ns += spread_ns;
        if (spread_ns < min_spread_ns) min_spread_ns = spread_ns;
        if (spread_ns > max_spread_ns) max_spread_ns = spread_ns;

        cm_release_synced(&frameset);

        /* ~30 fps pacing */
        usleep(33000);
    }

    fclose(csv);

    /* Print summary statistics */
    printf("\n========================================\n");
    printf("Synchronization Statistics (%d captures)\n", NUM_CAPTURES);
    printf("========================================\n");
    printf("  Mean spread:  %.2f ms\n", (total_spread_ns / NUM_CAPTURES) / 1e6);
    printf("  Min spread:   %.2f ms\n", min_spread_ns / 1e6);
    printf("  Max spread:   %.2f ms\n", max_spread_ns / 1e6);
    printf("\nTimestamp log saved to: sync_timestamps.csv\n");

    /* Close cameras */
    printf("\nClosing cameras...\n");
    for (int i = 0; i < count; i++) {
        cm_close_camera(&cameras[i]);
    }

    cm_shutdown();
    printf("Done.\n");
    return 0;
}
