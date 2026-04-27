/*
 * pt_stream_main_mps.m - CLI test harness for MPS streaming pipeline.
 *
 * Usage:
 *   ./pt_stream_main_mps \
 *       --calibration calibration.toml \
 *       --yolo models/coreml/yolov10s.mlpackage \
 *       --vitpose models/coreml/vitpose_base.mlpackage \
 *       --num-cameras 3 \
 *       --width 640 --height 480
 *
 * Without live cameras, this creates the pipeline and immediately
 * exits to verify model loading and calibration parsing work.
 * With the camera backend, it processes live frames from cm_capture_synced().
 */

#import <Foundation/Foundation.h>
#include "pt_stream_mps.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>

static volatile int g_running = 1;

static void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
}

static void log_callback(const char *message, void *user_data) {
    (void)user_data;
    fprintf(stdout, "[LOG] %s\n", message);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        /* Parse command line args */
        const char *calibration_path = NULL;
        const char *yolo_path = NULL;
        const char *vitpose_path = NULL;
        int num_cameras = 3;
        int width = 640, height = 480;
        int max_persons = 2;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--calibration") == 0 && i + 1 < argc)
                calibration_path = argv[++i];
            else if (strcmp(argv[i], "--yolo") == 0 && i + 1 < argc)
                yolo_path = argv[++i];
            else if (strcmp(argv[i], "--vitpose") == 0 && i + 1 < argc)
                vitpose_path = argv[++i];
            else if (strcmp(argv[i], "--num-cameras") == 0 && i + 1 < argc)
                num_cameras = atoi(argv[++i]);
            else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc)
                width = atoi(argv[++i]);
            else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc)
                height = atoi(argv[++i]);
            else if (strcmp(argv[i], "--max-persons") == 0 && i + 1 < argc)
                max_persons = atoi(argv[++i]);
        }

        if (!calibration_path || !yolo_path || !vitpose_path) {
            fprintf(stderr, "Usage: %s --calibration <toml> --yolo <mlpackage> --vitpose <mlpackage>\n", argv[0]);
            fprintf(stderr, "       [--num-cameras N] [--width W] [--height H] [--max-persons N]\n");
            return 1;
        }

        /* Configure pipeline */
        PT_MPS_StreamConfig config;
        memset(&config, 0, sizeof(config));
        strncpy(config.yolo_model_path, yolo_path, sizeof(config.yolo_model_path) - 1);
        strncpy(config.vitpose_model_path, vitpose_path, sizeof(config.vitpose_model_path) - 1);
        strncpy(config.calibration_toml_path, calibration_path, sizeof(config.calibration_toml_path) - 1);
        config.num_cameras = num_cameras;
        config.frame_width = width;
        config.frame_height = height;
        config.max_persons = max_persons;
        config.log_callback = log_callback;

        /* Create pipeline */
        printf("Creating MPS streaming pipeline...\n");
        PT_MPS_Stream *stream = NULL;
        int rc = pt_mps_stream_create(&stream, &config);
        if (rc != PT_OK) {
            fprintf(stderr, "Failed to create pipeline (error %d)\n", rc);
            return 1;
        }

        printf("Pipeline created successfully!\n");
        printf("Ready for live camera frames.\n");
        printf("(No camera backend connected — exiting after initialization test)\n");

        /* In a full integration, we would loop here calling:
         *   cm_capture_synced(cameras, num_cameras, &synced_set);
         *   pt_mps_stream_process_frame(stream, &frame_set, &result);
         *
         * For now, just verify creation/destruction works. */

        signal(SIGINT, signal_handler);

        /* Print stats */
        PT_MPS_StreamStats stats;
        pt_mps_stream_get_stats(stream, &stats);
        printf("Frames processed: %d\n", stats.frames_processed);

        /* Cleanup */
        printf("Destroying pipeline...\n");
        pt_mps_stream_destroy(stream);
        printf("Done.\n");
    }

    return 0;
}
