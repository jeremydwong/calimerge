/*
 * pt_main_mps.m - Standalone CLI smoke test for the MPS offline pipeline.
 *
 * Usage:
 *   pt_main_mps <recording_dir> <calibration_toml> [options]
 *
 * Options:
 *   --batch-size N       Sync indices per CoreML batch (default 1)
 *   --skip N             Process every Nth sync index (default 1)
 *   --max-persons N      Max tracked persons (default 2)
 *   --person-conf F      YOLO detection threshold (default 0.1)
 *   --yolo PATH          Path to YOLO .mlpackage / .mlmodelc
 *   --vitpose PATH       Path to VitPose .mlpackage / .mlmodelc
 *
 * Mirrors src/cuda_pipeline/pt_main.cpp. Discovers port_*.mp4 files in
 * the recording directory and runs the full offline pipeline against them.
 */

#import <Foundation/Foundation.h>
#include "pt_offline_mps.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

/* ============================================================================
 * Argument parsing
 * ============================================================================ */

typedef struct {
    char recording_dir[512];
    char calibration_toml[512];
    int  batch_size;
    int  skip;
    int  max_persons;
    float person_conf;
    char yolo_path[512];
    char vitpose_path[512];
} CliArgs;

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <recording_dir> <calibration_toml> [options]\n"
        "\n"
        "Options:\n"
        "  --batch-size N       Sync indices per CoreML batch (default 1)\n"
        "  --skip N             Process every Nth sync index (default 1)\n"
        "  --max-persons N      Max tracked persons (default 2)\n"
        "  --person-conf F      YOLO detection threshold (default 0.1)\n"
        "  --yolo PATH          Path to YOLO .mlpackage\n"
        "  --vitpose PATH       Path to VitPose .mlpackage\n",
        prog
    );
}

static int parse_args(int argc, char **argv, CliArgs *args) {
    memset(args, 0, sizeof(CliArgs));
    args->batch_size = 1;
    args->skip = 1;
    args->max_persons = 2;
    args->person_conf = 0.1f;

    if (argc < 3) {
        print_usage(argv[0]);
        return -1;
    }

    strncpy(args->recording_dir, argv[1], sizeof(args->recording_dir) - 1);
    strncpy(args->calibration_toml, argv[2], sizeof(args->calibration_toml) - 1);

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            args->batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--skip") == 0 && i + 1 < argc) {
            args->skip = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-persons") == 0 && i + 1 < argc) {
            args->max_persons = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--person-conf") == 0 && i + 1 < argc) {
            args->person_conf = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--yolo") == 0 && i + 1 < argc) {
            strncpy(args->yolo_path, argv[++i], sizeof(args->yolo_path) - 1);
        } else if (strcmp(argv[i], "--vitpose") == 0 && i + 1 < argc) {
            strncpy(args->vitpose_path, argv[++i], sizeof(args->vitpose_path) - 1);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return -1;
        }
    }
    return 0;
}

/* ============================================================================
 * port_*.mp4 discovery
 * ============================================================================ */

static int parse_port_from_filename(const char *filename) {
    if (strncmp(filename, "port", 4) != 0) return -1;
    const char *p = filename + 4;
    if (*p == '_') p++;
    if (*p < '0' || *p > '9') return -1;
    return atoi(p);
}

static int discover_videos(const char *dir,
                            char video_paths[][512],
                            int *ports,
                            int max_cameras) {
    int count = 0;
    DIR *dp = opendir(dir);
    if (!dp) {
        fprintf(stderr, "Cannot open directory: %s\n", dir);
        return 0;
    }

    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        if (count >= max_cameras) break;
        const char *name = entry->d_name;
        size_t len = strlen(name);
        if (len < 9) continue;
        if (strncmp(name, "port", 4) != 0) continue;
        if (strcmp(name + len - 4, ".mp4") != 0) continue;

        int port = parse_port_from_filename(name);
        if (port < 0) continue;

        ports[count] = port;
        snprintf(video_paths[count], 512, "%s/%s", dir, name);
        printf("  Camera port %d: %s\n", port, name);
        count++;
    }
    closedir(dp);

    /* Insertion-sort by port */
    for (int i = 1; i < count; i++) {
        int kp = ports[i];
        char kpath[512];
        memcpy(kpath, video_paths[i], 512);
        int j = i - 1;
        while (j >= 0 && ports[j] > kp) {
            ports[j + 1] = ports[j];
            memcpy(video_paths[j + 1], video_paths[j], 512);
            j--;
        }
        ports[j + 1] = kp;
        memcpy(video_paths[j + 1], kpath, 512);
    }

    return count;
}

/* ============================================================================
 * Callbacks
 * ============================================================================ */

static void progress_cb(const char *step, float fraction, void *user_data) {
    (void)user_data;
    printf("[%5.1f%%] %s\n", fraction * 100.0f, step);
}

static void log_cb(const char *message, void *user_data) {
    (void)user_data;
    printf("  %s\n", message);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char **argv) {
    @autoreleasepool {
        printf("pt_main_mps - MPS / CoreML Offline Pose Tracking Pipeline\n");
        printf("============================================================\n\n");

        CliArgs args;
        if (parse_args(argc, argv, &args) != 0) return 1;

        printf("Recording dir:  %s\n", args.recording_dir);
        printf("Calibration:    %s\n", args.calibration_toml);
        printf("Batch size:     %d\n", args.batch_size);
        printf("Skip:           %d\n", args.skip);
        printf("Max persons:    %d\n", args.max_persons);
        printf("Person conf:    %.2f\n", args.person_conf);
        printf("\n");

        printf("Discovering port_*.mp4 files...\n");
        char video_paths[PT_MAX_CAMERAS][512];
        int  ports[PT_MAX_CAMERAS];
        int num_cameras = discover_videos(args.recording_dir, video_paths,
                                           ports, PT_MAX_CAMERAS);

        if (num_cameras < 2) {
            fprintf(stderr, "ERROR: Need at least 2 cameras, found %d\n", num_cameras);
            return 1;
        }
        printf("Found %d cameras\n\n", num_cameras);

        /* Build config */
        PT_MPS_OfflineConfig config;
        memset(&config, 0, sizeof(config));
        config.num_cameras = num_cameras;
        for (int i = 0; i < num_cameras; i++) {
            strncpy(config.video_paths[i], video_paths[i], 511);
            config.ports[i] = ports[i];
        }

        if (args.yolo_path[0]) {
            strncpy(config.yolo_model_path, args.yolo_path, 511);
        } else {
            snprintf(config.yolo_model_path, sizeof(config.yolo_model_path),
                     "%s/yolo_v10s.mlpackage", args.recording_dir);
        }
        if (args.vitpose_path[0]) {
            strncpy(config.vitpose_model_path, args.vitpose_path, 511);
        } else {
            snprintf(config.vitpose_model_path, sizeof(config.vitpose_model_path),
                     "%s/vitpose_synthpose.mlpackage", args.recording_dir);
        }
        strncpy(config.calibration_toml_path, args.calibration_toml, 511);

        snprintf(config.frame_time_csv_path, sizeof(config.frame_time_csv_path),
                 "%s/frame_time_history.csv", args.recording_dir);
        snprintf(config.output_dir, sizeof(config.output_dir),
                 "%s/tracking_output", args.recording_dir);

        config.batch_size = args.batch_size;
        config.skip_sync_indices = args.skip;
        config.max_persons = args.max_persons;
        config.person_confidence = args.person_conf;
        config.keypoint_confidence = 0.1f;
        config.epipolar_threshold = 10.0f;
        config.max_track_distance = 0.5f;
        config.track_patience = 60;

        config.progress_callback = progress_cb;
        config.log_callback = log_cb;
        config.callback_user_data = NULL;

        printf("YOLO model:     %s\n", config.yolo_model_path);
        printf("VitPose model:  %s\n", config.vitpose_model_path);
        printf("Frame CSV:      %s\n", config.frame_time_csv_path);
        printf("Output dir:     %s\n", config.output_dir);
        printf("\n");

        printf("Creating offline pipeline...\n");
        PT_MPS_Offline *pipeline = NULL;
        int rc = pt_mps_offline_create(&pipeline, &config);
        if (rc != PT_OK) {
            fprintf(stderr, "ERROR: pt_mps_offline_create failed (%d)\n", rc);
            return 1;
        }

        printf("\nRunning pipeline...\n");
        printf("------------------------------------------------------------\n");
        rc = pt_mps_offline_run(pipeline);
        printf("------------------------------------------------------------\n");
        if (rc != PT_OK) {
            fprintf(stderr, "ERROR: pt_mps_offline_run failed (%d)\n", rc);
            pt_mps_offline_destroy(pipeline);
            return 1;
        }

        PT_MPS_OfflineStats stats;
        memset(&stats, 0, sizeof(stats));
        pt_mps_offline_get_stats(pipeline, &stats);

        printf("\n");
        printf("============================================================\n");
        printf("  Pipeline Statistics\n");
        printf("============================================================\n");
        printf("  Total time:        %8.2f s\n", stats.total_seconds);
        printf("  Decode:            %8.2f s\n", stats.decode_seconds);
        printf("  Inference:         %8.2f s\n", stats.inference_seconds);
        printf("  Matching:          %8.2f s\n", stats.matching_seconds);
        printf("  Triangulation:     %8.2f s\n", stats.triangulation_seconds);
        printf("  Export:            %8.2f s\n", stats.export_seconds);
        printf("  Frames processed:  %8d\n", stats.frames_processed);
        if (stats.total_seconds > 0 && stats.frames_processed > 0) {
            double fps = stats.frames_processed / stats.total_seconds;
            printf("  Throughput:        %8.1f sync-frames/s\n", fps);
            printf("  Throughput:        %8.1f camera-frames/s\n", fps * num_cameras);
        }
        printf("============================================================\n");

        pt_mps_offline_destroy(pipeline);
        printf("\nDone.\n");
    }
    return 0;
}
