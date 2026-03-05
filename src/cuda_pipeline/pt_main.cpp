/*
 * pt_main.cpp - Standalone CLI test harness for the CUDA pose tracking pipeline.
 *
 * Usage:
 *   pt_main.exe <recording_dir> <calibration_toml> [options]
 *
 * Options:
 *   --batch-size N       Sync indices per batch (default 8)
 *   --skip N             Process every Nth sync index (default 1)
 *   --max-persons N      Max tracked persons (default 2)
 *   --person-conf F      YOLO detection threshold (default 0.1)
 *   --yolo PATH          Path to YOLO ONNX model
 *   --vitpose PATH       Path to VitPose ONNX model
 *
 * The program discovers port_*.mp4 files in the recording directory,
 * runs the full pipeline, and prints timing stats.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

#include "pt_pipeline.h"

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
        "  --batch-size N       Sync indices per batch (default 8)\n"
        "  --skip N             Process every Nth sync index (default 1)\n"
        "  --max-persons N      Max tracked persons (default 2)\n"
        "  --person-conf F      YOLO detection threshold (default 0.1)\n"
        "  --yolo PATH          Path to YOLO ONNX model\n"
        "  --vitpose PATH       Path to VitPose ONNX model\n",
        prog
    );
}

static int parse_args(int argc, char **argv, CliArgs *args) {
    memset(args, 0, sizeof(CliArgs));
    args->batch_size = 8;
    args->skip = 1;
    args->max_persons = 2;
    args->person_conf = 0.1f;

    if (argc < 3) {
        print_usage(argv[0]);
        return -1;
    }

    /* Positional arguments */
    strncpy(args->recording_dir, argv[1], sizeof(args->recording_dir) - 1);
    strncpy(args->calibration_toml, argv[2], sizeof(args->calibration_toml) - 1);

    /* Optional arguments */
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
 * Video file discovery
 * ============================================================================ */

/*
 * Parse port number from a filename like "port_0.mp4" or "port0_ABC123.mp4".
 * Returns port number on success, -1 on failure.
 */
static int parse_port_from_filename(const char *filename) {
    /* Must start with "port" */
    if (strncmp(filename, "port", 4) != 0) return -1;

    const char *p = filename + 4;

    /* Skip optional underscore: "port_0" or "port0" */
    if (*p == '_') p++;

    /* Parse the number */
    if (*p < '0' || *p > '9') return -1;
    int port = atoi(p);

    return port;
}

/*
 * Discover video files matching port*.mp4 in the recording directory.
 * Fills video_paths[port] and returns the number of cameras found.
 */
static int discover_videos(
    const char *dir,
    char video_paths[][512],
    int *ports,
    int max_cameras
) {
    int count = 0;

#ifdef _WIN32
    /* Windows: FindFirstFile / FindNextFile */
    char pattern[1024];
    snprintf(pattern, sizeof(pattern), "%s\\port*.mp4", dir);

    WIN32_FIND_DATAA fdata;
    HANDLE hFind = FindFirstFileA(pattern, &fdata);
    if (hFind == INVALID_HANDLE_VALUE) {
        /* Try forward slashes */
        snprintf(pattern, sizeof(pattern), "%s/port*.mp4", dir);
        hFind = FindFirstFileA(pattern, &fdata);
    }
    if (hFind == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "No port*.mp4 files found in %s\n", dir);
        return 0;
    }

    do {
        if (count >= max_cameras) break;

        int port = parse_port_from_filename(fdata.cFileName);
        if (port < 0) continue;

        ports[count] = port;
        snprintf(video_paths[count], 512, "%s/%s", dir, fdata.cFileName);
        printf("  Camera port %d: %s\n", port, fdata.cFileName);
        count++;
    } while (FindNextFileA(hFind, &fdata));

    FindClose(hFind);
#else
    /* POSIX: opendir / readdir */
    DIR *dp = opendir(dir);
    if (!dp) {
        fprintf(stderr, "Cannot open directory: %s\n", dir);
        return 0;
    }

    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        if (count >= max_cameras) break;

        /* Check if filename starts with "port" and ends with ".mp4" */
        const char *name = entry->d_name;
        size_t len = strlen(name);
        if (len < 9) continue; /* "port0.mp4" minimum */
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
#endif

    /* Sort by port number (simple insertion sort) */
    for (int i = 1; i < count; i++) {
        int key_port = ports[i];
        char key_path[512];
        memcpy(key_path, video_paths[i], 512);

        int j = i - 1;
        while (j >= 0 && ports[j] > key_port) {
            ports[j + 1] = ports[j];
            memcpy(video_paths[j + 1], video_paths[j], 512);
            j--;
        }
        ports[j + 1] = key_port;
        memcpy(video_paths[j + 1], key_path, 512);
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
    printf("pt_main - CUDA Pose Tracking Pipeline Test\n");
    printf("============================================\n\n");

    /* Parse arguments */
    CliArgs args;
    if (parse_args(argc, argv, &args) != 0) {
        return 1;
    }

    printf("Recording dir:  %s\n", args.recording_dir);
    printf("Calibration:    %s\n", args.calibration_toml);
    printf("Batch size:     %d\n", args.batch_size);
    printf("Skip:           %d\n", args.skip);
    printf("Max persons:    %d\n", args.max_persons);
    printf("Person conf:    %.2f\n", args.person_conf);
    printf("\n");

    /* Discover video files */
    printf("Discovering video files...\n");
    char video_paths[PT_MAX_CAMERAS][512];
    int  ports[PT_MAX_CAMERAS];
    int num_cameras = discover_videos(args.recording_dir, video_paths, ports, PT_MAX_CAMERAS);

    if (num_cameras < 2) {
        fprintf(stderr, "ERROR: Need at least 2 cameras, found %d\n", num_cameras);
        return 1;
    }
    printf("Found %d cameras\n\n", num_cameras);

    /* Build pipeline config */
    PT_PipelineConfig config;
    memset(&config, 0, sizeof(config));

    config.num_cameras = num_cameras;
    for (int i = 0; i < num_cameras; i++) {
        strncpy(config.video_paths[i], video_paths[i], 511);
    }

    /* YOLO model path */
    if (args.yolo_path[0]) {
        strncpy(config.yolo_onnx_path, args.yolo_path, 511);
    } else {
        snprintf(config.yolo_onnx_path, sizeof(config.yolo_onnx_path),
                 "%s/yolo_v10s.onnx", args.recording_dir);
    }

    /* VitPose model path */
    if (args.vitpose_path[0]) {
        strncpy(config.vitpose_onnx_path, args.vitpose_path, 511);
    } else {
        snprintf(config.vitpose_onnx_path, sizeof(config.vitpose_onnx_path),
                 "%s/vitpose_base_coco.onnx", args.recording_dir);
    }

    /* Engine cache */
    snprintf(config.engine_cache_dir, sizeof(config.engine_cache_dir),
             "%s/engine_cache", args.recording_dir);

    /* Frame time CSV */
    snprintf(config.frame_time_csv_path, sizeof(config.frame_time_csv_path),
             "%s/frame_time_history.csv", args.recording_dir);

    /* Output directory */
    snprintf(config.output_dir, sizeof(config.output_dir),
             "%s/tracking_output", args.recording_dir);

    /* Processing parameters */
    config.batch_size = args.batch_size;
    config.skip_sync_indices = args.skip;
    config.max_persons = args.max_persons;
    config.person_confidence = args.person_conf;
    config.keypoint_confidence = 0.1f;
    config.epipolar_threshold = 10.0f;
    config.max_track_distance = 0.15f;
    config.track_patience = 30;
    config.use_fp16_yolo = 1;

    /* Callbacks */
    config.progress_callback = progress_cb;
    config.log_callback = log_cb;
    config.callback_user_data = NULL;

    /* Print model paths */
    printf("YOLO model:     %s\n", config.yolo_onnx_path);
    printf("VitPose model:  %s\n", config.vitpose_onnx_path);
    printf("Engine cache:   %s\n", config.engine_cache_dir);
    printf("Frame CSV:      %s\n", config.frame_time_csv_path);
    printf("Output dir:     %s\n", config.output_dir);
    printf("\n");

    /* Create pipeline */
    printf("Creating pipeline...\n");
    PT_Pipeline *pipeline = NULL;
    int rc = pt_pipeline_create(&pipeline, &config);
    if (rc != PT_OK) {
        fprintf(stderr, "ERROR: pt_pipeline_create failed with code %d\n", rc);
        return 1;
    }

    /* Load calibration */
    printf("Loading calibration...\n");
    rc = pt_pipeline_load_calibration(pipeline, args.calibration_toml);
    if (rc != PT_OK) {
        fprintf(stderr, "ERROR: pt_pipeline_load_calibration failed with code %d\n", rc);
        pt_pipeline_destroy(pipeline);
        return 1;
    }

    /* Load sync table */
    printf("Loading sync table...\n");
    rc = pt_pipeline_load_sync_table(pipeline, config.frame_time_csv_path);
    if (rc != PT_OK) {
        fprintf(stderr, "ERROR: pt_pipeline_load_sync_table failed with code %d\n", rc);
        pt_pipeline_destroy(pipeline);
        return 1;
    }

    /* Run pipeline */
    printf("\nRunning pipeline...\n");
    printf("--------------------------------------------\n");
    rc = pt_pipeline_run(pipeline);
    printf("--------------------------------------------\n");
    if (rc != PT_OK) {
        fprintf(stderr, "ERROR: pt_pipeline_run failed with code %d\n", rc);
        pt_pipeline_destroy(pipeline);
        return 1;
    }

    /* Get stats */
    PT_Stats stats;
    memset(&stats, 0, sizeof(stats));
    pt_pipeline_get_stats(pipeline, &stats);

    printf("\n");
    printf("============================================\n");
    printf("  Pipeline Statistics\n");
    printf("============================================\n");
    printf("  Total time:         %8.2f s\n", stats.total_seconds);
    printf("  Decode:             %8.2f s\n", stats.decode_seconds);
    printf("  YOLO inference:     %8.2f s\n", stats.yolo_seconds);
    printf("  VitPose inference:  %8.2f s\n", stats.vitpose_seconds);
    printf("  Matching:           %8.2f s\n", stats.matching_seconds);
    printf("  Triangulation:      %8.2f s\n", stats.triangulation_seconds);
    printf("  Export:             %8.2f s\n", stats.export_seconds);
    printf("  Frames processed:   %8d\n", stats.frames_processed);
    printf("  Persons tracked:    %8d\n", stats.persons_tracked);

    if (stats.total_seconds > 0.0 && stats.frames_processed > 0) {
        double fps = (double)stats.frames_processed / stats.total_seconds;
        printf("  Throughput:         %8.1f sync-frames/s\n", fps);
        printf("  Throughput:         %8.1f camera-frames/s\n", fps * num_cameras);
    }
    printf("============================================\n");

    /* Cleanup */
    pt_pipeline_destroy(pipeline);

    printf("\nDone.\n");
    return 0;
}
