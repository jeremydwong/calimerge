/*
 * pt_stream_main.cpp - Test harness for the streaming CUDA pose tracking pipeline.
 *
 * Reads video frames from recorded files using OpenCV, feeds them through the
 * streaming API one sync frame at a time, and reports timing statistics.
 * Optionally compares output against the batch pipeline's CSV.
 *
 * Usage:
 *   pt_stream_main.exe <recording_dir> <calibration_toml> [options]
 *
 * Options:
 *   --max-persons N      Max tracked persons (default 2)
 *   --yolo PATH          Path to YOLO ONNX model
 *   --vitpose PATH       Path to VitPose ONNX model
 *   --skip N             Process every Nth sync index (default 1)
 *   --max-frames N       Stop after N frames (0 = all, default 0)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static double harness_time_seconds(void) {
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
}
#else
#include <dirent.h>
#include <time.h>
static double harness_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/* OpenCV for video reading */
#ifdef HAS_OPENCV
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#endif

#include "pt_stream.h"

/* ============================================================================
 * Argument parsing
 * ============================================================================ */

typedef struct {
    char recording_dir[512];
    char calibration_toml[512];
    int  max_persons;
    int  skip;
    int  max_frames;
    char yolo_path[512];
    char vitpose_path[512];
} StreamCliArgs;

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <recording_dir> <calibration_toml> [options]\n"
        "\n"
        "Options:\n"
        "  --max-persons N      Max tracked persons (default 2)\n"
        "  --yolo PATH          Path to YOLO ONNX model\n"
        "  --vitpose PATH       Path to VitPose ONNX model\n"
        "  --skip N             Process every Nth sync index (default 1)\n"
        "  --max-frames N       Stop after N frames (0 = all, default 0)\n",
        prog
    );
}

static int parse_args(int argc, char **argv, StreamCliArgs *args) {
    memset(args, 0, sizeof(StreamCliArgs));
    args->max_persons = 2;
    args->skip = 1;
    args->max_frames = 0;

    if (argc < 3) {
        print_usage(argv[0]);
        return -1;
    }

    strncpy(args->recording_dir, argv[1], sizeof(args->recording_dir) - 1);
    strncpy(args->calibration_toml, argv[2], sizeof(args->calibration_toml) - 1);

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--max-persons") == 0 && i + 1 < argc) {
            args->max_persons = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--skip") == 0 && i + 1 < argc) {
            args->skip = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-frames") == 0 && i + 1 < argc) {
            args->max_frames = atoi(argv[++i]);
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
 * Video file discovery (same as pt_main.cpp)
 * ============================================================================ */

static int parse_port_from_filename(const char *filename) {
    if (strncmp(filename, "port", 4) != 0) return -1;
    const char *p = filename + 4;
    if (*p == '_') p++;
    if (*p < '0' || *p > '9') return -1;
    return atoi(p);
}

static int discover_videos(const char *dir, char video_paths[][512],
                            int *ports, int max_cameras) {
    int count = 0;

#ifdef _WIN32
    char pattern[1024];
    snprintf(pattern, sizeof(pattern), "%s\\port*.mp4", dir);
    WIN32_FIND_DATAA fdata;
    HANDLE hFind = FindFirstFileA(pattern, &fdata);
    if (hFind == INVALID_HANDLE_VALUE) {
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
    DIR *dp = opendir(dir);
    if (!dp) { fprintf(stderr, "Cannot open directory: %s\n", dir); return 0; }
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
#endif

    /* Sort by port number */
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
 * Sync table reader (minimal -- just reads sync_index, port, frame mapping)
 * ============================================================================ */

typedef struct {
    int sync_index;
    int port;
    int frame_index;
    double frame_time;
} SyncRow;

typedef struct {
    int *sync_indices;     /* unique sync indices */
    int *frame_map;        /* frame_map[sync_slot * num_cameras + cam_idx] = video_frame */
    int  num_sync_indices;
    int  num_cameras;
} SimpleSyncTable;

static int compare_rows_port_time(const void *a, const void *b) {
    const SyncRow *ra = (const SyncRow *)a;
    const SyncRow *rb = (const SyncRow *)b;
    if (ra->port != rb->port) return ra->port - rb->port;
    if (ra->frame_time < rb->frame_time) return -1;
    if (ra->frame_time > rb->frame_time) return 1;
    return 0;
}

static int load_simple_sync_table(SimpleSyncTable *table, const char *path,
                                   const int ports[], int num_cameras) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    int capacity = 4096;
    SyncRow *rows = (SyncRow *)malloc(capacity * sizeof(SyncRow));
    int num_rows = 0;

    char line[512];
    if (!fgets(line, sizeof(line), f)) { free(rows); fclose(f); return -1; }

    while (fgets(line, sizeof(line), f)) {
        int si, port, fi;
        double ft = 0.0;
        if (sscanf(line, "%d,%d,%d,%lf", &si, &port, &fi, &ft) >= 3) {
            if (num_rows >= capacity) {
                capacity *= 2;
                rows = (SyncRow *)realloc(rows, capacity * sizeof(SyncRow));
            }
            rows[num_rows].sync_index = si;
            rows[num_rows].port = port;
            rows[num_rows].frame_index = fi;
            rows[num_rows].frame_time = ft;
            num_rows++;
        }
    }
    fclose(f);

    if (num_rows == 0) { free(rows); return -1; }

    /* Derive 0-based video frame indices per camera (sorted by frame_time) */
    SyncRow *sorted = (SyncRow *)malloc(num_rows * sizeof(SyncRow));
    memcpy(sorted, rows, num_rows * sizeof(SyncRow));
    qsort(sorted, num_rows, sizeof(SyncRow), compare_rows_port_time);

    /* Assign sequential frame indices per port */
    int *derived = (int *)calloc(num_rows, sizeof(int));
    int cur_port = -1, cur_idx = -1;
    for (int i = 0; i < num_rows; i++) {
        if (sorted[i].port != cur_port) {
            cur_port = sorted[i].port;
            cur_idx = 0;
        }
        /* Find the original row for this (sync_index, port) pair */
        for (int r = 0; r < num_rows; r++) {
            if (rows[r].sync_index == sorted[i].sync_index &&
                rows[r].port == sorted[i].port) {
                derived[r] = cur_idx;
                break;
            }
        }
        cur_idx++;
    }
    free(sorted);

    /* Build unique sync indices list */
    int max_sync = num_rows;
    int *unique_sync = (int *)malloc(max_sync * sizeof(int));
    int num_unique = 0;
    for (int i = 0; i < num_rows; i++) {
        int si = rows[i].sync_index;
        int found = 0;
        for (int j = 0; j < num_unique; j++) {
            if (unique_sync[j] == si) { found = 1; break; }
        }
        if (!found) unique_sync[num_unique++] = si;
    }

    /* Sort unique sync indices */
    for (int i = 0; i < num_unique - 1; i++) {
        for (int j = i + 1; j < num_unique; j++) {
            if (unique_sync[j] < unique_sync[i]) {
                int tmp = unique_sync[i];
                unique_sync[i] = unique_sync[j];
                unique_sync[j] = tmp;
            }
        }
    }

    /* Build frame map */
    int *frame_map = (int *)malloc(num_unique * num_cameras * sizeof(int));
    for (int i = 0; i < num_unique * num_cameras; i++) frame_map[i] = -1;

    for (int r = 0; r < num_rows; r++) {
        /* Find sync slot */
        int sync_slot = -1;
        for (int s = 0; s < num_unique; s++) {
            if (unique_sync[s] == rows[r].sync_index) { sync_slot = s; break; }
        }
        if (sync_slot < 0) continue;

        /* Find camera index */
        int cam_idx = -1;
        for (int c = 0; c < num_cameras; c++) {
            if (ports[c] == rows[r].port) { cam_idx = c; break; }
        }
        if (cam_idx < 0) continue;

        frame_map[sync_slot * num_cameras + cam_idx] = derived[r];
    }

    free(derived);
    free(rows);

    table->sync_indices = unique_sync;
    table->frame_map = frame_map;
    table->num_sync_indices = num_unique;
    table->num_cameras = num_cameras;
    return 0;
}

static void free_simple_sync_table(SimpleSyncTable *table) {
    free(table->sync_indices);
    free(table->frame_map);
    memset(table, 0, sizeof(SimpleSyncTable));
}

/* ============================================================================
 * Callbacks
 * ============================================================================ */

static void log_cb(const char *message, void *user_data) {
    (void)user_data;
    printf("  %s\n", message);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char **argv) {
#ifndef HAS_OPENCV
    fprintf(stderr, "ERROR: pt_stream_main requires OpenCV (build with OPENCV_PATH set)\n");
    return 1;
#else
    printf("pt_stream_main - Streaming Pipeline Test\n");
    printf("==========================================\n\n");

    StreamCliArgs args;
    if (parse_args(argc, argv, &args) != 0) return 1;

    printf("Recording dir:  %s\n", args.recording_dir);
    printf("Calibration:    %s\n", args.calibration_toml);
    printf("Max persons:    %d\n", args.max_persons);
    printf("Skip:           %d\n", args.skip);
    printf("Max frames:     %d\n\n", args.max_frames);

    /* Discover video files */
    printf("Discovering video files...\n");
    char video_paths[PT_MAX_CAMERAS][512];
    int ports[PT_MAX_CAMERAS];
    int num_cameras = discover_videos(args.recording_dir, video_paths, ports, PT_MAX_CAMERAS);

    if (num_cameras < 2) {
        fprintf(stderr, "ERROR: Need at least 2 cameras, found %d\n", num_cameras);
        return 1;
    }
    printf("Found %d cameras\n\n", num_cameras);

    /* Open videos with OpenCV */
    printf("Opening videos with OpenCV...\n");
    cv::VideoCapture caps[PT_MAX_CAMERAS];
    int frame_w = 0, frame_h = 0;

    for (int i = 0; i < num_cameras; i++) {
        if (!caps[i].open(video_paths[i])) {
            fprintf(stderr, "ERROR: Cannot open video: %s\n", video_paths[i]);
            return 1;
        }
        int w = (int)caps[i].get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)caps[i].get(cv::CAP_PROP_FRAME_HEIGHT);
        if (i == 0) { frame_w = w; frame_h = h; }
        printf("  Port %d: %dx%d, %.0f frames\n", ports[i], w, h,
               caps[i].get(cv::CAP_PROP_FRAME_COUNT));
    }

    /* Load sync table */
    char csv_path[1024];
    snprintf(csv_path, sizeof(csv_path), "%s/frame_time_history.csv", args.recording_dir);
    printf("\nLoading sync table from %s...\n", csv_path);

    SimpleSyncTable sync_table;
    memset(&sync_table, 0, sizeof(sync_table));
    if (load_simple_sync_table(&sync_table, csv_path, ports, num_cameras) != 0) {
        fprintf(stderr, "ERROR: Failed to load sync table\n");
        return 1;
    }
    printf("Loaded %d sync indices\n\n", sync_table.num_sync_indices);

    /* Create streaming pipeline */
    PT_StreamConfig config;
    memset(&config, 0, sizeof(config));

    if (args.yolo_path[0]) {
        strncpy(config.yolo_onnx_path, args.yolo_path, 511);
    } else {
        snprintf(config.yolo_onnx_path, sizeof(config.yolo_onnx_path),
                 "%s/yolo_v10s.onnx", args.recording_dir);
    }
    if (args.vitpose_path[0]) {
        strncpy(config.vitpose_onnx_path, args.vitpose_path, 511);
    } else {
        snprintf(config.vitpose_onnx_path, sizeof(config.vitpose_onnx_path),
                 "%s/vitpose_base_coco.onnx", args.recording_dir);
    }
    snprintf(config.engine_cache_dir, sizeof(config.engine_cache_dir),
             "%s/engine_cache", args.recording_dir);
    strncpy(config.calibration_toml_path, args.calibration_toml, 511);

    config.num_cameras = num_cameras;
    config.frame_width = frame_w;
    config.frame_height = frame_h;
    config.max_persons = args.max_persons;
    config.person_confidence = 0.1f;
    config.keypoint_confidence = 0.1f;
    config.epipolar_threshold = 10.0f;
    config.max_track_distance = 0.15f;
    config.track_patience = 30;
    config.use_fp16_yolo = 1;
    config.log_callback = log_cb;

    printf("Creating streaming pipeline...\n");
    PT_Stream *stream = NULL;
    int rc = pt_stream_create(&stream, &config);
    if (rc != PT_OK) {
        fprintf(stderr, "ERROR: pt_stream_create failed with code %d\n", rc);
        free_simple_sync_table(&sync_table);
        return 1;
    }

    /* Pre-allocate frame buffers */
    cv::Mat frames[PT_MAX_CAMERAS];

    /* Track current video position per camera for seeking */
    int current_pos[PT_MAX_CAMERAS];
    for (int i = 0; i < num_cameras; i++) current_pos[i] = 0;

    /* Process sync indices */
    int total_sync = sync_table.num_sync_indices;
    int skip = args.skip;
    int frames_to_process = 0;
    for (int i = 0; i < total_sync; i += skip) frames_to_process++;
    if (args.max_frames > 0 && frames_to_process > args.max_frames) {
        frames_to_process = args.max_frames;
    }

    printf("\nProcessing %d sync frames (of %d total, skip=%d)...\n",
           frames_to_process, total_sync, skip);
    printf("--------------------------------------------\n");

    double t_total_start = harness_time_seconds();
    double t_read_total = 0.0;
    double t_process_total = 0.0;
    int processed = 0;

    for (int si = 0; si < total_sync && processed < frames_to_process; si += skip) {
        int sync_index = sync_table.sync_indices[si];

        /* Read frames from videos */
        double t_read = harness_time_seconds();

        PT_StreamFrameSet frameset;
        memset(&frameset, 0, sizeof(frameset));
        frameset.sync_index = (uint64_t)sync_index;
        frameset.num_frames = 0;

        for (int cam = 0; cam < num_cameras; cam++) {
            int target_frame = sync_table.frame_map[si * num_cameras + cam];
            if (target_frame < 0) continue;

            /* Seek to correct frame if needed */
            if (current_pos[cam] != target_frame) {
                caps[cam].set(cv::CAP_PROP_POS_FRAMES, target_frame);
                current_pos[cam] = target_frame;
            }

            if (!caps[cam].read(frames[cam])) {
                fprintf(stderr, "Warning: failed to read frame %d from camera %d\n",
                        target_frame, ports[cam]);
                continue;
            }
            current_pos[cam] = target_frame + 1;

            int fi = frameset.num_frames;
            frameset.frames[fi].pixels = frames[cam].data;
            frameset.frames[fi].width = frames[cam].cols;
            frameset.frames[fi].height = frames[cam].rows;
            frameset.frames[fi].stride = (int)frames[cam].step;
            frameset.frames[fi].port = ports[cam];
            frameset.num_frames++;
        }

        t_read_total += harness_time_seconds() - t_read;

        /* Process through streaming pipeline */
        double t_proc = harness_time_seconds();

        PT_StreamResult result;
        rc = pt_stream_process_frame(stream, &frameset, &result);

        t_process_total += harness_time_seconds() - t_proc;

        if (rc != PT_OK) {
            fprintf(stderr, "ERROR: pt_stream_process_frame failed (sync=%d, code=%d)\n",
                    sync_index, rc);
            break;
        }

        processed++;

        /* Progress every 100 frames */
        if (processed % 100 == 0 || processed == frames_to_process) {
            printf("  [%d/%d] sync=%d  persons=%d  %.1fms\n",
                   processed, frames_to_process, sync_index,
                   result.num_persons, result.processing_time_ms);
        }
    }

    double t_total = harness_time_seconds() - t_total_start;

    printf("--------------------------------------------\n\n");

    /* Get streaming stats */
    PT_StreamStats stats;
    pt_stream_get_stats(stream, &stats);

    printf("==========================================\n");
    printf("  Streaming Pipeline Statistics\n");
    printf("==========================================\n");
    printf("  Frames processed:   %8d\n", stats.frames_processed);
    printf("  Total wall time:    %8.2f s\n", t_total);
    printf("  Video read time:    %8.2f s\n", t_read_total);
    printf("  GPU process time:   %8.2f s\n", t_process_total);
    printf("\n");
    printf("  Per-frame breakdown (cumulative ms):\n");
    printf("    Upload H2D:       %8.1f ms  (%.1f ms/frame)\n",
           stats.upload_ms, stats.upload_ms / processed);
    printf("    YOLO:             %8.1f ms  (%.1f ms/frame)\n",
           stats.yolo_ms, stats.yolo_ms / processed);
    printf("    VitPose:          %8.1f ms  (%.1f ms/frame)\n",
           stats.vitpose_ms, stats.vitpose_ms / processed);
    printf("    Matching:         %8.1f ms  (%.1f ms/frame)\n",
           stats.matching_ms, stats.matching_ms / processed);
    printf("    Triangulation:    %8.1f ms  (%.1f ms/frame)\n",
           stats.triangulation_ms, stats.triangulation_ms / processed);
    printf("    Tracking:         %8.1f ms  (%.1f ms/frame)\n",
           stats.tracking_ms, stats.tracking_ms / processed);
    printf("    Total GPU+CPU:    %8.1f ms  (%.1f ms/frame)\n",
           stats.total_ms, stats.total_ms / processed);

    if (t_process_total > 0.0 && processed > 0) {
        double fps = (double)processed / t_process_total;
        printf("\n");
        printf("  Throughput (GPU only): %.1f sync-frames/s (%.1f camera-frames/s)\n",
               fps, fps * num_cameras);
        double total_fps = (double)processed / t_total;
        printf("  Throughput (incl read): %.1f sync-frames/s (%.1f camera-frames/s)\n",
               total_fps, total_fps * num_cameras);
    }
    printf("==========================================\n");

    /* Export CSV for comparison with batch pipeline */
    {
        char export_path[1024];
        snprintf(export_path, sizeof(export_path),
                 "%s/tracking_output_stream/output_3d_poses_tracked.csv",
                 args.recording_dir);
        printf("\nExporting CSV to %s ...\n", export_path);
        rc = pt_stream_export_csv(stream, export_path);
        if (rc == PT_OK) {
            printf("CSV export complete.\n");
        } else {
            fprintf(stderr, "WARNING: CSV export failed (code %d)\n", rc);
        }
    }

    /* Cleanup */
    pt_stream_destroy(stream);
    free_simple_sync_table(&sync_table);
    for (int i = 0; i < num_cameras; i++) caps[i].release();

    printf("\nDone.\n");
    return 0;
#endif /* HAS_OPENCV */
}
