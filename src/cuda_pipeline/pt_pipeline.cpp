/*
 * pt_pipeline.cpp - CUDA pose tracking pipeline orchestrator.
 *
 * Ties together all Phase 1 modules into a single batch-processing loop:
 *   decode -> NV12->BGR -> YOLO -> VitPose -> matching -> triangulation -> tracking
 *
 * The pipeline uses a single CUDA stream for v1 simplicity.  All GPU work is
 * serialized on that stream, with async CPU<->GPU transfers for overlap.
 *
 * TOML parsing and CSV parsing are hand-rolled for the specific formats
 * this pipeline reads -- no external dependencies beyond the C standard library.
 *
 * Style: Plain C structs + free functions. No classes, no templates, no STL.
 */

#include "pt_pipeline.h"
#include "pt_arena.h"
#include "pt_nvdec.h"
#include "pt_tensorrt.h"
#include "pt_kernels.h"
#include "pt_matching.h"
#include "pt_triangulation.h"
#include "pt_tracker.h"
#include "pt_export.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

/* High-resolution timing */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifdef _WIN32
static double pt_time_seconds(void) {
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double pt_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/* ============================================================================
 * Internal pipeline struct (opaque to callers)
 * ============================================================================ */

struct PT_Pipeline {
    PT_PipelineConfig config;
    PT_GpuArena arena;
    PT_CameraConstants constants;
    PT_SyncTable sync_table;
    PT_TrackState tracks;
    PT_Stats stats;

    PT_TrtEngine yolo_engine;
    PT_TrtEngine vitpose_engine;

    PT_VideoDecoder decoders[PT_MAX_CAMERAS];

    cudaStream_t stream;

    /* Scratch buffers for CPU-side processing per sync index */
    PT_Detection2D detections[PT_MAX_CAMERAS][PT_MAX_DETECTIONS];
    PT_Group groups[PT_MAX_GROUPS];
    PT_CandidateGroup candidate_groups[PT_MAX_GROUPS];

    int is_initialized;
};

/* ============================================================================
 * Logging helper
 * ============================================================================ */

static void pipeline_log(const PT_Pipeline *p, const char *fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (p->config.log_callback) {
        p->config.log_callback(buf, p->config.callback_user_data);
    }
    fprintf(stderr, "[pt_pipeline] %s\n", buf);
}

static void pipeline_progress(const PT_Pipeline *p, const char *step, float fraction) {
    if (p->config.progress_callback) {
        p->config.progress_callback(step, fraction, p->config.callback_user_data);
    }
}

/* ============================================================================
 * TOML parser -- hand-rolled for the calibration file format
 *
 * Only handles:
 *   [camera_N]         section headers
 *   key = value        where value is int, string, flat array, or nested array
 *
 * Nested arrays like [[1,2,3],[4,5,6],[7,8,9]] are parsed as flat row-major.
 * ============================================================================ */

/* Trim leading whitespace in-place, return pointer to first non-space char */
static const char *skip_ws(const char *s) {
    while (*s == ' ' || *s == '\t') s++;
    return s;
}

/* Parse a flat or nested array of doubles from the string starting at *pos.
 * Writes values to out[], returns count of values parsed.
 * Handles: [1, 2, 3] and [[1, 2], [3, 4]] (flattened to [1,2,3,4]). */
static int parse_double_array(const char *s, double *out, int max_count) {
    int count = 0;
    const char *p = s;

    /* Skip to first '[' */
    while (*p && *p != '[') p++;
    if (!*p) return 0;
    p++; /* skip '[' */

    while (*p && count < max_count) {
        /* Skip whitespace and commas */
        while (*p == ' ' || *p == '\t' || *p == ',' || *p == '\n' || *p == '\r') p++;

        if (*p == ']') {
            p++;
            /* If next non-whitespace is also ']', we've closed the outer array */
            while (*p == ' ' || *p == '\t') p++;
            if (*p == ']') break; /* end of outer ]] */
            /* Otherwise check if there's a comma and more data */
            continue;
        }

        if (*p == '[') {
            /* Nested array -- recurse into it */
            p++;
            continue;
        }

        if (*p == '\0') break;

        /* Try to parse a number */
        char *end;
        double val = strtod(p, &end);
        if (end > p) {
            out[count++] = val;
            p = end;
        } else {
            /* Not a number, skip character */
            p++;
        }
    }

    return count;
}

/* Parse a single integer value from "key = 123" after the '=' */
static int parse_int_value(const char *s) {
    const char *p = strchr(s, '=');
    if (!p) return 0;
    p++;
    return atoi(p);
}

/* Parse a quoted string value from 'key = "value"' after the '=' */
static int parse_string_value(const char *s, char *out, int max_len) {
    const char *p = strchr(s, '=');
    if (!p) return 0;
    p++;
    /* Find opening quote */
    while (*p && *p != '"') p++;
    if (!*p) return 0;
    p++; /* skip opening quote */
    int i = 0;
    while (*p && *p != '"' && i < max_len - 1) {
        out[i++] = *p++;
    }
    out[i] = '\0';
    return i;
}

/* Rodrigues vector to 3x3 rotation matrix.
 * r[3] is the axis-angle vector; theta = ||r||.
 * R = cos(t)I + (1-cos(t))r_hat*r_hat^T + sin(t)[r_hat]_x */
static void rodrigues_to_matrix(const double r[3], double R[3][3]) {
    double theta = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    if (theta < 1e-12) {
        /* Identity */
        R[0][0] = 1; R[0][1] = 0; R[0][2] = 0;
        R[1][0] = 0; R[1][1] = 1; R[1][2] = 0;
        R[2][0] = 0; R[2][1] = 0; R[2][2] = 1;
        return;
    }
    double rx = r[0]/theta, ry = r[1]/theta, rz = r[2]/theta;
    double c = cos(theta), s = sin(theta), t = 1.0 - c;
    R[0][0] = c + rx*rx*t;     R[0][1] = rx*ry*t - rz*s; R[0][2] = rx*rz*t + ry*s;
    R[1][0] = ry*rx*t + rz*s;  R[1][1] = c + ry*ry*t;    R[1][2] = ry*rz*t - rx*s;
    R[2][0] = rz*rx*t - ry*s;  R[2][1] = rz*ry*t + rx*s; R[2][2] = c + rz*rz*t;
}

/* Try to start accumulating an array value. Returns 1 if complete on this line. */
static int try_parse_array_inline(const char *trimmed, char *array_buf, int buf_size, int *in_array) {
    const char *eq = strchr(trimmed, '=');
    if (!eq) return 0;
    eq = skip_ws(eq + 1);
    strncpy(array_buf, eq, buf_size - 1);
    array_buf[buf_size - 1] = '\0';
    int open = 0, close = 0;
    for (const char *c = array_buf; *c; c++) {
        if (*c == '[') open++;
        if (*c == ']') close++;
    }
    if (close >= open && open > 0) return 1; /* complete */
    *in_array = 1;
    return 0;
}

static int load_calibration_toml(PT_CameraConstants *constants, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return PT_ERR_FILE_NOT_FOUND;

    char line[2048];
    int current_cam = -1;
    int num_cameras_found = 0;

    /* We need to handle multi-line arrays. Accumulate lines for array values. */
    char array_buf[4096];
    int in_array = 0;   /* 1 = accumulating a multi-line array */
    int array_type = 0; /* 0=camera_matrix, 1=distortion, 2=rotation, 3=translation */

    /* Track whether rotation was given as Rodrigues (3 values) or matrix (9 values).
     * Caliscope uses Rodrigues; calimerge uses 3x3 matrix. */
    int rotation_is_rodrigues[PT_MAX_CAMERAS];
    memset(rotation_is_rodrigues, 0, sizeof(rotation_is_rodrigues));

    while (fgets(line, sizeof(line), f)) {
        /* If we are accumulating a multi-line array, append this line */
        if (in_array) {
            size_t cur_len = strlen(array_buf);
            size_t line_len = strlen(line);
            if (cur_len + line_len < sizeof(array_buf) - 1) {
                memcpy(array_buf + cur_len, line, line_len + 1);
            }
            /* Check if the array is complete: count brackets */
            int open = 0, close = 0;
            for (const char *c = array_buf; *c; c++) {
                if (*c == '[') open++;
                if (*c == ']') close++;
            }
            if (close >= open && open > 0) {
                /* Array complete, parse it */
                double vals[16];
                int n = parse_double_array(array_buf, vals, 16);

                if (current_cam >= 0 && current_cam < PT_MAX_CAMERAS) {
                    if (array_type == 0 && n >= 9) {
                        for (int r = 0; r < 3; r++)
                            for (int c = 0; c < 3; c++)
                                constants->camera_matrix[current_cam][r][c] = vals[r * 3 + c];
                    } else if (array_type == 1 && n >= 5) {
                        for (int i = 0; i < 5; i++)
                            constants->distortion[current_cam][i] = vals[i];
                    } else if (array_type == 2) {
                        if (n >= 9) {
                            for (int r = 0; r < 3; r++)
                                for (int c = 0; c < 3; c++)
                                    constants->rotation[current_cam][r][c] = vals[r * 3 + c];
                        } else if (n == 3) {
                            rotation_is_rodrigues[current_cam] = 1;
                            rodrigues_to_matrix(vals, constants->rotation[current_cam]);
                        }
                    } else if (array_type == 3 && n >= 3) {
                        for (int i = 0; i < 3; i++)
                            constants->translation[current_cam][i] = vals[i];
                    }
                }
                in_array = 0;
            }
            continue;
        }

        const char *trimmed = skip_ws(line);

        /* Skip empty lines and comments */
        if (*trimmed == '\0' || *trimmed == '\n' || *trimmed == '\r' || *trimmed == '#') {
            continue;
        }

        /* Section header: [camera_N] or [cam_N] (caliscope format) */
        if (*trimmed == '[') {
            int cam_idx = -1;
            if (sscanf(trimmed, "[camera_%d]", &cam_idx) == 1 ||
                sscanf(trimmed, "[cam_%d]", &cam_idx) == 1) {
                if (cam_idx >= 0 && cam_idx < PT_MAX_CAMERAS) {
                    current_cam = cam_idx;
                    /* Default port to section index (matches Python cs_parse.py:648
                     * which infers port from cam_N suffix). May be overridden by
                     * explicit 'port' key below. */
                    constants->ports[current_cam] = cam_idx;
                    if (cam_idx >= num_cameras_found) {
                        num_cameras_found = cam_idx + 1;
                    }
                }
            }
            continue;
        }

        /* Key-value pairs within a camera section */
        if (current_cam < 0) continue;

        if (strncmp(trimmed, "port", 4) == 0 && (trimmed[4] == ' ' || trimmed[4] == '=')) {
            constants->ports[current_cam] = parse_int_value(trimmed);
        } else if (strncmp(trimmed, "size", 4) == 0 && (trimmed[4] == ' ' || trimmed[4] == '=')) {
            /* Caliscope format: size = [640, 480] — store per-camera */
            double vals[2];
            const char *eq = strchr(trimmed, '=');
            if (eq && parse_double_array(eq, vals, 2) >= 2) {
                constants->cam_width[current_cam] = (int)vals[0];
                constants->cam_height[current_cam] = (int)vals[1];
                /* Also update common frame dimensions (last camera wins,
                 * overridden later from video files if available) */
                constants->frame_width = (int)vals[0];
                constants->frame_height = (int)vals[1];
            }
        } else if (strncmp(trimmed, "camera_matrix", 13) == 0 ||
                   (strncmp(trimmed, "matrix", 6) == 0 && trimmed[6] != '_' &&
                    (trimmed[6] == ' ' || trimmed[6] == '='))) {
            /* "camera_matrix" (calimerge) or "matrix" (caliscope) */
            array_type = 0;
            if (try_parse_array_inline(trimmed, array_buf, sizeof(array_buf), &in_array)) {
                double vals[16];
                int n = parse_double_array(array_buf, vals, 16);
                if (n >= 9 && current_cam < PT_MAX_CAMERAS) {
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++)
                            constants->camera_matrix[current_cam][r][c] = vals[r * 3 + c];
                }
            }
        } else if (strncmp(trimmed, "distortion", 10) == 0) {
            /* "distortion" or "distortions" (caliscope) */
            array_type = 1;
            if (try_parse_array_inline(trimmed, array_buf, sizeof(array_buf), &in_array)) {
                double vals[8];
                int n = parse_double_array(array_buf, vals, 8);
                if (n >= 5 && current_cam < PT_MAX_CAMERAS) {
                    for (int i = 0; i < 5; i++)
                        constants->distortion[current_cam][i] = vals[i];
                }
            }
        } else if (strncmp(trimmed, "rotation", 8) == 0 && trimmed[8] != '_') {
            array_type = 2;
            if (try_parse_array_inline(trimmed, array_buf, sizeof(array_buf), &in_array)) {
                double vals[16];
                int n = parse_double_array(array_buf, vals, 16);
                if (current_cam < PT_MAX_CAMERAS) {
                    if (n >= 9) {
                        for (int r = 0; r < 3; r++)
                            for (int c = 0; c < 3; c++)
                                constants->rotation[current_cam][r][c] = vals[r * 3 + c];
                    } else if (n == 3) {
                        rotation_is_rodrigues[current_cam] = 1;
                        rodrigues_to_matrix(vals, constants->rotation[current_cam]);
                    }
                }
            }
        } else if (strncmp(trimmed, "translation", 11) == 0) {
            array_type = 3;
            if (try_parse_array_inline(trimmed, array_buf, sizeof(array_buf), &in_array)) {
                double vals[4];
                int n = parse_double_array(array_buf, vals, 4);
                if (n >= 3 && current_cam < PT_MAX_CAMERAS) {
                    for (int i = 0; i < 3; i++)
                        constants->translation[current_cam][i] = vals[i];
                }
            }
        }
    }

    fclose(f);

    if (num_cameras_found <= 0) {
        fprintf(stderr, "[pt_pipeline] No cameras found in TOML file: %s\n", path);
        return PT_ERR_INVALID_PARAM;
    }

    constants->num_cameras = num_cameras_found;

    /* Log Rodrigues conversions */
    for (int i = 0; i < num_cameras_found; i++) {
        if (rotation_is_rodrigues[i]) {
            fprintf(stderr, "[pt_pipeline] Camera %d: converted Rodrigues rotation to 3x3 matrix\n", i);
        }
    }

    return PT_OK;
}

/* ============================================================================
 * Sync table CSV parser
 *
 * Format:
 *   sync_index,port,frame_index,frame_time
 *   73361,0,0,0.033
 *   73361,1,0,0.034
 *
 * Build mapping: for each unique sync_index, store the frame_index per camera.
 * ============================================================================ */

/* Temporary structure for reading CSV rows before building the packed table */
typedef struct {
    int sync_index;
    int port;
    int frame_index;         /* raw frame_index from CSV (camera counter, NOT video position) */
    double frame_time;       /* frame_time from CSV, used to derive 0-based video position */
    int derived_frame_index; /* 0-based video position per camera (computed from frame_time rank) */
} SyncRow;

static int compare_sync_rows(const void *a, const void *b) {
    const SyncRow *ra = (const SyncRow *)a;
    const SyncRow *rb = (const SyncRow *)b;
    if (ra->sync_index != rb->sync_index) return ra->sync_index - rb->sync_index;
    return ra->port - rb->port;
}

/* Compare SyncRows by (port, frame_time) for deriving per-camera frame index */
static int compare_by_port_time(const void *a, const void *b) {
    const SyncRow *ra = (const SyncRow *)a;
    const SyncRow *rb = (const SyncRow *)b;
    if (ra->port != rb->port) return ra->port - rb->port;
    if (ra->frame_time < rb->frame_time) return -1;
    if (ra->frame_time > rb->frame_time) return 1;
    return 0;
}

static int load_sync_table_csv(PT_SyncTable *table, const char *path, int num_cameras,
                                const int ports[], int num_ports) {
    FILE *f = fopen(path, "r");
    if (!f) return PT_ERR_FILE_NOT_FOUND;

    /* First pass: count lines (overestimate) */
    int capacity = 4096;
    SyncRow *rows = (SyncRow *)malloc(capacity * sizeof(SyncRow));
    if (!rows) { fclose(f); return PT_ERR_OUT_OF_MEMORY; }
    int num_rows = 0;

    char line[512];
    /* Skip header */
    if (!fgets(line, sizeof(line), f)) {
        free(rows);
        fclose(f);
        return PT_ERR_INVALID_PARAM;
    }

    while (fgets(line, sizeof(line), f)) {
        int sync_idx, port, frame_idx;
        double frame_time = 0.0;
        if (sscanf(line, "%d,%d,%d,%lf", &sync_idx, &port, &frame_idx, &frame_time) >= 3) {
            if (num_rows >= capacity) {
                capacity *= 2;
                SyncRow *new_rows = (SyncRow *)realloc(rows, capacity * sizeof(SyncRow));
                if (!new_rows) { free(rows); fclose(f); return PT_ERR_OUT_OF_MEMORY; }
                rows = new_rows;
            }
            rows[num_rows].sync_index = sync_idx;
            rows[num_rows].port = port;
            rows[num_rows].frame_index = frame_idx;
            rows[num_rows].frame_time = frame_time;
            rows[num_rows].derived_frame_index = -1;
            num_rows++;
        }
    }
    fclose(f);

    if (num_rows == 0) {
        free(rows);
        return PT_ERR_INVALID_PARAM;
    }

    /* Derive 0-based video frame index per camera.
     *
     * Python equivalent (from process_image_batches.py):
     *   df = df.sort_values(by=['port', 'frame_time'])
     *   df['derived_frame_index'] = df.groupby('port')['frame_time'].rank(method='min').astype(int) - 1
     *
     * The raw frame_index in the CSV is the camera's internal counter (e.g. 73338),
     * NOT a 0-based position in the video file.  The video file contains frames
     * sequentially from 0..N-1, so we need to compute the rank of each frame
     * within its camera, ordered by frame_time.
     */
    qsort(rows, num_rows, sizeof(SyncRow), compare_by_port_time);

    {
        int i = 0;
        while (i < num_rows) {
            /* Find range of rows with the same port */
            int port_start = i;
            int current_port = rows[i].port;
            while (i < num_rows && rows[i].port == current_port) {
                i++;
            }
            /* Assign 0-based rank within this port (already sorted by frame_time) */
            int rank = 0;
            for (int j = port_start; j < i; j++) {
                /* rank(method='min'): equal frame_times get the same rank */
                if (j > port_start && rows[j].frame_time != rows[j - 1].frame_time) {
                    rank = j - port_start;
                }
                rows[j].derived_frame_index = rank;
            }
        }
    }

    /* Sort by (sync_index, port) for building the packed table */
    qsort(rows, num_rows, sizeof(SyncRow), compare_sync_rows);

    /* Count unique sync indices */
    int num_unique = 1;
    for (int i = 1; i < num_rows; i++) {
        if (rows[i].sync_index != rows[i - 1].sync_index) {
            num_unique++;
        }
    }

    /* Build port-index lookup: port number -> camera index (0..num_cameras-1).
     * Ports in the calibration may not be 0,1,2,... so we map them. */
    int port_to_cam[256];
    memset(port_to_cam, -1, sizeof(port_to_cam));
    for (int i = 0; i < num_ports && i < PT_MAX_CAMERAS; i++) {
        int port_num = ports[i];
        if (port_num >= 0 && port_num < 256) {
            port_to_cam[port_num] = i;
        }
    }

    /* Allocate sync table arrays */
    table->num_sync_indices = num_unique;
    table->num_cameras = num_cameras;
    table->sync_indices = (int *)malloc(num_unique * sizeof(int));
    table->sync_to_frame = (int *)malloc(num_unique * num_cameras * sizeof(int));

    if (!table->sync_indices || !table->sync_to_frame) {
        free(table->sync_indices);
        free(table->sync_to_frame);
        free(rows);
        table->sync_indices = NULL;
        table->sync_to_frame = NULL;
        return PT_ERR_OUT_OF_MEMORY;
    }

    /* Initialize all frame indices to -1 (missing) */
    for (int i = 0; i < num_unique * num_cameras; i++) {
        table->sync_to_frame[i] = -1;
    }

    /* Fill in the table */
    int sync_slot = 0;
    table->sync_indices[0] = rows[0].sync_index;
    for (int i = 0; i < num_rows; i++) {
        /* Advance sync slot if new sync_index */
        if (i > 0 && rows[i].sync_index != rows[i - 1].sync_index) {
            sync_slot++;
            table->sync_indices[sync_slot] = rows[i].sync_index;
        }

        int cam_idx = -1;
        if (rows[i].port >= 0 && rows[i].port < 256) {
            cam_idx = port_to_cam[rows[i].port];
        }
        if (cam_idx >= 0 && cam_idx < num_cameras) {
            table->sync_to_frame[sync_slot * num_cameras + cam_idx] = rows[i].derived_frame_index;
        }
    }

    free(rows);
    return PT_OK;
}

static void free_sync_table(PT_SyncTable *table) {
    free(table->sync_to_frame);
    free(table->sync_indices);
    table->sync_to_frame = NULL;
    table->sync_indices = NULL;
    table->num_sync_indices = 0;
    table->num_cameras = 0;
}

/* ============================================================================
 * Pipeline lifecycle
 * ============================================================================ */

extern "C" int pt_pipeline_create(PT_Pipeline **out, const PT_PipelineConfig *config) {
    if (!out || !config) return PT_ERR_INVALID_PARAM;

    PT_Pipeline *p = (PT_Pipeline *)calloc(1, sizeof(PT_Pipeline));
    if (!p) return PT_ERR_OUT_OF_MEMORY;

    /* Copy configuration */
    memcpy(&p->config, config, sizeof(PT_PipelineConfig));

    /* Apply defaults for unset parameters */
    if (p->config.batch_size <= 0) p->config.batch_size = 8;
    if (p->config.batch_size > PT_BATCH_SIZE_MAX) p->config.batch_size = PT_BATCH_SIZE_MAX;
    if (p->config.skip_sync_indices <= 0) p->config.skip_sync_indices = 1;
    if (p->config.max_persons <= 0) p->config.max_persons = PT_MAX_PERSONS;
    if (p->config.person_confidence <= 0.0f) p->config.person_confidence = 0.1f;
    if (p->config.keypoint_confidence <= 0.0f) p->config.keypoint_confidence = 0.1f;
    if (p->config.epipolar_threshold <= 0.0f) p->config.epipolar_threshold = 10.0f;
    if (p->config.max_track_distance <= 0.0f) p->config.max_track_distance = 0.15f;
    if (p->config.track_patience <= 0) p->config.track_patience = 30;

    /* Initialize track state */
    pt_track_init(&p->tracks);

    /* Zero out stats */
    memset(&p->stats, 0, sizeof(PT_Stats));

    *out = p;
    return PT_OK;
}

extern "C" void pt_pipeline_destroy(PT_Pipeline *p) {
    if (!p) return;

    /* Close video decoders */
    for (int i = 0; i < PT_MAX_CAMERAS; i++) {
        pt_video_close(&p->decoders[i]);
    }

    /* Destroy TensorRT engines */
    pt_trt_destroy_engine(&p->yolo_engine);
    pt_trt_destroy_engine(&p->vitpose_engine);

    /* Destroy CUDA stream */
    if (p->stream) {
        cudaStreamDestroy(p->stream);
        p->stream = NULL;
    }

    /* Free GPU arena */
    pt_arena_destroy(&p->arena);

    /* Free sync table */
    free_sync_table(&p->sync_table);

    /* Shutdown TensorRT runtime */
    pt_trt_shutdown();

    free(p);
}

/* ============================================================================
 * Load calibration
 * ============================================================================ */

extern "C" int pt_pipeline_load_calibration(PT_Pipeline *p, const char *toml_path) {
    if (!p || !toml_path) return PT_ERR_INVALID_PARAM;

    memset(&p->constants, 0, sizeof(PT_CameraConstants));

    int rc = load_calibration_toml(&p->constants, toml_path);
    if (rc != PT_OK) {
        fprintf(stderr, "[pt_pipeline] Failed to load calibration from: %s (error %d)\n", toml_path, rc);
        return rc;
    }

    /* Compute derived matrices */
    pt_compute_projection_matrices(&p->constants);
    pt_precompute_fundamentals(&p->constants);

    pipeline_log(p, "Loaded calibration for %d cameras from %s", p->constants.num_cameras, toml_path);

    return PT_OK;
}

/* ============================================================================
 * Standalone calibration loader (shared with pt_stream)
 * ============================================================================ */

extern "C" int pt_load_calibration(PT_CameraConstants *constants, const char *toml_path) {
    if (!constants || !toml_path) return PT_ERR_INVALID_PARAM;

    memset(constants, 0, sizeof(PT_CameraConstants));

    int rc = load_calibration_toml(constants, toml_path);
    if (rc != PT_OK) return rc;

    pt_compute_projection_matrices(constants);
    pt_precompute_fundamentals(constants);

    return PT_OK;
}

/* ============================================================================
 * Load sync table
 * ============================================================================ */

extern "C" int pt_pipeline_load_sync_table(PT_Pipeline *p, const char *csv_path) {
    if (!p || !csv_path) return PT_ERR_INVALID_PARAM;

    /* Free any existing sync table */
    free_sync_table(&p->sync_table);

    int rc = load_sync_table_csv(&p->sync_table, csv_path,
                                  p->constants.num_cameras,
                                  p->constants.ports, p->constants.num_cameras);
    if (rc != PT_OK) {
        fprintf(stderr, "[pt_pipeline] Failed to load sync table from: %s (error %d)\n", csv_path, rc);
        return rc;
    }

    pipeline_log(p, "Loaded sync table: %d sync indices, %d cameras from %s",
                 p->sync_table.num_sync_indices, p->sync_table.num_cameras, csv_path);

    return PT_OK;
}

/* ============================================================================
 * Get statistics
 * ============================================================================ */

extern "C" void pt_pipeline_get_stats(const PT_Pipeline *p, PT_Stats *out) {
    if (!p || !out) return;
    memcpy(out, &p->stats, sizeof(PT_Stats));
}

/* ============================================================================
 * Main pipeline run
 *
 * Processing loop per batch of sync_indices:
 *   1. DECODE: decode one frame per camera per sync_index -> NV12
 *   2. NV12->BGR: colorspace convert all frames
 *   3. YOLO: letterbox + infer + filter detections
 *   4. Copy detection counts to CPU
 *   5. VITPOSE: crop+normalize + infer + heatmap decode
 *   6. Copy 2D keypoints to CPU
 *   7. cudaStreamSynchronize
 *   8. CPU: matching + triangulation + tracking for each sync_index
 * ============================================================================ */

extern "C" int pt_pipeline_run(PT_Pipeline *p) {
    if (!p) return PT_ERR_INVALID_PARAM;
    if (p->constants.num_cameras <= 0) {
        fprintf(stderr, "[pt_pipeline] No calibration loaded. Call pt_pipeline_load_calibration() first.\n");
        return PT_ERR_NOT_INITIALIZED;
    }
    if (p->sync_table.num_sync_indices <= 0) {
        fprintf(stderr, "[pt_pipeline] No sync table loaded. Call pt_pipeline_load_sync_table() first.\n");
        return PT_ERR_NOT_INITIALIZED;
    }

    int rc;
    double t_start = pt_time_seconds();
    double t_phase;

    int num_cameras = p->constants.num_cameras;
    int batch_size = p->config.batch_size;
    int skip = p->config.skip_sync_indices;

    /* --- Step 0: Open video decoders and determine frame dimensions --- */

    pipeline_log(p, "Opening %d video decoders...", num_cameras);
    int frame_w = 0, frame_h = 0;

    for (int i = 0; i < num_cameras; i++) {
        memset(&p->decoders[i], 0, sizeof(PT_VideoDecoder));
        rc = pt_video_open(&p->decoders[i], p->config.video_paths[i]);

        if (rc != PT_OK) {
            fprintf(stderr, "[pt_pipeline] Failed to open video %d: %s (error %d)\n",
                    i, p->config.video_paths[i], rc);
            return rc;
        }

        int w, h;
        pt_video_get_dimensions(&p->decoders[i], &w, &h);

        if (i == 0) {
            frame_w = w;
            frame_h = h;
        } else if (w != frame_w || h != frame_h) {
            fprintf(stderr, "[pt_pipeline] Warning: camera %d has different dimensions (%dx%d vs %dx%d). "
                    "Using first camera's dimensions.\n", i, w, h, frame_w, frame_h);
        }
    }

    p->constants.frame_width = frame_w;
    p->constants.frame_height = frame_h;

    pipeline_log(p, "Video dimensions: %dx%d, %d cameras", frame_w, frame_h, num_cameras);

    /* --- Step 1: Create CUDA stream --- */

    cudaError_t cerr = cudaStreamCreate(&p->stream);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "[pt_pipeline] cudaStreamCreate failed: %s\n", cudaGetErrorString(cerr));
        return PT_ERR_CUDA;
    }

    /* --- Step 2: Initialize GPU arena --- */

    pipeline_log(p, "Initializing GPU arena (batch_size=%d)...", batch_size);
    rc = pt_arena_init(&p->arena, num_cameras, frame_w, frame_h, batch_size);
    if (rc != PT_OK) {
        fprintf(stderr, "[pt_pipeline] Arena init failed (error %d)\n", rc);
        return rc;
    }
    pt_arena_print_stats(&p->arena);

    /* --- Step 3: Build/load TensorRT engines --- */

    pipeline_log(p, "Initializing TensorRT runtime...");
    rc = pt_trt_init();
    if (rc != PT_OK) {
        fprintf(stderr, "[pt_pipeline] TensorRT init failed (error %d)\n", rc);
        return rc;
    }

    /* YOLO engine: max batch = batch_size * num_cameras */
    int yolo_max_batch = batch_size * num_cameras;
    pipeline_log(p, "Building YOLO engine (max_batch=%d, fp16=%d)...",
                 yolo_max_batch, p->config.use_fp16_yolo);
    memset(&p->yolo_engine, 0, sizeof(PT_TrtEngine));
    rc = pt_trt_build_engine(&p->yolo_engine, p->config.yolo_onnx_path,
                              p->config.engine_cache_dir, yolo_max_batch,
                              p->config.use_fp16_yolo);
    if (rc != PT_OK) {
        fprintf(stderr, "[pt_pipeline] YOLO engine build failed (error %d)\n", rc);
        return rc;
    }
    pipeline_log(p, "YOLO engine ready.");

    /* VitPose engine: max batch = num_cameras * PT_MAX_DETECTIONS.
     * VitPose is called per sync_index (inside the bi loop), so the max
     * crops in a single inference call is num_cameras * max_dets_per_image,
     * NOT batch_size * num_cameras * max_dets (which would be too large
     * for TensorRT to build). */
    int vitpose_max_batch = num_cameras * PT_MAX_DETECTIONS;
    pipeline_log(p, "Building VitPose engine (max_batch=%d)...", vitpose_max_batch);
    memset(&p->vitpose_engine, 0, sizeof(PT_TrtEngine));
    rc = pt_trt_build_engine(&p->vitpose_engine, p->config.vitpose_onnx_path,
                              p->config.engine_cache_dir, vitpose_max_batch, 1);
    if (rc != PT_OK) {
        fprintf(stderr, "[pt_pipeline] VitPose engine build failed (error %d)\n", rc);
        return rc;
    }
    pipeline_log(p, "VitPose engine ready.");

    /* --- Step 4: Compute letterbox parameters --- */

    float lb_scale = fminf((float)PT_YOLO_INPUT_W / frame_w,
                           (float)PT_YOLO_INPUT_H / frame_h);
    int lb_new_w = (int)(frame_w * lb_scale);
    int lb_new_h = (int)(frame_h * lb_scale);
    int lb_pad_x = (PT_YOLO_INPUT_W - lb_new_w) / 2;
    int lb_pad_y = (PT_YOLO_INPUT_H - lb_new_h) / 2;

    pipeline_log(p, "Letterbox: scale=%.4f, pad=(%d,%d), new_size=(%d,%d)",
                 lb_scale, lb_pad_x, lb_pad_y, lb_new_w, lb_new_h);

    /* --- Step 5: Determine which sync indices to process --- */

    int total_sync = p->sync_table.num_sync_indices;
    int process_count = 0;
    for (int i = 0; i < total_sync; i += skip) {
        process_count++;
    }

    pipeline_log(p, "Processing %d of %d sync indices (skip=%d) in batches of %d",
                 process_count, total_sync, skip, batch_size);

    /* --- Step 6: Main batch processing loop --- */

    /* Timing accumulators for each phase (seconds) */
    double t_decode_total = 0.0;
    double t_yolo_total = 0.0;
    double t_vitpose_total = 0.0;
    double t_matching_total = 0.0;
    double t_triangulation_total = 0.0;
    int frames_processed = 0;

    /* Build list of sync indices to process */
    int *process_indices = (int *)malloc(process_count * sizeof(int));
    if (!process_indices) return PT_ERR_OUT_OF_MEMORY;
    {
        int idx = 0;
        for (int i = 0; i < total_sync; i += skip) {
            process_indices[idx++] = i; /* index into sync_table, not the sync_index value */
        }
    }

    int num_batches = (process_count + batch_size - 1) / batch_size;

    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int batch_start = batch_idx * batch_size;
        int batch_end = batch_start + batch_size;
        if (batch_end > process_count) batch_end = process_count;
        int this_batch_size = batch_end - batch_start;
        int num_images = this_batch_size * num_cameras;

        /* ================================================================
         * 6a. DECODE: For each sync_index in batch, decode one frame per camera
         * ================================================================ */

        t_phase = pt_time_seconds();

        for (int bi = 0; bi < this_batch_size; bi++) {
            int sync_slot = process_indices[batch_start + bi];

            /* Gather frame indices for each camera at this sync_index */
            int frame_indices[PT_MAX_CAMERAS];
            uint8_t *nv12_ptrs[PT_MAX_CAMERAS];
            for (int cam = 0; cam < num_cameras; cam++) {
                frame_indices[cam] = p->sync_table.sync_to_frame[sync_slot * num_cameras + cam];
                /* Use pipeline depth slot 0 (single-buffered for v1) */
                nv12_ptrs[cam] = p->arena.decoded_nv12[0][cam];
            }

            rc = pt_video_decode_batch(p->decoders, num_cameras, frame_indices,
                                        nv12_ptrs, p->stream);
            /* Non-fatal: some frames may be missing. Continue processing. */

            /* ============================================================
             * 6b. NV12->BGR: colorspace convert all frames from this sync_index
             * ============================================================ */

            for (int cam = 0; cam < num_cameras; cam++) {
                if (frame_indices[cam] < 0) continue; /* no frame for this camera */

                int nv12_stride = frame_w; /* NV12 Y plane stride = width for most codecs */
                int bgr_stride = frame_w * 3;

                pt_launch_nv12_to_bgr(
                    p->arena.decoded_nv12[0][cam],
                    p->arena.decoded_bgr[0][cam],
                    frame_w, frame_h,
                    nv12_stride, bgr_stride,
                    p->stream
                );
            }
        }

        t_decode_total += pt_time_seconds() - t_phase;

        /* ================================================================
         * 6c. YOLO: letterbox all images, run TensorRT, filter detections
         *
         * Build an array of BGR pointers for all images in this batch
         * (batch_size * num_cameras images).
         * ================================================================ */

        t_phase = pt_time_seconds();

        /* Collect all BGR image pointers for this batch.
         * For v1 (single pipeline depth slot), we re-decode per sync_index above
         * but only have arena space for num_cameras BGR buffers.
         * So we process each sync_index's YOLO independently within the batch.
         *
         * NOTE: In a more optimized version, we'd have arena slots for the
         * full batch.  For v1, we process sync indices sequentially within
         * a batch for the GPU-heavy stages, then batch the CPU math. */

        /* We need to collect per-image detection results across the batch
         * for the CPU math phase.  Store detection data per (sync_offset, cam). */

        /* Temporary: per-image host detection data for the whole batch */
        int batch_det_counts[PT_BATCH_SIZE_MAX * PT_MAX_CAMERAS];
        memset(batch_det_counts, 0, sizeof(batch_det_counts));

        /* We process YOLO + VitPose per sync_index since arena only has
         * single-frame-per-camera buffers in the decode slots. */
        for (int bi = 0; bi < this_batch_size; bi++) {
            int sync_slot = process_indices[batch_start + bi];

            /* Redecode if bi > 0 (first iteration already decoded above).
             * This is the v1 simplicity tradeoff -- arena has only one set
             * of decode buffers.  A future version would expand the arena. */
            if (bi > 0) {
                double t_redecode = pt_time_seconds();
                int frame_indices[PT_MAX_CAMERAS];
                uint8_t *nv12_ptrs[PT_MAX_CAMERAS];
                for (int cam = 0; cam < num_cameras; cam++) {
                    frame_indices[cam] = p->sync_table.sync_to_frame[sync_slot * num_cameras + cam];
                    nv12_ptrs[cam] = p->arena.decoded_nv12[0][cam];
                }
                pt_video_decode_batch(p->decoders, num_cameras, frame_indices,
                                       nv12_ptrs, p->stream);
                for (int cam = 0; cam < num_cameras; cam++) {
                    if (frame_indices[cam] < 0) continue;
                    pt_launch_nv12_to_bgr(
                        p->arena.decoded_nv12[0][cam],
                        p->arena.decoded_bgr[0][cam],
                        frame_w, frame_h,
                        frame_w, frame_w * 3,
                        p->stream
                    );
                }
                t_decode_total += pt_time_seconds() - t_redecode;
            }

            /* Build GPU pointer array for letterbox (one per camera) */
            uint8_t *bgr_ptrs[PT_MAX_CAMERAS];
            int valid_image_count = 0;
            int image_cam_map[PT_MAX_CAMERAS]; /* image index -> camera index */

            for (int cam = 0; cam < num_cameras; cam++) {
                int fi = p->sync_table.sync_to_frame[sync_slot * num_cameras + cam];
                if (fi >= 0) {
                    bgr_ptrs[valid_image_count] = p->arena.decoded_bgr[0][cam];
                    image_cam_map[valid_image_count] = cam;
                    valid_image_count++;
                }
            }

            if (valid_image_count == 0) {
                frames_processed++;
                continue;
            }

            /* --- YOLO phase timing --- */
            double t_yolo_start = pt_time_seconds();

            /* Upload BGR pointer array to GPU for the letterbox kernel */
            uint8_t **d_bgr_ptrs;
            cudaMalloc(&d_bgr_ptrs, valid_image_count * sizeof(uint8_t *));
            cudaMemcpyAsync(d_bgr_ptrs, bgr_ptrs, valid_image_count * sizeof(uint8_t *),
                            cudaMemcpyHostToDevice, p->stream);

            /* Letterbox + normalize -> YOLO input */
            pt_launch_letterbox_batch(
                d_bgr_ptrs, p->arena.yolo_input,
                valid_image_count,
                frame_w, frame_h,
                PT_YOLO_INPUT_W, PT_YOLO_INPUT_H,
                p->stream
            );

            /* YOLO inference */
            pt_trt_infer(&p->yolo_engine, p->arena.yolo_input, p->arena.yolo_output,
                          valid_image_count, p->stream);

            /* Filter detections: threshold + undo letterbox */
            pt_launch_filter_detections(
                p->arena.yolo_output,
                p->arena.detection_boxes,
                p->arena.detection_scores,
                p->arena.detection_counts,
                valid_image_count,
                p->config.person_confidence,
                lb_scale, lb_pad_x, lb_pad_y,
                frame_w, frame_h,
                p->stream
            );

            /* Copy detection counts to CPU to know how many VitPose crops to run */
            cudaMemcpyAsync(p->arena.host_detection_counts,
                            p->arena.detection_counts,
                            valid_image_count * sizeof(int),
                            cudaMemcpyDeviceToHost, p->stream);

            /* Synchronize to get detection counts on CPU */
            cudaStreamSynchronize(p->stream);

            t_yolo_total += pt_time_seconds() - t_yolo_start;

            /* ============================================================
             * 6d. VITPOSE: crop+normalize for each detected person, infer, decode
             * ============================================================ */

            double t_vp_start = pt_time_seconds();

            /* Compute total crops for this sync_index */
            int total_crops = 0;
            int crop_offsets[PT_MAX_CAMERAS]; /* per-image start offset into VitPose batch */
            for (int img = 0; img < valid_image_count; img++) {
                crop_offsets[img] = total_crops;
                int det_count = p->arena.host_detection_counts[img];
                if (det_count > PT_MAX_DETECTIONS) det_count = PT_MAX_DETECTIONS;
                total_crops += det_count;
            }

            if (total_crops > 0) {
                /* For each image, launch crop+normalize for its detections */
                for (int img = 0; img < valid_image_count; img++) {
                    int det_count = p->arena.host_detection_counts[img];
                    if (det_count <= 0) continue;
                    if (det_count > PT_MAX_DETECTIONS) det_count = PT_MAX_DETECTIONS;

                    int cam = image_cam_map[img];

                    /* Boxes for this image start at detection_boxes[img * MAX_DET * 4] */
                    float *img_boxes = p->arena.detection_boxes + img * PT_MAX_DETECTIONS * 4;

                    /* VitPose input/affine for these crops start at the crop offset */
                    int crop_start = crop_offsets[img];
                    float *vp_input = p->arena.vitpose_input +
                                      crop_start * 3 * PT_VITPOSE_INPUT_H * PT_VITPOSE_INPUT_W;
                    float *vp_affine = p->arena.vitpose_affine + crop_start * 6; /* 2x3 = 6 floats */

                    pt_launch_crop_normalize_vitpose(
                        p->arena.decoded_bgr[0][cam],
                        frame_w, frame_h,
                        img_boxes, det_count,
                        vp_input, vp_affine,
                        p->stream
                    );
                }

                /* VitPose inference on all crops at once */
                pt_trt_infer(&p->vitpose_engine, p->arena.vitpose_input,
                              p->arena.vitpose_heatmaps, total_crops, p->stream);

                /* Heatmap decode: heatmaps -> 2D keypoints */
                pt_launch_heatmap_decode(
                    p->arena.vitpose_heatmaps,
                    p->arena.vitpose_affine,
                    p->arena.keypoints_2d,
                    total_crops,
                    p->stream
                );

                /* Copy results to CPU (pinned memory for async transfer) */
                cudaMemcpyAsync(p->arena.host_keypoints_2d,
                                p->arena.keypoints_2d,
                                total_crops * PT_NUM_KEYPOINTS * 3 * sizeof(float),
                                cudaMemcpyDeviceToHost, p->stream);

                cudaMemcpyAsync(p->arena.host_detection_boxes,
                                p->arena.detection_boxes,
                                valid_image_count * PT_MAX_DETECTIONS * 4 * sizeof(float),
                                cudaMemcpyDeviceToHost, p->stream);

                cudaMemcpyAsync(p->arena.host_detection_scores,
                                p->arena.detection_scores,
                                valid_image_count * PT_MAX_DETECTIONS * sizeof(float),
                                cudaMemcpyDeviceToHost, p->stream);
            }

            /* Synchronize: all GPU work for this sync_index is complete */
            cudaStreamSynchronize(p->stream);

            t_vitpose_total += pt_time_seconds() - t_vp_start;

            cudaFree(d_bgr_ptrs);

            /* ============================================================
             * 6e. CPU MATH: Build detections, match, triangulate, track
             * ============================================================ */

            double t_match_start = pt_time_seconds();

            /* Build PT_Detection2D arrays from host keypoints */
            int det_counts_per_cam[PT_MAX_CAMERAS];
            memset(p->detections, 0, sizeof(p->detections));
            memset(det_counts_per_cam, 0, sizeof(det_counts_per_cam));

            for (int img = 0; img < valid_image_count; img++) {
                int cam = image_cam_map[img];
                int det_count = p->arena.host_detection_counts[img];
                if (det_count <= 0) continue;
                if (det_count > PT_MAX_DETECTIONS) det_count = PT_MAX_DETECTIONS;

                det_counts_per_cam[cam] = det_count;

                for (int d = 0; d < det_count; d++) {
                    PT_Detection2D *det = &p->detections[cam][d];
                    det->valid = 1;

                    /* Copy bounding box */
                    int box_offset = img * PT_MAX_DETECTIONS * 4 + d * 4;
                    det->bbox[0] = p->arena.host_detection_boxes[box_offset + 0];
                    det->bbox[1] = p->arena.host_detection_boxes[box_offset + 1];
                    det->bbox[2] = p->arena.host_detection_boxes[box_offset + 2];
                    det->bbox[3] = p->arena.host_detection_boxes[box_offset + 3];

                    /* Copy person confidence */
                    int score_offset = img * PT_MAX_DETECTIONS + d;
                    det->person_confidence = p->arena.host_detection_scores[score_offset];

                    /* Copy 2D keypoints */
                    int crop_idx = crop_offsets[img] + d;
                    float *kp_base = p->arena.host_keypoints_2d +
                                     crop_idx * PT_NUM_KEYPOINTS * 3;
                    for (int k = 0; k < PT_NUM_KEYPOINTS; k++) {
                        det->keypoints[k][0] = kp_base[k * 3 + 0]; /* x */
                        det->keypoints[k][1] = kp_base[k * 3 + 1]; /* y */
                        det->keypoints[k][2] = kp_base[k * 3 + 2]; /* confidence */
                    }

                    /* Compute center-of-mass from hip keypoints (indices 11, 12) */
                    float lhip_x = det->keypoints[PT_HIP_LEFT_INDEX][0];
                    float lhip_y = det->keypoints[PT_HIP_LEFT_INDEX][1];
                    float lhip_c = det->keypoints[PT_HIP_LEFT_INDEX][2];
                    float rhip_x = det->keypoints[PT_HIP_RIGHT_INDEX][0];
                    float rhip_y = det->keypoints[PT_HIP_RIGHT_INDEX][1];
                    float rhip_c = det->keypoints[PT_HIP_RIGHT_INDEX][2];

                    if (lhip_c > p->config.keypoint_confidence &&
                        rhip_c > p->config.keypoint_confidence) {
                        det->com_2d[0] = (lhip_x + rhip_x) * 0.5f;
                        det->com_2d[1] = (lhip_y + rhip_y) * 0.5f;
                    } else if (lhip_c > p->config.keypoint_confidence) {
                        det->com_2d[0] = lhip_x;
                        det->com_2d[1] = lhip_y;
                    } else if (rhip_c > p->config.keypoint_confidence) {
                        det->com_2d[0] = rhip_x;
                        det->com_2d[1] = rhip_y;
                    } else {
                        /* Fall back to bounding box center */
                        det->com_2d[0] = (det->bbox[0] + det->bbox[2]) * 0.5f;
                        det->com_2d[1] = (det->bbox[1] + det->bbox[3]) * 0.5f;
                    }
                }
            }

            t_matching_total += pt_time_seconds() - t_match_start;
            t_match_start = pt_time_seconds();

            /* Cross-view matching: group detections across cameras */
            int num_groups = pt_match_cross_view(
                p->detections, det_counts_per_cam,
                &p->constants,
                p->config.epipolar_threshold,
                p->groups, PT_MAX_GROUPS
            );

            t_matching_total += pt_time_seconds() - t_match_start;
            double t_tri_start = pt_time_seconds();

            /* Generate 3D candidates from groups (all view subsets) */
            int num_candidate_groups = pt_generate_candidates(
                p->groups, num_groups,
                p->detections,
                &p->constants,
                p->config.keypoint_confidence,
                p->candidate_groups, PT_MAX_GROUPS
            );

            /* Update tracks with new candidate groups */
            int actual_sync_index = p->sync_table.sync_indices[sync_slot];

            pt_track_frame(
                &p->tracks,
                p->candidate_groups, num_candidate_groups,
                actual_sync_index,
                p->config.max_track_distance,
                p->config.max_persons,
                p->config.track_patience
            );

            t_triangulation_total += pt_time_seconds() - t_tri_start;
            frames_processed++;
        }

        /* Report progress at end of each batch */
        float progress = (float)(batch_end) / (float)(process_count);
        pipeline_progress(p, "processing", progress);
    }

    free(process_indices);

    /* --- Step 7: Export results --- */

    double t_export_start = pt_time_seconds();

    pipeline_log(p, "Exporting results to %s ...", p->config.output_dir);
    {
        /* Build output base path: output_dir/output_3d_poses_tracked.csv */
        char output_base[1024];
        snprintf(output_base, sizeof(output_base), "%s/output_3d_poses_tracked.csv",
                 p->config.output_dir);

        rc = pt_export_csv(&p->tracks, output_base);
        if (rc != PT_OK) {
            fprintf(stderr, "[pt_pipeline] CSV export failed (error %d)\n", rc);
            /* Non-fatal: don't return error, just log it */
        } else {
            pipeline_log(p, "Export complete.");
        }
    }

    double t_export_end = pt_time_seconds();

    /* --- Step 8: Compute final stats --- */

    double t_end = pt_time_seconds();

    p->stats.total_seconds = t_end - t_start;
    p->stats.decode_seconds = t_decode_total;
    p->stats.yolo_seconds = t_yolo_total;
    p->stats.vitpose_seconds = t_vitpose_total;
    p->stats.matching_seconds = t_matching_total;
    p->stats.triangulation_seconds = t_triangulation_total;
    p->stats.export_seconds = t_export_end - t_export_start;
    p->stats.frames_processed = frames_processed;
    p->stats.persons_tracked = pt_track_count_active(&p->tracks);

    pipeline_log(p, "Pipeline complete: %d frames, %d persons, %.2fs total",
                 frames_processed, p->stats.persons_tracked, p->stats.total_seconds);
    pipeline_log(p, "  decode: %.2fs, yolo: %.2fs, vitpose: %.2fs",
                 p->stats.decode_seconds, p->stats.yolo_seconds, p->stats.vitpose_seconds);
    pipeline_log(p, "  matching: %.2fs, triangulation: %.2fs, export: %.2fs",
                 p->stats.matching_seconds, p->stats.triangulation_seconds, p->stats.export_seconds);

    pipeline_progress(p, "complete", 1.0f);

    p->is_initialized = 1;
    return PT_OK;
}
