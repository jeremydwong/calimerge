/*
 * pt_calibration.c - Calibration TOML loader.
 *
 * Ported from pt_pipeline.cpp's load_calibration_toml().
 * Pure C — no GPU dependencies.
 *
 * TODO: This should be moved to pt_shared/ so both CUDA and MPS pipelines
 * share the same implementation. For now, it's a standalone copy.
 */

#include "pt_calibration.h"
#include "../pt_shared/pt_matching.h"   /* pt_compute_projection_matrices, pt_precompute_fundamentals */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * TOML parser helpers (from pt_pipeline.cpp)
 * ============================================================================ */

static const char *skip_ws(const char *s) {
    while (*s == ' ' || *s == '\t') s++;
    return s;
}

static int parse_double_array(const char *s, double *out, int max_count) {
    int count = 0;
    const char *p = s;
    while (*p && *p != '[') p++;
    if (!*p) return 0;
    p++;

    while (*p && count < max_count) {
        while (*p == ' ' || *p == '\t' || *p == ',' || *p == '\n' || *p == '\r') p++;
        if (*p == ']') {
            p++;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == ']') break;
            continue;
        }
        if (*p == '[') { p++; continue; }
        if (*p == '\0') break;

        char *end;
        double val = strtod(p, &end);
        if (end > p) {
            out[count++] = val;
            p = end;
        } else {
            p++;
        }
    }
    return count;
}

static int parse_int_value(const char *s) {
    const char *p = strchr(s, '=');
    if (!p) return 0;
    return atoi(p + 1);
}

static void rodrigues_to_matrix(const double r[3], double R[3][3]) {
    double theta = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    if (theta < 1e-12) {
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
    if (close >= open && open > 0) return 1;
    *in_array = 1;
    return 0;
}

/* ============================================================================
 * Main TOML loader
 * ============================================================================ */

int pt_load_calibration(PT_CameraConstants *constants, const char *toml_path) {
    if (!constants || !toml_path) return PT_ERR_INVALID_ARGS;

    memset(constants, 0, sizeof(PT_CameraConstants));

    FILE *f = fopen(toml_path, "r");
    if (!f) return PT_ERR_FILE_NOT_FOUND;

    char line[2048];
    int current_cam = -1;
    int num_cameras_found = 0;

    char array_buf[4096];
    int in_array = 0;
    int array_type = 0;

    int rotation_is_rodrigues[PT_MAX_CAMERAS];
    memset(rotation_is_rodrigues, 0, sizeof(rotation_is_rodrigues));

    while (fgets(line, sizeof(line), f)) {
        if (in_array) {
            size_t cur_len = strlen(array_buf);
            size_t line_len = strlen(line);
            if (cur_len + line_len < sizeof(array_buf) - 1) {
                memcpy(array_buf + cur_len, line, line_len + 1);
            }
            int open = 0, close = 0;
            for (const char *c = array_buf; *c; c++) {
                if (*c == '[') open++;
                if (*c == ']') close++;
            }
            if (close >= open && open > 0) {
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

        if (*trimmed == '\0' || *trimmed == '\n' || *trimmed == '\r' || *trimmed == '#')
            continue;

        if (*trimmed == '[') {
            int cam_idx = -1;
            if (sscanf(trimmed, "[camera_%d]", &cam_idx) == 1 ||
                sscanf(trimmed, "[cam_%d]", &cam_idx) == 1) {
                if (cam_idx >= 0 && cam_idx < PT_MAX_CAMERAS) {
                    current_cam = cam_idx;
                    constants->ports[current_cam] = cam_idx;
                    if (cam_idx >= num_cameras_found)
                        num_cameras_found = cam_idx + 1;
                }
            }
            continue;
        }

        if (current_cam < 0) continue;

        if (strncmp(trimmed, "port", 4) == 0 && (trimmed[4] == ' ' || trimmed[4] == '=')) {
            constants->ports[current_cam] = parse_int_value(trimmed);
        } else if (strncmp(trimmed, "size", 4) == 0 && (trimmed[4] == ' ' || trimmed[4] == '=')) {
            double vals[2];
            const char *eq = strchr(trimmed, '=');
            if (eq && parse_double_array(eq, vals, 2) >= 2) {
                constants->cam_width[current_cam] = (int)vals[0];
                constants->cam_height[current_cam] = (int)vals[1];
                constants->frame_width = (int)vals[0];
                constants->frame_height = (int)vals[1];
            }
        } else if (strncmp(trimmed, "camera_matrix", 13) == 0 ||
                   (strncmp(trimmed, "matrix", 6) == 0 && trimmed[6] != '_' &&
                    (trimmed[6] == ' ' || trimmed[6] == '='))) {
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
        fprintf(stderr, "[pt_calibration] No cameras found in TOML: %s\n", toml_path);
        return PT_ERR_INVALID_ARGS;
    }

    constants->num_cameras = num_cameras_found;

    for (int i = 0; i < num_cameras_found; i++) {
        if (rotation_is_rodrigues[i]) {
            fprintf(stderr, "[pt_calibration] Camera %d: converted Rodrigues to 3x3 matrix\n", i);
        }
    }

    /* Compute derived matrices */
    pt_compute_projection_matrices(constants);
    pt_precompute_fundamentals(constants);

    return PT_OK;
}

void pt_compute_derived_matrices(PT_CameraConstants *constants) {
    if (!constants) return;
    pt_compute_projection_matrices(constants);
    pt_precompute_fundamentals(constants);
}
