/*
 * pt_export.cpp - CSV export for tracked 3D pose data.
 *
 * Writes one CSV file per tracked person.  The format matches the Python
 * output convention: output_3d_poses_tracked.csv_person0.csv
 *
 * Column order: sync_index, person_id, then for each of the 52 SynthPose markers:
 *   {MarkerName}_X, {MarkerName}_Y, {MarkerName}_Z
 * giving 2 + 52*3 = 158 columns total.
 *
 * The model outputs 17 COCO keypoints (PT_NUM_KEYPOINTS).
 * Columns 17-51 are padded with empty fields (NaN) to match the
 * Python SynthPoseMarkers format (PT_NUM_KEYPOINTS = 52).
 *
 * Invalid (non-triangulated) keypoints are written as empty fields (,,)
 * which Python/pandas reads as NaN.
 *
 * The ring buffer in PT_PersonTrack is walked in chronological order
 * (oldest to newest) to produce sorted output.
 *
 * PT_EXPORT_MARKER_NAMES is defined here (moved out of pt_common.h to
 * avoid NVCC compilation issues with large static string arrays).
 *
 * Style: Plain C structs + free functions. No classes, no templates, no STL.
 */

#include "pt_export.h"

static const char *PT_EXPORT_MARKER_NAMES[PT_NUM_KEYPOINTS] = {
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
    "sternum", "rshoulder", "lshoulder",
    "r_lelbow", "l_lelbow", "r_melbow", "l_melbow",
    "r_lwrist", "l_lwrist", "r_mwrist", "l_mwrist",
    "r_ASIS", "l_ASIS", "r_PSIS", "l_PSIS",
    "r_knee", "l_knee", "r_mknee", "l_mknee",
    "r_ankle", "l_ankle", "r_mankle", "l_mankle",
    "r_5meta", "l_5meta", "r_toe", "l_toe",
    "r_big_toe", "l_big_toe", "l_calc", "r_calc",
    "C7", "L2", "T11", "T6"
};

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <direct.h>
#define pt_mkdir(path) _mkdir(path)
#else
#include <sys/stat.h>
#define pt_mkdir(path) mkdir(path, 0755)
#endif

/* ============================================================================
 * Internal: Write CSV header line
 *
 * Format: sync_index,Nose_X,Nose_Y,Nose_Z,L_Eye_X,...,T6_Z
 * ============================================================================ */

static int write_csv_header(FILE *f) {
    if (fprintf(f, "sync_index,person_id") < 0) return -1;

    for (int k = 0; k < PT_NUM_KEYPOINTS; k++) {
        if (fprintf(f, ",%s_X,%s_Y,%s_Z",
                    PT_EXPORT_MARKER_NAMES[k],
                    PT_EXPORT_MARKER_NAMES[k],
                    PT_EXPORT_MARKER_NAMES[k]) < 0) {
            return -1;
        }
    }
    if (fprintf(f, "\n") < 0) return -1;

    return 0;
}

/* ============================================================================
 * Internal: Write one data row for a single frame
 *
 * sync_index, person_id, followed by 52 * 3 values.
 * First 17 keypoints come from model output; 17-51 are NaN padding.
 * Invalid keypoints produce empty fields (,,) which pandas reads as NaN.
 * Matches Python save_person_csv() in process_synced_poses.py:1424.
 * ============================================================================ */

static int write_csv_row(FILE *f, int sync_index, int person_id,
                          const double keypoints_3d[PT_NUM_KEYPOINTS][3],
                          const int keypoints_valid[PT_NUM_KEYPOINTS]) {
    /* Python uses %.4f format: process_synced_poses.py:1460 */
    if (fprintf(f, "%d,%d", sync_index, person_id) < 0) return -1;

    /* Write actual model keypoints (0 to PT_NUM_KEYPOINTS-1 = 0..16) */
    for (int k = 0; k < PT_NUM_KEYPOINTS; k++) {
        if (keypoints_valid[k]) {
            if (fprintf(f, ",%.4f,%.4f,%.4f",
                        keypoints_3d[k][0],
                        keypoints_3d[k][1],
                        keypoints_3d[k][2]) < 0) {
                return -1;
            }
        } else {
            if (fprintf(f, ",,,") < 0) return -1;
        }
    }

    /* Pad remaining columns (PT_NUM_KEYPOINTS to PT_NUM_KEYPOINTS-1 = 17..51) with NaN */
    for (int k = PT_NUM_KEYPOINTS; k < PT_NUM_KEYPOINTS; k++) {
        if (fprintf(f, ",,,") < 0) return -1;
    }

    if (fprintf(f, "\n") < 0) return -1;
    return 0;
}

/* ============================================================================
 * Internal: Sort helper for chronological ring buffer traversal
 *
 * The ring buffer may have wrapped around, so we need to produce frames
 * in chronological order (ascending sync_index).  We collect (ring_idx,
 * sync_index) pairs, sort by sync_index, then write in that order.
 * ============================================================================ */

typedef struct {
    int ring_idx;
    int sync_index;
} ExportEntry;

static int compare_export_entries(const void *a, const void *b) {
    const ExportEntry *ea = (const ExportEntry *)a;
    const ExportEntry *eb = (const ExportEntry *)b;
    return ea->sync_index - eb->sync_index;
}

/* ============================================================================
 * pt_export_csv
 * ============================================================================ */

extern "C" int pt_export_csv(const PT_TrackState *tracks, const char *output_base_path) {
    if (!tracks || !output_base_path) return PT_ERR_INVALID_PARAM;

    /* Ensure the parent directory exists */
    {
        char dir[1024];
        strncpy(dir, output_base_path, sizeof(dir) - 1);
        dir[sizeof(dir) - 1] = '\0';
        /* Find last slash or backslash */
        char *last_sep = NULL;
        for (char *p = dir; *p; p++) {
            if (*p == '/' || *p == '\\') last_sep = p;
        }
        if (last_sep) {
            *last_sep = '\0';
            pt_mkdir(dir);
        }
    }

    int persons_exported = 0;

    for (int t = 0; t < tracks->num_tracks; t++) {
        const PT_PersonTrack *track = &tracks->tracks[t];

        /* Skip tracks with no history */
        if (track->history_count <= 0) continue;

        /* Build output filename: {base}_person{id}.csv */
        char filename[1024];
        snprintf(filename, sizeof(filename), "%s_person%d.csv",
                 output_base_path, track->person_id);

        FILE *f = fopen(filename, "w");
        if (!f) {
            fprintf(stderr, "[pt_export] Cannot open output file: %s\n", filename);
            return PT_ERR_FILE_NOT_FOUND;
        }

        /* Write header */
        if (write_csv_header(f) < 0) {
            fprintf(stderr, "[pt_export] Write error on header: %s\n", filename);
            fclose(f);
            return PT_ERR_FILE_NOT_FOUND;
        }

        /* Collect ring buffer entries and sort chronologically.
         *
         * The ring buffer has capacity PT_TRACK_HISTORY_SIZE.
         * history_count = total frames ever written (may exceed capacity).
         * history_write_idx = next write position (wraps around).
         *
         * If history_count <= PT_TRACK_HISTORY_SIZE, all entries from
         * index 0 to history_count-1 are valid.
         *
         * If history_count > PT_TRACK_HISTORY_SIZE, the buffer has wrapped
         * and we have exactly PT_TRACK_HISTORY_SIZE valid entries starting
         * from history_write_idx (oldest) wrapping around. */

        int num_entries;
        if (track->history_count <= PT_TRACK_HISTORY_SIZE) {
            num_entries = track->history_count;
        } else {
            num_entries = PT_TRACK_HISTORY_SIZE;
        }

        /* Allocate sort buffer on the stack if small enough, else heap.
         * PT_TRACK_HISTORY_SIZE is 2048, so this is ~16KB -- fine for stack. */
        ExportEntry entries[PT_TRACK_HISTORY_SIZE];

        if (track->history_count <= PT_TRACK_HISTORY_SIZE) {
            /* No wrapping: entries are at indices 0..history_count-1 */
            for (int i = 0; i < num_entries; i++) {
                entries[i].ring_idx = i;
                entries[i].sync_index = track->sync_indices[i];
            }
        } else {
            /* Buffer has wrapped: oldest entry is at history_write_idx */
            int start = track->history_write_idx;
            for (int i = 0; i < num_entries; i++) {
                int ring_idx = (start + i) % PT_TRACK_HISTORY_SIZE;
                entries[i].ring_idx = ring_idx;
                entries[i].sync_index = track->sync_indices[ring_idx];
            }
        }

        /* Sort by sync_index ascending */
        qsort(entries, num_entries, sizeof(ExportEntry), compare_export_entries);

        /* Write data rows */
        for (int i = 0; i < num_entries; i++) {
            int ri = entries[i].ring_idx;
            if (write_csv_row(f, entries[i].sync_index,
                               track->person_id,
                               track->keypoints_3d[ri],
                               track->keypoints_valid[ri]) < 0) {
                fprintf(stderr, "[pt_export] Write error on row %d: %s\n", i, filename);
                fclose(f);
                return PT_ERR_FILE_NOT_FOUND;
            }
        }

        fclose(f);
        persons_exported++;

        fprintf(stderr, "[pt_export] Wrote %d frames for person %d -> %s\n",
                num_entries, track->person_id, filename);
    }

    fprintf(stderr, "[pt_export] Exported %d person(s) total.\n", persons_exported);
    return PT_OK;
}

/* ============================================================================
 * pt_export_get_person_count
 * ============================================================================ */

extern "C" int pt_export_get_person_count(const PT_TrackState *tracks) {
    if (!tracks) return 0;

    int count = 0;
    for (int t = 0; t < tracks->num_tracks; t++) {
        if (tracks->tracks[t].history_count > 0) {
            count++;
        }
    }
    return count;
}
