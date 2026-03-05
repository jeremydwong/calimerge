/*
 * pt_tracker.cpp - Multi-person track management and candidate-to-track assignment.
 *
 * Translates the Python algorithms from tracker.py:
 *   - PersonTrack class (state management, COM computation, ring buffer history)
 *   - generate_3d_candidates_from_groups()
 *   - assign_3d_candidates_to_tracks() (Hungarian on COM distance)
 *
 * The Python version has elaborate logic for choosing view subsets and scoring
 * new track candidates. We simplify:
 *   - Triangulate using all views in a group (more views = better SVD)
 *   - New tracks: pick unmatched candidates closest to existing tracks
 *     but above min_new_track_distance
 */

#include "pt_tracker.h"
#include "pt_triangulation.h"
#include "pt_matching.h"  /* For pt_hungarian */
#include <math.h>
#include <string.h>

/* ============================================================================
 * Internal helpers
 * ============================================================================ */

/* Euclidean distance between two 3D points */
static double dist3d(const double a[3], const double b[3])
{
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double dz = a[2] - b[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

/* Write one frame of data into a track's ring buffer */
static void track_write_history(
    PT_PersonTrack *track,
    const PT_Candidate3D *candidate,
    int sync_index)
{
    int wi = track->history_write_idx;

    /* Copy keypoints */
    memcpy(track->keypoints_3d[wi], candidate->xyz,
           PT_NUM_KEYPOINTS * 3 * sizeof(double));
    memcpy(track->keypoints_valid[wi], candidate->valid,
           PT_NUM_KEYPOINTS * sizeof(int));
    track->sync_indices[wi] = sync_index;

    /* Advance ring buffer */
    track->history_write_idx = (wi + 1) % PT_TRACK_HISTORY_SIZE;
    if (track->history_count < PT_TRACK_HISTORY_SIZE) {
        track->history_count++;
    }
}

/* Update a track with a new candidate match */
static void track_update(
    PT_PersonTrack *track,
    const PT_Candidate3D *candidate,
    int sync_index)
{
    /* Write to ring buffer */
    track_write_history(track, candidate, sync_index);

    /* Update last known state */
    track->last_com_3d[0] = candidate->com_3d[0];
    track->last_com_3d[1] = candidate->com_3d[1];
    track->last_com_3d[2] = candidate->com_3d[2];

    memcpy(track->last_views, candidate->views_used,
           PT_MAX_CAMERAS * sizeof(int));
    track->num_last_views = candidate->num_views;
    track->last_sync_index = sync_index;

    /* Reset lost counter */
    track->frames_since_seen = 0;
}

/* Create a new track from a candidate */
static void track_create(
    PT_TrackState *state,
    const PT_Candidate3D *candidate,
    int sync_index,
    int patience)
{
    if (state->num_tracks >= PT_MAX_TRACKS) return;

    PT_PersonTrack *track = &state->tracks[state->num_tracks];
    memset(track, 0, sizeof(PT_PersonTrack));

    track->person_id = state->next_person_id++;
    track->is_active = 1;
    track->frames_since_seen = 0;
    track->patience = patience;

    track->last_com_3d[0] = candidate->com_3d[0];
    track->last_com_3d[1] = candidate->com_3d[1];
    track->last_com_3d[2] = candidate->com_3d[2];

    memcpy(track->last_views, candidate->views_used,
           PT_MAX_CAMERAS * sizeof(int));
    track->num_last_views = candidate->num_views;
    track->last_sync_index = sync_index;

    track->history_count = 0;
    track->history_write_idx = 0;

    /* Write first frame to history */
    track_write_history(track, candidate, sync_index);

    state->num_tracks++;
}

/* ============================================================================
 * Track initialization
 * ============================================================================ */

void pt_track_init(PT_TrackState *state)
{
    memset(state, 0, sizeof(PT_TrackState));
}

/* ============================================================================
 * Candidate generation
 * ============================================================================ */

int pt_generate_candidates(
    const PT_Group *groups, int num_groups,
    const PT_Detection2D detections[PT_MAX_CAMERAS][PT_MAX_DETECTIONS],
    const PT_CameraConstants *constants,
    float keypoint_confidence,
    PT_Candidate3D *out_candidates,
    int max_candidates)
{
    int gi, mi;
    int n_candidates = 0;

    for (gi = 0; gi < num_groups; gi++) {
        if (n_candidates >= max_candidates) break;

        const PT_Group *group = &groups[gi];
        if (group->num_members < 2) continue;

        /* Collect detection pointers and camera indices for this group */
        const PT_Detection2D *det_ptrs[PT_MAX_CAMERAS];
        int cam_indices[PT_MAX_CAMERAS];
        int n_views = 0;

        for (mi = 0; mi < group->num_members; mi++) {
            int pi = group->members[mi].port_index;
            int di = group->members[mi].detection_index;

            if (!detections[pi][di].valid) continue;

            det_ptrs[n_views] = &detections[pi][di];
            cam_indices[n_views] = pi;
            n_views++;
        }

        if (n_views < 2) continue;

        /* Triangulate all keypoints for this person */
        PT_Candidate3D *cand = &out_candidates[n_candidates];
        int ok = pt_triangulate_person(
            det_ptrs,
            cam_indices,
            n_views,
            constants,
            keypoint_confidence,
            cand
        );

        if (ok && cand->com_valid) {
            n_candidates++;
        }
    }

    return n_candidates;
}

/* ============================================================================
 * Per-frame track update
 * ============================================================================ */

void pt_track_frame(
    PT_TrackState *state,
    const PT_Candidate3D *candidates, int num_candidates,
    int sync_index,
    float max_distance,
    int max_persons,
    int patience)
{
    int i, j;

    /*
     * Special case: no existing tracks.
     * Create new tracks from candidates (up to max_persons).
     */
    if (state->num_tracks == 0 || pt_track_count_active(state) == 0) {
        int n_to_create = num_candidates;
        if (n_to_create > max_persons) n_to_create = max_persons;

        for (i = 0; i < n_to_create; i++) {
            if (candidates[i].com_valid) {
                track_create(state, &candidates[i], sync_index, patience);
            }
        }
        return;
    }

    /*
     * Build cost matrix: active_tracks x candidates.
     * cost[t][c] = ||track[t].com - candidate[c].com|| if < max_distance, else 1000.0
     */

    /* Collect active track indices */
    int active_indices[PT_MAX_TRACKS];
    int n_active = 0;
    for (i = 0; i < state->num_tracks; i++) {
        if (state->tracks[i].is_active) {
            active_indices[n_active++] = i;
        }
    }

    if (n_active == 0 && num_candidates == 0) return;

    /* If no active tracks, just create new ones */
    if (n_active == 0) {
        int n_to_create = num_candidates;
        if (n_to_create > max_persons) n_to_create = max_persons;
        for (i = 0; i < n_to_create; i++) {
            if (candidates[i].com_valid) {
                track_create(state, &candidates[i], sync_index, patience);
            }
        }
        return;
    }

    /* Build cost matrix */
    double cost[PT_MAX_TRACKS * PT_MAX_GROUPS]; /* flat: n_active x num_candidates */
    int n_rows = n_active;
    int n_cols = num_candidates;

    for (i = 0; i < n_rows; i++) {
        PT_PersonTrack *track = &state->tracks[active_indices[i]];
        for (j = 0; j < n_cols; j++) {
            if (!candidates[j].com_valid) {
                cost[i * n_cols + j] = 1000.0;
                continue;
            }

            double d = dist3d(track->last_com_3d, candidates[j].com_3d);
            if (d < (double)max_distance) {
                cost[i * n_cols + j] = d;
            } else {
                cost[i * n_cols + j] = 1000.0;
            }
        }
    }

    /* Run Hungarian algorithm */
    int row_assign[PT_MAX_TRACKS];
    int col_assign[PT_MAX_GROUPS];
    memset(row_assign, -1, sizeof(int) * (size_t)n_rows);
    memset(col_assign, -1, sizeof(int) * (size_t)n_cols);

    if (n_rows > 0 && n_cols > 0) {
        pt_hungarian(cost, n_rows, n_cols, row_assign, col_assign);
    }

    /* Process matches */
    int track_matched[PT_MAX_TRACKS];
    int cand_matched[PT_MAX_GROUPS];
    memset(track_matched, 0, sizeof(track_matched));
    memset(cand_matched, 0, sizeof(cand_matched));

    for (i = 0; i < n_rows; i++) {
        j = row_assign[i];
        if (j >= 0 && j < n_cols && cost[i * n_cols + j] < (double)max_distance) {
            /* Valid match: update track */
            int ti = active_indices[i];
            track_update(&state->tracks[ti], &candidates[j], sync_index);
            track_matched[i] = 1;
            cand_matched[j] = 1;
        }
    }

    /* Increment lost counter for unmatched active tracks */
    for (i = 0; i < n_rows; i++) {
        if (!track_matched[i]) {
            int ti = active_indices[i];
            state->tracks[ti].frames_since_seen++;
        }
    }

    /* Create new tracks from unmatched candidates */
    int current_active = pt_track_count_active(state);
    if (current_active < max_persons) {
        for (j = 0; j < n_cols; j++) {
            if (cand_matched[j]) continue;
            if (!candidates[j].com_valid) continue;
            if (current_active >= max_persons) break;

            /*
             * Check that this candidate is not too close to any existing track.
             * The Python code uses min_new_track_distance = 0.3m.
             * We use max_distance as the minimum separation threshold --
             * if a candidate is within max_distance of any active track, it
             * would have been assigned. Being here means it is far from all
             * tracks, so it is safe to create.
             */
            int too_close = 0;
            int t;
            for (t = 0; t < state->num_tracks; t++) {
                if (!state->tracks[t].is_active) continue;
                double d = dist3d(state->tracks[t].last_com_3d,
                                  candidates[j].com_3d);
                if (d < (double)max_distance) {
                    too_close = 1;
                    break;
                }
            }

            if (!too_close) {
                track_create(state, &candidates[j], sync_index, patience);
                current_active++;
            }
        }
    }

    /* Deactivate tracks that exceeded patience */
    for (i = 0; i < state->num_tracks; i++) {
        if (state->tracks[i].is_active &&
            state->tracks[i].frames_since_seen > patience) {
            state->tracks[i].is_active = 0;
        }
    }
}

/* ============================================================================
 * History retrieval
 * ============================================================================ */

int pt_track_get_history(
    const PT_TrackState *state,
    int person_id,
    int *out_sync_indices,
    double (*out_keypoints_3d)[PT_NUM_KEYPOINTS][3],
    int (*out_keypoints_valid)[PT_NUM_KEYPOINTS],
    int max_frames)
{
    int ti = pt_track_find_by_id(state, person_id);
    if (ti < 0) return 0;

    const PT_PersonTrack *track = &state->tracks[ti];
    int count = track->history_count;
    if (count > max_frames) count = max_frames;

    /*
     * The ring buffer write index points to the next slot to write.
     * The oldest entry is at (write_idx - history_count + RING_SIZE) % RING_SIZE.
     * We read from oldest to newest.
     */
    int start;
    if (track->history_count >= PT_TRACK_HISTORY_SIZE) {
        /* Buffer is full: oldest is at write_idx */
        start = track->history_write_idx;
    } else {
        /* Buffer is not full: oldest is at 0 */
        start = 0;
    }

    /* We want the most recent 'count' frames.
     * Most recent is at (write_idx - 1 + RING_SIZE) % RING_SIZE.
     * Start of the last 'count' frames: (write_idx - count + RING_SIZE) % RING_SIZE. */
    int read_start = (track->history_write_idx - count + PT_TRACK_HISTORY_SIZE)
                     % PT_TRACK_HISTORY_SIZE;

    int i;
    for (i = 0; i < count; i++) {
        int ri = (read_start + i) % PT_TRACK_HISTORY_SIZE;

        if (out_sync_indices) {
            out_sync_indices[i] = track->sync_indices[ri];
        }
        if (out_keypoints_3d) {
            memcpy(out_keypoints_3d[i], track->keypoints_3d[ri],
                   PT_NUM_KEYPOINTS * 3 * sizeof(double));
        }
        if (out_keypoints_valid) {
            memcpy(out_keypoints_valid[i], track->keypoints_valid[ri],
                   PT_NUM_KEYPOINTS * sizeof(int));
        }
    }

    return count;
}

/* ============================================================================
 * Utility
 * ============================================================================ */

int pt_track_count_active(const PT_TrackState *state)
{
    int i, count = 0;
    for (i = 0; i < state->num_tracks; i++) {
        if (state->tracks[i].is_active) {
            count++;
        }
    }
    return count;
}

int pt_track_find_by_id(const PT_TrackState *state, int person_id)
{
    int i;
    for (i = 0; i < state->num_tracks; i++) {
        if (state->tracks[i].person_id == person_id) {
            return i;
        }
    }
    return -1;
}
