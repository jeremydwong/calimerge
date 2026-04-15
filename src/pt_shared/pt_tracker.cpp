/*
 * pt_tracker.cpp - Multi-person track management and candidate-to-track assignment.
 *
 * Faithful port of Python process_synced_poses.py:
 *   - generate_3d_candidates_from_groups() -- all view subsets via itertools.combinations
 *   - assign_3d_candidates_to_tracks()     -- Hungarian with exact view-set constraint
 *   - PersonTrack class                     -- ring buffer history, COM, view tracking
 */

#include "pt_tracker.h"
#include "pt_triangulation.h"
#include "pt_matching.h"  /* For pt_hungarian */
#include <math.h>
#include <string.h>
#include <stdio.h>

/* Python default: min_new_track_distance = 0.3m (process_synced_poses.py:322) */
#define PT_MIN_NEW_TRACK_DISTANCE  0.3

/* Python default: max_new_track_distance = 5.0m (process_synced_poses.py:322) */
#define PT_MAX_NEW_TRACK_DISTANCE  5.0

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

/* popcount for int (number of set bits) */
static int popcount(unsigned int x)
{
    int c = 0;
    while (x) { c += x & 1; x >>= 1; }
    return c;
}

/*
 * Check if two view sets are identical (order-independent).
 * Views are stored as sorted arrays of camera indices.
 */
static int views_match(const int *a, int na, const int *b, int nb)
{
    int i, j;
    if (na != nb) return 0;
    for (i = 0; i < na; i++) {
        int found = 0;
        for (j = 0; j < nb; j++) {
            if (a[i] == b[j]) { found = 1; break; }
        }
        if (!found) return 0;
    }
    return 1;
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

/* Create a new track from a candidate, reusing inactive slots first */
static void track_create(
    PT_TrackState *state,
    const PT_Candidate3D *candidate,
    int sync_index,
    int patience)
{
    PT_PersonTrack *track = NULL;

    /* Try to reuse an inactive slot */
    for (int i = 0; i < state->num_tracks; i++) {
        if (!state->tracks[i].is_active) {
            track = &state->tracks[i];
            break;
        }
    }

    /* No inactive slot — append if space remains */
    if (track == NULL) {
        if (state->num_tracks >= PT_MAX_TRACKS) return;
        track = &state->tracks[state->num_tracks];
        state->num_tracks++;
    }

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
}

/* ============================================================================
 * Track initialization
 * ============================================================================ */

void pt_track_init(PT_TrackState *state)
{
    memset(state, 0, sizeof(PT_TrackState));
}

/* ============================================================================
 * Candidate generation — all view subsets (port of itertools.combinations)
 * ============================================================================ */

int pt_generate_candidates(
    const PT_Group *groups, int num_groups,
    const PT_Detection2D detections[PT_MAX_CAMERAS][PT_MAX_DETECTIONS],
    const PT_CameraConstants *constants,
    float keypoint_confidence,
    PT_CandidateGroup *out_groups,
    int max_groups)
{
    int gi, mi;
    int n_out = 0;

    for (gi = 0; gi < num_groups && n_out < max_groups; gi++) {
        const PT_Group *group = &groups[gi];
        if (group->num_members < 2) continue;

        /* Collect valid detection pointers and camera indices for this group */
        const PT_Detection2D *all_dets[PT_MAX_CAMERAS];
        int all_cams[PT_MAX_CAMERAS];
        int n_views = 0;

        for (mi = 0; mi < group->num_members; mi++) {
            int pi = group->members[mi].port_index;
            int di = group->members[mi].detection_index;

            if (!detections[pi][di].valid) continue;

            all_dets[n_views] = &detections[pi][di];
            all_cams[n_views] = pi;
            n_views++;
        }

        if (n_views < 2) continue;

        /*
         * Enumerate all subsets of size 2..n_views using bitmasks.
         * Python: for num_views in range(2, len(active_ports)+1):
         *             for combo in itertools.combinations(active_ports, num_views):
         *
         * Bitmask approach: iterate mask from 3 to (1<<n_views)-1,
         * skip masks with popcount < 2.
         */
        PT_CandidateGroup *cg = &out_groups[n_out];
        cg->num_candidates = 0;

        unsigned int max_mask = (1u << n_views) - 1;
        unsigned int mask;

        for (mask = 3; mask <= max_mask; mask++) {
            if (popcount(mask) < 2) continue;
            if (cg->num_candidates >= PT_MAX_VIEW_SUBSETS) break;

            /* Extract subset views */
            const PT_Detection2D *sub_dets[PT_MAX_CAMERAS];
            int sub_cams[PT_MAX_CAMERAS];
            int n_sub = 0;
            int b;

            for (b = 0; b < n_views; b++) {
                if (mask & (1u << b)) {
                    sub_dets[n_sub] = all_dets[b];
                    sub_cams[n_sub] = all_cams[b];
                    n_sub++;
                }
            }

            /* Triangulate this view subset */
            PT_Candidate3D *cand = &cg->candidates[cg->num_candidates];
            int ok = pt_triangulate_person(
                sub_dets, sub_cams, n_sub,
                constants, keypoint_confidence, cand
            );

            if (ok && cand->com_valid) {
                cg->num_candidates++;
            }
        }

        if (cg->num_candidates > 0) {
            n_out++;
        }
    }

    return n_out;
}

/* ============================================================================
 * Per-frame track update — faithful port of assign_3d_candidates_to_tracks()
 * ============================================================================ */

void pt_track_frame(
    PT_TrackState *state,
    const PT_CandidateGroup *groups, int num_groups,
    int sync_index,
    float max_distance,
    int max_persons,
    int patience)
{
    int i, j, ci;

    /*
     * Special case: no existing tracks.
     * Python (line 360-379): pick candidates matching default_views (first group's first
     * candidate's views).  We pick the candidate with the most views from each group.
     */
    if (state->num_tracks == 0 || pt_track_count_active(state) == 0) {
        /* Determine default views from first group's first candidate */
        const int *default_views = NULL;
        int n_default = 0;

        for (i = 0; i < num_groups; i++) {
            if (groups[i].num_candidates > 0) {
                default_views = groups[i].candidates[0].views_used;
                n_default = groups[i].candidates[0].num_views;
                break;
            }
        }

        int n_created = 0;
        for (i = 0; i < num_groups && n_created < max_persons; i++) {
            const PT_CandidateGroup *cg = &groups[i];
            /* Find candidate matching default views */
            for (ci = 0; ci < cg->num_candidates; ci++) {
                const PT_Candidate3D *c = &cg->candidates[ci];
                if (c->com_valid &&
                    (default_views == NULL ||
                     views_match(c->views_used, c->num_views, default_views, n_default))) {
                    track_create(state, c, sync_index, patience);
                    n_created++;
                    break;
                }
            }
        }
        return;
    }

    /* Collect active track indices */
    int active_indices[PT_MAX_TRACKS];
    int n_active = 0;
    for (i = 0; i < state->num_tracks; i++) {
        if (state->tracks[i].is_active) {
            active_indices[n_active++] = i;
        }
    }

    if (n_active == 0 && num_groups == 0) return;

    /* If no active tracks, create new ones (same as above) */
    if (n_active == 0) {
        int n_created = 0;
        for (i = 0; i < num_groups && n_created < max_persons; i++) {
            const PT_CandidateGroup *cg = &groups[i];
            /* Pick the candidate with most views */
            int best_ci = -1, best_nv = 0;
            for (ci = 0; ci < cg->num_candidates; ci++) {
                if (cg->candidates[ci].com_valid && cg->candidates[ci].num_views > best_nv) {
                    best_nv = cg->candidates[ci].num_views;
                    best_ci = ci;
                }
            }
            if (best_ci >= 0) {
                track_create(state, &cg->candidates[best_ci], sync_index, patience);
                n_created++;
            }
        }
        return;
    }

    /*
     * Build cost matrix: n_active tracks x num_groups.
     *
     * Python (line 388-420): For each (track, group), find the candidate in the
     * group whose view set EXACTLY matches the track's last views.  Among matching
     * candidates, pick the one with smallest COM distance.  Hard constraint: if no
     * candidate matches the track's views, cost = HI.
     */
    int n_rows = n_active;
    int n_cols = num_groups;
    double cost[PT_MAX_TRACKS * PT_MAX_GROUPS]; /* flat: n_rows x n_cols */

    /* best_cand_idx[i * n_cols + j] = which candidate in group j matches track i */
    int best_cand_idx[PT_MAX_TRACKS * PT_MAX_GROUPS];

    for (i = 0; i < n_rows; i++) {
        PT_PersonTrack *track = &state->tracks[active_indices[i]];
        for (j = 0; j < n_cols; j++) {
            const PT_CandidateGroup *cg = &groups[j];

            double best_dist = 1000.0;
            int best_ci_local = -1;

            for (ci = 0; ci < cg->num_candidates; ci++) {
                const PT_Candidate3D *c = &cg->candidates[ci];
                if (!c->com_valid) continue;

                /* Hard constraint: exact view set match (Python line 405-406) */
                if (!views_match(c->views_used, c->num_views,
                                 track->last_views, track->num_last_views)) {
                    continue;
                }

                double d = dist3d(track->last_com_3d, c->com_3d);
                if (d < (double)max_distance && d < best_dist) {
                    best_dist = d;
                    best_ci_local = ci;
                }
            }

            cost[i * n_cols + j] = (best_ci_local >= 0) ? best_dist : 1000.0;
            best_cand_idx[i * n_cols + j] = best_ci_local;
        }
    }

    /* Run Hungarian algorithm */
    int row_assign[PT_MAX_TRACKS];
    int col_assign[PT_MAX_GROUPS];
    memset(row_assign, -1, sizeof(int) * (size_t)n_rows);
    memset(col_assign, -1, sizeof(int) * (size_t)n_cols);

    /* Check if all costs are HI (no valid matches at all) */
    int any_valid = 0;
    for (i = 0; i < n_rows * n_cols; i++) {
        if (cost[i] < 999.0) { any_valid = 1; break; }
    }

    if (any_valid && n_rows > 0 && n_cols > 0) {
        pt_hungarian(cost, n_rows, n_cols, row_assign, col_assign);
    }

    /* Process matches */
    int track_matched[PT_MAX_TRACKS];
    int group_matched[PT_MAX_GROUPS];
    memset(track_matched, 0, sizeof(track_matched));
    memset(group_matched, 0, sizeof(group_matched));

    for (i = 0; i < n_rows; i++) {
        j = row_assign[i];
        if (j >= 0 && j < n_cols && cost[i * n_cols + j] < (double)max_distance) {
            int bci = best_cand_idx[i * n_cols + j];
            if (bci >= 0) {
                int ti = active_indices[i];
                track_update(&state->tracks[ti], &groups[j].candidates[bci], sync_index);
                track_matched[i] = 1;
                group_matched[j] = 1;
            }
        }
    }

    /* Increment lost counter for unmatched active tracks */
    for (i = 0; i < n_rows; i++) {
        if (!track_matched[i]) {
            int ti = active_indices[i];
            state->tracks[ti].frames_since_seen++;
        }
    }

    /*
     * Create new tracks from unmatched groups.
     * Python (line 441-551): elaborate scoring with min_new_track_distance=0.3m
     * and max_new_track_distance=5.0m relative to a reference position.
     *
     * We port the essential constraints:
     *   - Must be >= min_new_track_distance from all active tracks and assigned candidates
     *   - Must be <= max_new_track_distance from a reference position (first track)
     *   - Among qualifying candidates, prefer the one with best view count
     */
    int current_active = pt_track_count_active(state);
    if (current_active < max_persons && num_groups > 0) {
        /* Reference position: first active track's COM (Python line 446-453) */
        double ref_pos[3] = {0, 0, 0};
        int have_ref = 0;
        for (i = 0; i < state->num_tracks; i++) {
            if (state->tracks[i].is_active) {
                ref_pos[0] = state->tracks[i].last_com_3d[0];
                ref_pos[1] = state->tracks[i].last_com_3d[1];
                ref_pos[2] = state->tracks[i].last_com_3d[2];
                have_ref = 1;
                break;
            }
        }

        if (have_ref) {
            for (j = 0; j < n_cols; j++) {
                if (group_matched[j]) continue;
                if (current_active >= max_persons) break;

                const PT_CandidateGroup *cg = &groups[j];
                int best_ci_new = -1;
                int best_nv_new = 0;

                for (ci = 0; ci < cg->num_candidates; ci++) {
                    const PT_Candidate3D *c = &cg->candidates[ci];
                    if (!c->com_valid) continue;

                    /* Check max distance from reference */
                    double d_ref = dist3d(ref_pos, c->com_3d);
                    if (d_ref > PT_MAX_NEW_TRACK_DISTANCE) continue;

                    /* Check min distance from ALL active tracks */
                    int too_close = 0;
                    int t;
                    for (t = 0; t < state->num_tracks; t++) {
                        if (!state->tracks[t].is_active) continue;
                        double d = dist3d(state->tracks[t].last_com_3d, c->com_3d);
                        if (d < PT_MIN_NEW_TRACK_DISTANCE) {
                            too_close = 1;
                            break;
                        }
                    }
                    if (too_close) continue;

                    /* Prefer more views */
                    if (c->num_views > best_nv_new) {
                        best_nv_new = c->num_views;
                        best_ci_new = ci;
                    }
                }

                if (best_ci_new >= 0) {
                    track_create(state, &cg->candidates[best_ci_new], sync_index, patience);
                    current_active++;
                }
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
