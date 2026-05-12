/*
 * pt_tracker.h - Multi-person track management and candidate-to-track assignment.
 *
 * Translates the Python algorithms from tracker.py:
 *   - PersonTrack class -> PT_PersonTrack / PT_TrackState structs
 *   - generate_3d_candidates_from_groups() -> pt_generate_candidates()
 *   - assign_3d_candidates_to_tracks() -> pt_track_frame()
 *
 * Tracks persist across frames using 3D center-of-mass matching.
 * Lost tracks are kept for `patience` frames before deactivation.
 *
 * Style: Plain C structs + free functions. No classes, no templates, no STL.
 */

#ifndef PT_TRACKER_H
#define PT_TRACKER_H

#include "pt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Track initialization
 * ============================================================================ */

/*
 * Initialize track state. Zeroes all tracks, sets num_tracks = 0,
 * next_person_id = 0.
 */
void pt_track_init(PT_TrackState *state);

/* ============================================================================
 * Candidate generation
 * ============================================================================ */

/*
 * Generate 3D candidates from cross-view groups.
 *
 * For each group, generates ALL view subsets (combinations of 2..N views)
 * and triangulates each subset independently.  This matches the Python
 * generate_3d_candidates_from_groups() which uses itertools.combinations.
 *
 * Each group produces a PT_CandidateGroup with up to PT_MAX_VIEW_SUBSETS
 * candidates (one per view combination).  The tracker then selects the
 * candidate whose view set matches the track's last views.
 *
 * Returns number of groups that produced at least one valid candidate.
 */
int pt_generate_candidates(
    const PT_Group *groups, int num_groups,
    const PT_Detection2D detections[PT_MAX_CAMERAS][PT_MAX_DETECTIONS],
    const PT_CameraConstants *constants,
    float keypoint_confidence,
    PT_CandidateGroup *out_groups,
    int max_groups
);

/* ============================================================================
 * Per-frame track update
 * ============================================================================ */

/*
 * Assign 3D candidates to existing tracks and update track state.
 *
 * Faithfully ports Python assign_3d_candidates_to_tracks():
 *   1. For each (track, group) pair, find the candidate whose view set
 *      exactly matches the track's last views.  Hard constraint: no match
 *      if views differ.
 *   2. Build cost matrix from COM distances of best-matching candidates.
 *   3. Run Hungarian algorithm on the cost matrix.
 *   4. Accept matches where cost < max_distance.
 *   5. Update matched tracks, increment lost counter for unmatched.
 *   6. Create new tracks from unmatched groups (min_new_track_distance=0.3m).
 *   7. Deactivate tracks where frames_since_seen > patience.
 */
void pt_track_frame(
    PT_TrackState *state,
    const PT_CandidateGroup *groups, int num_groups,
    int sync_index,
    float max_distance,
    int max_persons,
    int patience
);

/* ============================================================================
 * History retrieval
 * ============================================================================ */

/*
 * Get all results for a specific person_id from the track history.
 *
 * Walks the ring buffer and copies sync indices, 3D keypoints, and validity
 * masks into the output arrays.
 *
 * Returns number of frames copied (up to max_frames).
 * Returns 0 if the person_id is not found.
 */
int pt_track_get_history(
    const PT_TrackState *state,
    int person_id,
    int *out_sync_indices,
    double (*out_keypoints_3d)[PT_NUM_KEYPOINTS][3],
    int (*out_keypoints_valid)[PT_NUM_KEYPOINTS],
    int max_frames
);

/* ============================================================================
 * Utility
 * ============================================================================ */

/*
 * Count the number of currently active tracks.
 */
int pt_track_count_active(const PT_TrackState *state);

/* ============================================================================
 * Post-processing: stitch fragmented tracks
 * ============================================================================ */

/*
 * pt_track_stitch - Merge tracks whose hip-COM trajectories are spatially
 * close and temporally adjacent.
 *
 * The per-frame tracker spawns a fresh track id whenever the camera subset
 * feeding triangulation changes for a single frame (e.g. one camera drops
 * a detection then comes back). For a single-subject trial this fragments
 * one person into many short tracks.
 *
 * This function re-merges tracks using the same greedy algorithm as the
 * Python track_stitch.py it replaces:
 *
 *   1. Two tracks may merge only if their sync-index ranges are disjoint
 *      (no overlap). Two people on the same syncs must not be stitched.
 *   2. The temporal gap (newer.first_sync - older.last_sync) must be
 *      <= max_gap_frames.
 *   3. The 3D hip COM distance at the seam must be <= max_distance_m.
 *   4. Among eligible pairs, prefer smallest gap, then smallest distance.
 *   5. Repeat until no more merges are possible.
 *
 * After stitching, consumed tracks have history_count == 0. Surviving
 * tracks contain the merged ring-buffer entries. pt_export_csv sorts by
 * sync_index, so insertion order in the ring buffer does not matter.
 *
 * Returns the number of merges performed.
 */
int pt_track_stitch(PT_TrackState *state, int max_gap_frames, float max_distance_m);

/*
 * Find track index by person_id. Returns -1 if not found.
 */
int pt_track_find_by_id(const PT_TrackState *state, int person_id);

#ifdef __cplusplus
}
#endif

#endif /* PT_TRACKER_H */
