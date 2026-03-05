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
 * For each group, triangulates all keypoints using views from the group
 * and computes the 3D center of mass.
 *
 * Algorithm (from tracker.py:generate_3d_candidates_from_groups):
 *   For each group:
 *     1. Collect all detection pointers and camera indices from group members
 *     2. Call pt_triangulate_person() to triangulate all keypoints
 *     3. If COM is valid, output as a candidate
 *
 * The Python version tries all view subsets (combinations of 2..N views) and
 * picks the best. For the C version we use all views in the group directly --
 * more views is strictly better for SVD triangulation when the cameras are
 * calibrated.
 *
 * Returns number of candidates generated.
 */
int pt_generate_candidates(
    const PT_Group *groups, int num_groups,
    const PT_Detection2D detections[PT_MAX_CAMERAS][PT_MAX_DETECTIONS],
    const PT_CameraConstants *constants,
    float keypoint_confidence,
    PT_Candidate3D *out_candidates,
    int max_candidates
);

/* ============================================================================
 * Per-frame track update
 * ============================================================================ */

/*
 * Assign 3D candidates to existing tracks and update track state.
 *
 * Algorithm (from tracker.py:assign_3d_candidates_to_tracks):
 *   1. Build cost matrix: cost[track][candidate] = ||track.com - candidate.com||
 *      Set cost = 1000.0 if distance > max_distance
 *   2. Run Hungarian algorithm on cost matrix
 *   3. Accept matches where cost < max_distance
 *   4. Update matched tracks:
 *      - Write new keypoints to ring buffer
 *      - Update last_com_3d, last_views, last_sync_index
 *      - Reset frames_since_seen = 0
 *   5. Increment frames_since_seen for unmatched active tracks
 *   6. Create new tracks from unmatched candidates (if under max_persons)
 *   7. Deactivate tracks where frames_since_seen > patience
 */
void pt_track_frame(
    PT_TrackState *state,
    const PT_Candidate3D *candidates, int num_candidates,
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

/*
 * Find track index by person_id. Returns -1 if not found.
 */
int pt_track_find_by_id(const PT_TrackState *state, int person_id);

#ifdef __cplusplus
}
#endif

#endif /* PT_TRACKER_H */
