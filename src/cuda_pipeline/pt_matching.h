/*
 * pt_matching.h - Cross-view epipolar matching, Hungarian algorithm, union-find.
 *
 * Matches person detections across camera views using epipolar geometry and
 * bipartite matching. Groups matched detections via union-find.
 *
 * All precomputation (projection matrices, fundamental matrices) is done once
 * at startup. The Python code recalculates fundamentals every frame -- this
 * eliminates that waste.
 *
 * Style: Plain C structs + free functions. No classes, no templates, no STL.
 */

#ifndef PT_MATCHING_H
#define PT_MATCHING_H

#include "pt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Union-Find (disjoint set)
 * ============================================================================ */

typedef struct {
    int parent[256];
    int rank[256];
    int count;
} PT_UnionFind;

void pt_uf_init(PT_UnionFind *uf, int n);
int  pt_uf_find(PT_UnionFind *uf, int x);
void pt_uf_union(PT_UnionFind *uf, int x, int y);

/* ============================================================================
 * Precomputation (called once at startup)
 * ============================================================================ */

/*
 * Compute projection matrix P = K * [R | t] for each camera.
 * Reads camera_matrix, rotation, translation from constants.
 * Writes projection[i] for i in [0, num_cameras).
 */
void pt_compute_projection_matrices(PT_CameraConstants *c);

/*
 * Compute fundamental matrix F for every camera pair.
 *
 * F[i][j] such that x2^T * F * x1 = 0.
 *
 * Algorithm (from triangulation.py:calculate_fundamental_matrix):
 *   1. C1 = null space of P1 (last row of V^T from SVD of P1)
 *   2. Normalize: C1 = C1 / C1[3]
 *   3. e2 = P2 * C1, normalize: e2 = e2 / e2[2]
 *   4. F = [e2]_x * P2 * pinv(P1)
 *   5. Enforce rank-2 via SVD: F = U * diag(s1, s2, 0) * V^T
 *   6. Normalize: F = F / ||F||
 *
 * This is called once. The Python code recalculates every frame.
 *
 * Writes fundamental[i * PT_MAX_CAMERAS + j] for all pairs.
 * Sets fundamental_valid[idx] = 1 for computed pairs.
 */
void pt_precompute_fundamentals(PT_CameraConstants *c);

/* ============================================================================
 * Epipolar distance
 * ============================================================================ */

/*
 * Distance from point p2 to the epipolar line induced by p1 in camera 2.
 *
 * l = F * [p1_x, p1_y, 1]^T
 * distance = |l . [p2_x, p2_y, 1]^T| / sqrt(l[0]^2 + l[1]^2)
 *
 * Returns HUGE_VAL if denominator is near zero.
 */
double pt_epipolar_distance(
    const double F[3][3],
    double p1_x, double p1_y,
    double p2_x, double p2_y
);

/* ============================================================================
 * Hungarian algorithm
 * ============================================================================ */

/*
 * Solve the linear assignment problem for a cost matrix.
 * Finds minimum-cost one-to-one matching (Munkres/Hungarian).
 *
 * For our use case, matrices are at most PT_MAX_DETECTIONS x PT_MAX_DETECTIONS (16x16).
 *
 * cost_matrix: row-major, dimensions (n_rows x n_cols).
 * row_assign[i] = column assigned to row i, or -1 if unassigned.
 * col_assign[j] = row assigned to column j, or -1 if unassigned.
 */
void pt_hungarian(
    const double *cost_matrix,
    int n_rows, int n_cols,
    int *row_assign,
    int *col_assign
);

/* ============================================================================
 * Cross-view matching
 * ============================================================================ */

/*
 * Match person detections across camera views using epipolar geometry.
 *
 * Algorithm (from tracker.py:group_detections_across_views_bipartite):
 * 1. For each camera pair (i, j) where i < j:
 *    a. Build cost matrix: cost[a][b] = epipolar_distance(F[i][j], det_i[a].com_2d, det_j[b].com_2d)
 *       Fill unmatched entries with 1000.0 (effectively infinity for assignment).
 *    b. Run Hungarian algorithm on cost matrix.
 *    c. Accept matches where cost < epipolar_threshold.
 * 2. Merge pairwise matches using union-find to form groups.
 *    Each element is (port_index, detection_index) encoded as port_index * PT_MAX_DETECTIONS + detection_index.
 * 3. Filter groups: require >= 2 cameras per group.
 *
 * Returns number of groups found (written to out_groups).
 */
int pt_match_cross_view(
    const PT_Detection2D detections[PT_MAX_CAMERAS][PT_MAX_DETECTIONS],
    const int detection_counts[PT_MAX_CAMERAS],
    const PT_CameraConstants *constants,
    float epipolar_threshold,
    PT_Group *out_groups,
    int max_groups
);

#ifdef __cplusplus
}
#endif

#endif /* PT_MATCHING_H */
