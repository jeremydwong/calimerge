/*
 * pt_triangulation.h - SVD triangulation and undistortion for multi-view 3D reconstruction.
 *
 * Translates the Python algorithms from triangulation.py:
 *   - triangulate_keypoints(): SVD multi-view triangulation
 *   - cv2.undistortPoints(): OpenCV iterative undistortion
 *
 * Key design decisions:
 *   - SVD via normal equations (A^T*A) + Jacobi eigendecomposition for 4x4
 *   - Iterative undistortion replaces per-keypoint OpenCV calls
 *   - All fixed-size: max 32 rows in A matrix (16 cameras * 2 rows each)
 *
 * Style: Plain C structs + free functions. No classes, no templates, no STL.
 */

#ifndef PT_TRIANGULATION_H
#define PT_TRIANGULATION_H

#include "pt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * SVD for 4x4 symmetric matrices (used internally by triangulation)
 * ============================================================================ */

/*
 * Eigendecompose a 4x4 symmetric matrix A^T*A to get the right singular vectors.
 *
 * Only needs V (right singular vectors) and S (singular values).
 * Uses Jacobi eigenvalue algorithm on the 4x4 normal equations matrix.
 *
 * V[4][4]: columns are eigenvectors (sorted by descending eigenvalue).
 * S[4]: singular values (descending).
 */
void pt_svd_4x4(const double ATA[4][4], double V[4][4], double S[4]);

/* ============================================================================
 * Undistortion
 * ============================================================================ */

/*
 * Undistort a single 2D point using the camera's intrinsic parameters.
 *
 * Implements the OpenCV distortion model (iterative inverse):
 *   1. Convert pixel (px, py) to normalized coordinates using K^{-1}
 *   2. Iteratively solve for undistorted normalized coords:
 *      x_d = x(1 + k1*r^2 + k2*r^4 + k3*r^6) + 2*p1*x*y + p2*(r^2 + 2*x^2)
 *      y_d = y(1 + k1*r^2 + k2*r^4 + k3*r^6) + p1*(r^2 + 2*y^2) + 2*p2*x*y
 *   3. Convert back to pixel coordinates using K
 *
 * Converges in ~5 iterations for typical distortion.
 * Output (out_ux, out_uy) is in pixel coordinates (same space as input).
 *
 * dist[5] = [k1, k2, p1, p2, k3]
 */
void pt_undistort_point(
    double px, double py,
    const double K[3][3],
    const double dist[5],
    double *out_ux, double *out_uy
);

/* ============================================================================
 * Triangulation
 * ============================================================================ */

/*
 * Triangulate one 3D point from 2D observations in multiple cameras.
 *
 * Algorithm (from triangulation.py:triangulate_keypoints):
 *   1. Undistort 2D points using each camera's intrinsics
 *   2. Build A matrix (2*num_views x 4):
 *      For each view i with projection matrix P[i] and undistorted point (x,y):
 *        A[2i]   = x * P[2,:] - P[0,:]
 *        A[2i+1] = y * P[2,:] - P[1,:]
 *   3. Compute A^T * A (4x4 symmetric)
 *   4. Eigendecompose: A^T*A = V * diag(eigenvalues) * V^T
 *   5. Solution = eigenvector with smallest eigenvalue (last column of V)
 *   6. point_3d = V[:3, last] / V[3, last]
 *
 * Returns 1 if successful, 0 if not enough views or degenerate.
 */
int pt_triangulate_point(
    const double undistorted_2d[][2],
    const int cam_indices[],
    int num_views,
    const PT_CameraConstants *constants,
    double out_3d[3]
);

/*
 * Triangulate all PT_NUM_KEYPOINTS keypoints for one person.
 *
 * Takes 2D keypoints from multiple camera views. For each keypoint:
 *   - Filters views where confidence >= confidence_threshold
 *   - Undistorts the 2D point
 *   - Triangulates if >= 2 valid views
 *
 * Fills out_candidate with:
 *   - xyz[k]: 3D position of keypoint k
 *   - valid[k]: 1 if keypoint k was triangulated
 *   - com_3d: center of mass from hip keypoints
 *   - com_valid: 1 if COM could be computed
 *   - views_used[]: which cameras contributed
 *   - num_views: number of views
 *
 * Returns 1 if at least one keypoint was triangulated, 0 otherwise.
 */
int pt_triangulate_person(
    const PT_Detection2D *detections[],
    const int cam_indices[],
    int num_views,
    const PT_CameraConstants *constants,
    float confidence_threshold,
    PT_Candidate3D *out_candidate
);

#ifdef __cplusplus
}
#endif

#endif /* PT_TRIANGULATION_H */
