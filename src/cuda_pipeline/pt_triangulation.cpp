/*
 * pt_triangulation.cpp - SVD triangulation and point undistortion.
 *
 * Translates the Python algorithms from triangulation.py:
 *   - triangulate_keypoints(): multi-view SVD triangulation
 *   - cv2.undistortPoints(): iterative undistortion
 *
 * The SVD for triangulation works on the 4x4 normal equations matrix A^T*A
 * (since A is at most 32x4 and we only need the null space). The 4x4
 * eigendecomposition uses Jacobi rotations -- simple, stable, converges
 * fast for our tiny matrices.
 */

#include "pt_triangulation.h"
#include <math.h>
#include <string.h>

/* ============================================================================
 * Jacobi eigenvalue algorithm for 4x4 symmetric matrices
 * ============================================================================ */

void pt_svd_4x4(const double ATA[4][4], double V[4][4], double S[4])
{
    double A[4][4];
    int i, j, k, iter;

    /* Copy input (Jacobi destroys the matrix) */
    memcpy(A, ATA, sizeof(A));

    /* Initialize V to identity */
    memset(V, 0, 16 * sizeof(double));
    for (i = 0; i < 4; i++) {
        V[i][i] = 1.0;
    }

    /* Jacobi iterations */
    for (iter = 0; iter < 200; iter++) {
        /* Find largest off-diagonal element */
        int p = 0, q = 1;
        double max_val = 0.0;
        for (i = 0; i < 4; i++) {
            for (j = i + 1; j < 4; j++) {
                double v = fabs(A[i][j]);
                if (v > max_val) {
                    max_val = v;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_val < 1e-15) break;

        /* Compute rotation angle */
        double app = A[p][p];
        double aqq = A[q][q];
        double apq = A[p][q];
        double theta;

        if (fabs(app - aqq) < 1e-15) {
            theta = 3.14159265358979323846 / 4.0;
        } else {
            theta = 0.5 * atan2(2.0 * apq, app - aqq);
        }

        double c = cos(theta);
        double s = sin(theta);

        /* Apply Givens rotation to A: A' = G^T * A * G */
        /* First: A <- A * G (affects columns p, q) */
        for (k = 0; k < 4; k++) {
            double akp = A[k][p];
            double akq = A[k][q];
            A[k][p] = c * akp + s * akq;
            A[k][q] = -s * akp + c * akq;
        }
        /* Then: A <- G^T * A (affects rows p, q) */
        for (k = 0; k < 4; k++) {
            double apk = A[p][k];
            double aqk = A[q][k];
            A[p][k] = c * apk + s * aqk;
            A[q][k] = -s * apk + c * aqk;
        }

        /* Update eigenvectors: V <- V * G */
        for (k = 0; k < 4; k++) {
            double vkp = V[k][p];
            double vkq = V[k][q];
            V[k][p] = c * vkp + s * vkq;
            V[k][q] = -s * vkp + c * vkq;
        }
    }

    /* Extract eigenvalues from diagonal */
    double eigenvalues[4];
    for (i = 0; i < 4; i++) {
        eigenvalues[i] = A[i][i];
    }

    /* Sort eigenvalues descending and rearrange V columns accordingly */
    for (i = 0; i < 3; i++) {
        int max_idx = i;
        for (j = i + 1; j < 4; j++) {
            if (eigenvalues[j] > eigenvalues[max_idx]) {
                max_idx = j;
            }
        }
        if (max_idx != i) {
            /* Swap eigenvalues */
            double tmp = eigenvalues[i];
            eigenvalues[i] = eigenvalues[max_idx];
            eigenvalues[max_idx] = tmp;

            /* Swap V columns */
            for (k = 0; k < 4; k++) {
                tmp = V[k][i];
                V[k][i] = V[k][max_idx];
                V[k][max_idx] = tmp;
            }
        }
    }

    /* Singular values = sqrt(eigenvalues) */
    for (i = 0; i < 4; i++) {
        S[i] = (eigenvalues[i] > 0.0) ? sqrt(eigenvalues[i]) : 0.0;
    }
}

/* ============================================================================
 * Undistortion
 * ============================================================================ */

void pt_undistort_point(
    double px, double py,
    const double K[3][3],
    const double dist[5],
    double *out_ux, double *out_uy)
{
    double fx = K[0][0];
    double fy = K[1][1];
    double cx = K[0][2];
    double cy = K[1][2];

    double k1 = dist[0];
    double k2 = dist[1];
    double p1 = dist[2];
    double p2 = dist[3];
    double k3 = dist[4];

    /* Convert pixel to normalized coordinates */
    double x_d = (px - cx) / fx;
    double y_d = (py - cy) / fy;

    /*
     * Iterative undistortion: start from distorted normalized coords,
     * iteratively solve for undistorted coords.
     *
     * The distortion model is:
     *   x_d = x * (1 + k1*r^2 + k2*r^4 + k3*r^6) + 2*p1*x*y + p2*(r^2 + 2*x^2)
     *   y_d = y * (1 + k1*r^2 + k2*r^4 + k3*r^6) + p1*(r^2 + 2*y^2) + 2*p2*x*y
     *
     * We want to find (x, y) given (x_d, y_d).
     * Start with x = x_d, y = y_d, and iterate.
     */
    double x = x_d;
    double y = y_d;
    int iter;

    for (iter = 0; iter < 10; iter++) {
        double r2 = x * x + y * y;
        double r4 = r2 * r2;
        double r6 = r4 * r2;

        double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

        double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

        x = (x_d - dx) / radial;
        y = (y_d - dy) / radial;
    }

    /* Convert back to pixel coordinates */
    *out_ux = x * fx + cx;
    *out_uy = y * fy + cy;
}

/* ============================================================================
 * Triangulation
 * ============================================================================ */

int pt_triangulate_point(
    const double undistorted_2d[][2],
    const int cam_indices[],
    int num_views,
    const PT_CameraConstants *constants,
    double out_3d[3])
{
    if (num_views < 2) return 0;

    /*
     * Build A matrix: 2*num_views rows x 4 columns.
     * For each view i with projection matrix P and undistorted point (x, y):
     *   A[2i]   = x * P[2,:] - P[0,:]
     *   A[2i+1] = y * P[2,:] - P[1,:]
     *
     * Max size: 2 * PT_MAX_CAMERAS * 4 = 32 * 4 = 128 doubles.
     */
    double A[PT_MAX_CAMERAS * 2][4];
    int n_rows = num_views * 2;
    int i, j, k;

    for (i = 0; i < num_views; i++) {
        int ci = cam_indices[i];
        double x = undistorted_2d[i][0];
        double y = undistorted_2d[i][1];

        const double (*P)[4] = (const double (*)[4])constants->projection[ci];

        for (j = 0; j < 4; j++) {
            A[2 * i][j]     = x * P[2][j] - P[0][j];
            A[2 * i + 1][j] = y * P[2][j] - P[1][j];
        }
    }

    /* Compute A^T * A (4x4 symmetric) */
    double ATA[4][4];
    for (i = 0; i < 4; i++) {
        for (j = i; j < 4; j++) {
            double s = 0.0;
            for (k = 0; k < n_rows; k++) {
                s += A[k][i] * A[k][j];
            }
            ATA[i][j] = s;
            ATA[j][i] = s; /* Symmetric */
        }
    }

    /* Eigendecompose A^T*A */
    double V[4][4], S[4];
    pt_svd_4x4(ATA, V, S);

    /*
     * The solution is the eigenvector with the smallest eigenvalue
     * (last column of V after sorting descending).
     */
    double w = V[3][3]; /* Homogeneous coordinate */
    if (fabs(w) < 1e-10) return 0;

    out_3d[0] = V[0][3] / w;
    out_3d[1] = V[1][3] / w;
    out_3d[2] = V[2][3] / w;

    /* Sanity check: reject points that are clearly wrong (behind all cameras, etc.) */
    /* For now, just check for NaN/Inf */
    if (out_3d[0] != out_3d[0] || out_3d[1] != out_3d[1] || out_3d[2] != out_3d[2]) {
        return 0; /* NaN check */
    }

    return 1;
}

int pt_triangulate_person(
    const PT_Detection2D *detections[],
    const int cam_indices[],
    int num_views,
    const PT_CameraConstants *constants,
    float confidence_threshold,
    PT_Candidate3D *out_candidate)
{
    int kp, vi;
    int any_valid = 0;

    /* Initialize output */
    memset(out_candidate, 0, sizeof(PT_Candidate3D));

    /* Record which views are used */
    out_candidate->num_views = num_views;
    for (vi = 0; vi < num_views && vi < PT_MAX_CAMERAS; vi++) {
        out_candidate->views_used[vi] = cam_indices[vi];
    }

    /* Triangulate each keypoint independently */
    for (kp = 0; kp < PT_NUM_KEYPOINTS; kp++) {
        /* Collect valid 2D observations for this keypoint */
        double undistorted[PT_MAX_CAMERAS][2];
        int valid_cams[PT_MAX_CAMERAS];
        int n_valid = 0;

        for (vi = 0; vi < num_views; vi++) {
            const PT_Detection2D *det = detections[vi];
            if (!det || !det->valid) continue;

            float conf = det->keypoints[kp][2];
            if (conf < confidence_threshold) continue;

            float raw_x = det->keypoints[kp][0];
            float raw_y = det->keypoints[kp][1];

            /* Skip NaN keypoints */
            if (raw_x != raw_x || raw_y != raw_y) continue;

            /* Undistort this 2D point using the camera's intrinsics */
            int ci = cam_indices[vi];
            double ux, uy;
            pt_undistort_point(
                (double)raw_x, (double)raw_y,
                constants->camera_matrix[ci],
                constants->distortion[ci],
                &ux, &uy
            );

            undistorted[n_valid][0] = ux;
            undistorted[n_valid][1] = uy;
            valid_cams[n_valid] = ci;
            n_valid++;
        }

        if (n_valid < 2) {
            out_candidate->valid[kp] = 0;
            continue;
        }

        /* Triangulate */
        double xyz[3];
        int ok = pt_triangulate_point(
            (const double (*)[2])undistorted,
            valid_cams,
            n_valid,
            constants,
            xyz
        );

        if (ok) {
            out_candidate->xyz[kp][0] = xyz[0];
            out_candidate->xyz[kp][1] = xyz[1];
            out_candidate->xyz[kp][2] = xyz[2];
            out_candidate->valid[kp] = 1;
            any_valid = 1;
        } else {
            out_candidate->valid[kp] = 0;
        }
    }

    /*
     * Compute center of mass from hip keypoints.
     * Hip indices: PT_HIP_LEFT_INDEX (11), PT_HIP_RIGHT_INDEX (12).
     */
    out_candidate->com_valid = 0;
    {
        int hip_indices[2] = { PT_HIP_LEFT_INDEX, PT_HIP_RIGHT_INDEX };
        double com[3] = { 0.0, 0.0, 0.0 };
        int n_hips = 0;
        int h;

        for (h = 0; h < 2; h++) {
            int hi = hip_indices[h];
            if (out_candidate->valid[hi]) {
                com[0] += out_candidate->xyz[hi][0];
                com[1] += out_candidate->xyz[hi][1];
                com[2] += out_candidate->xyz[hi][2];
                n_hips++;
            }
        }

        if (n_hips >= 1) {
            out_candidate->com_3d[0] = com[0] / (double)n_hips;
            out_candidate->com_3d[1] = com[1] / (double)n_hips;
            out_candidate->com_3d[2] = com[2] / (double)n_hips;
            out_candidate->com_valid = 1;
        }
    }

    return any_valid ? 1 : 0;
}
