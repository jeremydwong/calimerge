/*
 * pt_matching.cpp - Cross-view epipolar matching, Hungarian algorithm, union-find.
 *
 * Translates the Python algorithms from:
 *   - tracker.py:group_detections_across_views_bipartite()
 *   - triangulation.py:calculate_fundamental_matrix()
 *   - triangulation.py:point_to_epipolar_line_distance()
 *
 * Key optimization over Python: fundamental matrices are precomputed once at
 * startup instead of recalculated every frame.
 */

#include "pt_matching.h"
#include <math.h>
#include <string.h>

/* ============================================================================
 * Internal helpers: small matrix operations
 *
 * We need SVD for 3x4 and 3x3 matrices, pseudoinverse for 3x4, and
 * basic matrix multiply. All sizes are known at compile time.
 * ============================================================================ */

/* 3x3 matrix multiply: C = A * B */
static void mat33_mul(const double A[3][3], const double B[3][3], double C[3][3])
{
    int i, j, k;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            double s = 0.0;
            for (k = 0; k < 3; k++) {
                s += A[i][k] * B[k][j];
            }
            C[i][j] = s;
        }
    }
}

/* 3x4 matrix multiply with 4x1 vector: out[3] = M[3][4] * v[4] */
static void mat34_mul_vec4(const double M[3][4], const double v[4], double out[3])
{
    int i, j;
    for (i = 0; i < 3; i++) {
        double s = 0.0;
        for (j = 0; j < 4; j++) {
            s += M[i][j] * v[j];
        }
        out[i] = s;
    }
}

/* Matrix multiply: C[m][p] = A[m][n] * B[n][p]  (generic, small sizes) */
static void matmul(const double *A, const double *B, double *C,
                   int m, int n, int p)
{
    int i, j, k;
    for (i = 0; i < m; i++) {
        for (j = 0; j < p; j++) {
            double s = 0.0;
            for (k = 0; k < n; k++) {
                s += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = s;
        }
    }
}

/* Transpose: B[n][m] = A[m][n]^T */
static void mat_transpose(const double *A, double *B, int m, int n)
{
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            B[j * m + i] = A[i * n + j];
        }
    }
}

/* Frobenius norm of a flat matrix */
static double mat_norm(const double *M, int size)
{
    double s = 0.0;
    int i;
    for (i = 0; i < size; i++) {
        s += M[i] * M[i];
    }
    return sqrt(s);
}

/* ============================================================================
 * Jacobi eigenvalue algorithm for symmetric 4x4 matrices
 *
 * Used to compute the SVD of the 3x4 projection matrix via the normal
 * equations: P^T * P = V * S^2 * V^T. We need V (right singular vectors)
 * and S (singular values).
 *
 * Also used for 3x3 symmetric matrices in the rank-2 enforcement step.
 * ============================================================================ */

/* Jacobi eigenvalue decomposition for NxN symmetric matrix (N <= 4).
 * A is destroyed. eigenvectors[i][j] = j-th component of i-th eigenvector.
 * eigenvalues[i] sorted descending. */
static void jacobi_eigen(double *A, double *eigenvalues, double *eigenvectors,
                         int n, int max_iter)
{
    int i, j, k, iter;

    /* Initialize eigenvectors to identity */
    memset(eigenvectors, 0, (size_t)(n * n) * sizeof(double));
    for (i = 0; i < n; i++) {
        eigenvectors[i * n + i] = 1.0;
    }

    for (iter = 0; iter < max_iter; iter++) {
        /* Find largest off-diagonal element */
        int p = 0, q = 1;
        double max_val = 0.0;
        for (i = 0; i < n; i++) {
            for (j = i + 1; j < n; j++) {
                double v = fabs(A[i * n + j]);
                if (v > max_val) {
                    max_val = v;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_val < 1e-15) break;

        /* Compute rotation angle */
        double app = A[p * n + p];
        double aqq = A[q * n + q];
        double apq = A[p * n + q];

        double theta;
        if (fabs(app - aqq) < 1e-15) {
            theta = 3.14159265358979323846 / 4.0;
        } else {
            theta = 0.5 * atan2(2.0 * apq, app - aqq);
        }

        double c = cos(theta);
        double s = sin(theta);

        /* Apply Givens rotation to A */
        for (k = 0; k < n; k++) {
            double akp = A[k * n + p];
            double akq = A[k * n + q];
            A[k * n + p] = c * akp + s * akq;
            A[k * n + q] = -s * akp + c * akq;
        }
        for (k = 0; k < n; k++) {
            double apk = A[p * n + k];
            double aqk = A[q * n + k];
            A[p * n + k] = c * apk + s * aqk;
            A[q * n + k] = -s * apk + c * aqk;
        }

        /* Update eigenvectors */
        for (k = 0; k < n; k++) {
            double vkp = eigenvectors[k * n + p];
            double vkq = eigenvectors[k * n + q];
            eigenvectors[k * n + p] = c * vkp + s * vkq;
            eigenvectors[k * n + q] = -s * vkp + c * vkq;
        }
    }

    /* Extract eigenvalues from diagonal */
    for (i = 0; i < n; i++) {
        eigenvalues[i] = A[i * n + i];
    }

    /* Sort eigenvalues descending, rearrange eigenvectors */
    for (i = 0; i < n - 1; i++) {
        int max_idx = i;
        for (j = i + 1; j < n; j++) {
            if (eigenvalues[j] > eigenvalues[max_idx]) {
                max_idx = j;
            }
        }
        if (max_idx != i) {
            double tmp = eigenvalues[i];
            eigenvalues[i] = eigenvalues[max_idx];
            eigenvalues[max_idx] = tmp;

            /* Swap eigenvector columns */
            for (k = 0; k < n; k++) {
                tmp = eigenvectors[k * n + i];
                eigenvectors[k * n + i] = eigenvectors[k * n + max_idx];
                eigenvectors[k * n + max_idx] = tmp;
            }
        }
    }
}

/*
 * SVD of a 3x4 matrix via normal equations.
 *
 * Given M (3x4), compute M^T * M (4x4 symmetric), eigendecompose to get V and S^2.
 * Then U can be recovered as M * V * S^{-1} if needed.
 *
 * Outputs:
 *   U[3][3]  - left singular vectors (columns)
 *   S[4]     - singular values (descending)
 *   Vt[4][4] - V^T (rows are right singular vectors)
 *
 * We need the last row of Vt (null space of M) for fundamental matrix computation.
 * We also need U and S for the 3x3 SVD rank-2 enforcement.
 */
static void svd_3x4(const double M[3][4],
                     double U[3][4], double S[4], double Vt[4][4])
{
    double Mt[4][3], MtM[4][4];
    double eigenvalues[4], eigenvectors[4 * 4];
    double MtM_copy[4 * 4];
    int i, j;

    /* M^T (4x3) */
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 4; j++) {
            Mt[j][i] = M[i][j];
        }
    }

    /* M^T * M (4x4) */
    matmul(&Mt[0][0], &M[0][0], &MtM[0][0], 4, 3, 4);

    /* Copy for Jacobi (it destroys the input) */
    memcpy(MtM_copy, MtM, sizeof(MtM_copy));

    /* Eigendecompose M^T * M = V * diag(eigenvalues) * V^T */
    jacobi_eigen(MtM_copy, eigenvalues, eigenvectors, 4, 200);

    /* Singular values = sqrt of eigenvalues */
    for (i = 0; i < 4; i++) {
        S[i] = (eigenvalues[i] > 0.0) ? sqrt(eigenvalues[i]) : 0.0;
    }

    /* V^T: rows are right singular vectors (eigenvectors are columns of V,
     * so V^T[i][j] = eigenvectors[j][i] = eigenvectors[j * 4 + i] ) */
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            Vt[i][j] = eigenvectors[j * 4 + i];
        }
    }

    /* U = M * V * S^{-1} (only for columns where S > 0) */
    /* U is 3x4 but only first 3 columns meaningful (since M is 3x4) */
    memset(U, 0, 3 * 4 * sizeof(double));
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 4; j++) {
            if (S[j] > 1e-12) {
                double s = 0.0;
                int k;
                for (k = 0; k < 4; k++) {
                    /* V column j = eigenvectors[row k, col j] = eigenvectors[k*4 + j] */
                    s += M[i][k] * eigenvectors[k * 4 + j];
                }
                U[i][j] = s / S[j];
            }
        }
    }
}

/*
 * SVD of a 3x3 matrix via normal equations + Jacobi.
 * Outputs U[3][3], S[3], Vt[3][3].
 */
static void svd_3x3(const double M[3][3],
                     double U[3][3], double S[3], double Vt[3][3])
{
    double Mt[3][3], MtM[3][3];
    double eigenvalues[3], eigenvectors[3 * 3];
    double MtM_copy[3 * 3];
    int i, j;

    /* M^T */
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            Mt[j][i] = M[i][j];
        }
    }

    /* M^T * M */
    mat33_mul(Mt, M, MtM);

    memcpy(MtM_copy, MtM, sizeof(MtM_copy));
    jacobi_eigen(MtM_copy, eigenvalues, eigenvectors, 3, 200);

    for (i = 0; i < 3; i++) {
        S[i] = (eigenvalues[i] > 0.0) ? sqrt(eigenvalues[i]) : 0.0;
    }

    /* V^T */
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            Vt[i][j] = eigenvectors[j * 3 + i];
        }
    }

    /* U = M * V * S^{-1} */
    memset(U, 0, 9 * sizeof(double));
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            if (S[j] > 1e-12) {
                double s = 0.0;
                int k;
                for (k = 0; k < 3; k++) {
                    s += M[i][k] * eigenvectors[k * 3 + j];
                }
                U[i][j] = s / S[j];
            }
        }
    }
}

/*
 * Pseudoinverse of a 3x4 matrix: pinv(M) is 4x3.
 * pinv(M) = V * S^{-1} * U^T
 * where M = U * S * V^T (from SVD).
 */
static void pinv_3x4(const double M[3][4], double pinv[4][3])
{
    double U[3][4], S[4], Vt[4][4];
    int i, j, k;

    svd_3x4(M, U, S, Vt);

    /* pinv = V * diag(1/S) * U^T
     * V[i][j] = Vt[j][i] (transpose of Vt)
     * U^T[j][k] = U[k][j] */
    memset(pinv, 0, 4 * 3 * sizeof(double));
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 3; j++) {
            double s = 0.0;
            for (k = 0; k < 3; k++) {
                /* Only the first 3 singular values are potentially nonzero for 3x4 */
                if (S[k] > 1e-12) {
                    double v_ik = Vt[k][i]; /* V[i][k] = Vt[k][i] */
                    double ut_kj = U[j][k]; /* U^T[k][j] = U[j][k] */
                    s += v_ik * (1.0 / S[k]) * ut_kj;
                }
            }
            pinv[i][j] = s;
        }
    }
}

/* ============================================================================
 * Union-Find
 * ============================================================================ */

void pt_uf_init(PT_UnionFind *uf, int n)
{
    int i;
    uf->count = n;
    for (i = 0; i < n; i++) {
        uf->parent[i] = i;
        uf->rank[i] = 0;
    }
}

int pt_uf_find(PT_UnionFind *uf, int x)
{
    /* Path compression */
    while (uf->parent[x] != x) {
        uf->parent[x] = uf->parent[uf->parent[x]];
        x = uf->parent[x];
    }
    return x;
}

void pt_uf_union(PT_UnionFind *uf, int x, int y)
{
    int rx = pt_uf_find(uf, x);
    int ry = pt_uf_find(uf, y);
    if (rx == ry) return;

    /* Union by rank */
    if (uf->rank[rx] < uf->rank[ry]) {
        uf->parent[rx] = ry;
    } else if (uf->rank[rx] > uf->rank[ry]) {
        uf->parent[ry] = rx;
    } else {
        uf->parent[ry] = rx;
        uf->rank[rx]++;
    }
}

/* ============================================================================
 * Projection matrix computation
 * ============================================================================ */

void pt_compute_projection_matrices(PT_CameraConstants *c)
{
    int cam;
    for (cam = 0; cam < c->num_cameras; cam++) {
        /* P = K * [R | t]
         * K is 3x3, R is 3x3, t is 3x1.
         * [R|t] is 3x4, P is 3x4. */
        double Rt[3][4];
        int i, j, k;

        /* Build [R | t] */
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                Rt[i][j] = c->rotation[cam][i][j];
            }
            Rt[i][3] = c->translation[cam][i];
        }

        /* P = K * Rt */
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 4; j++) {
                double s = 0.0;
                for (k = 0; k < 3; k++) {
                    s += c->camera_matrix[cam][i][k] * Rt[k][j];
                }
                c->projection[cam][i][j] = s;
            }
        }
    }
}

/* ============================================================================
 * Fundamental matrix precomputation
 * ============================================================================ */

void pt_precompute_fundamentals(PT_CameraConstants *c)
{
    int i, j;
    int nc = c->num_cameras;

    /* Clear all */
    memset(c->fundamental_valid, 0, sizeof(c->fundamental_valid));

    for (i = 0; i < nc; i++) {
        for (j = i + 1; j < nc; j++) {
            int idx_ij = i * PT_MAX_CAMERAS + j;
            int idx_ji = j * PT_MAX_CAMERAS + i;

            const double (*P1)[4] = (const double (*)[4])c->projection[i];
            const double (*P2)[4] = (const double (*)[4])c->projection[j];

            /* Step 1: Null space of P1 via SVD.
             * C1 = last row of V^T from SVD of P1 (smallest singular value). */
            double U1[3][4], S1[4], Vt1[4][4];
            svd_3x4(P1, U1, S1, Vt1);

            double C1[4];
            int k;
            for (k = 0; k < 4; k++) {
                C1[k] = Vt1[3][k]; /* Last row of V^T */
            }

            /* Normalize: C1 = C1 / C1[3] */
            if (fabs(C1[3]) < 1e-15) continue;
            for (k = 0; k < 4; k++) {
                C1[k] /= C1[3];
            }

            /* Step 2: e2 = P2 * C1 */
            double e2[3];
            mat34_mul_vec4(P2, C1, e2);

            /* Normalize: e2 = e2 / e2[2] */
            if (fabs(e2[2]) < 1e-15) continue;
            e2[0] /= e2[2];
            e2[1] /= e2[2];
            e2[2] = 1.0;

            /* Step 3: Skew-symmetric matrix [e2]_x */
            double e2x[3][3];
            e2x[0][0] = 0.0;     e2x[0][1] = -e2[2]; e2x[0][2] = e2[1];
            e2x[1][0] = e2[2];   e2x[1][1] = 0.0;    e2x[1][2] = -e2[0];
            e2x[2][0] = -e2[1];  e2x[2][1] = e2[0];  e2x[2][2] = 0.0;

            /* Step 4: F = [e2]_x * P2 * pinv(P1)
             * pinv(P1) is 4x3, P2 is 3x4, so P2*pinv(P1) is 3x3.
             * Then [e2]_x (3x3) * (3x3) = 3x3. */
            double P1_pinv[4][3];
            pinv_3x4(P1, P1_pinv);

            /* T = P2 * pinv(P1) (3x3) */
            double T[3][3];
            matmul(&P2[0][0], &P1_pinv[0][0], &T[0][0], 3, 4, 3);

            /* F = [e2]_x * T */
            double F[3][3];
            mat33_mul(e2x, T, F);

            /* Step 5: Enforce rank-2 via SVD of 3x3 F */
            double Uf[3][3], Sf[3], Vtf[3][3];
            svd_3x3(F, Uf, Sf, Vtf);

            /* Zero the smallest singular value */
            Sf[2] = 0.0;

            /* Reconstruct: F = Uf * diag(Sf) * Vtf */
            {
                double diag_S_Vt[3][3];
                int r, col;
                for (r = 0; r < 3; r++) {
                    for (col = 0; col < 3; col++) {
                        diag_S_Vt[r][col] = Sf[r] * Vtf[r][col];
                    }
                }
                mat33_mul(Uf, diag_S_Vt, F);
            }

            /* Step 6: Normalize */
            double norm = mat_norm(&F[0][0], 9);
            if (norm > 1e-15) {
                int ii;
                for (ii = 0; ii < 9; ii++) {
                    ((double *)F)[ii] /= norm;
                }
            }

            /* Store F[i][j] */
            memcpy(c->fundamental[idx_ij], F, sizeof(double) * 9);
            c->fundamental_valid[idx_ij] = 1;

            /* F[j][i] = F[i][j]^T */
            {
                int r, col;
                for (r = 0; r < 3; r++) {
                    for (col = 0; col < 3; col++) {
                        c->fundamental[idx_ji][r][col] = F[col][r];
                    }
                }
            }
            c->fundamental_valid[idx_ji] = 1;
        }
    }
}

/* ============================================================================
 * Epipolar distance
 * ============================================================================ */

double pt_epipolar_distance(
    const double F[3][3],
    double p1_x, double p1_y,
    double p2_x, double p2_y)
{
    /* l = F * [p1_x, p1_y, 1]^T */
    double l[3];
    l[0] = F[0][0] * p1_x + F[0][1] * p1_y + F[0][2];
    l[1] = F[1][0] * p1_x + F[1][1] * p1_y + F[1][2];
    l[2] = F[2][0] * p1_x + F[2][1] * p1_y + F[2][2];

    /* distance = |l . [p2_x, p2_y, 1]| / sqrt(l[0]^2 + l[1]^2) */
    double numerator = fabs(l[0] * p2_x + l[1] * p2_y + l[2]);
    double denominator = sqrt(l[0] * l[0] + l[1] * l[1]);

    if (denominator < 1e-10) {
        return HUGE_VAL;
    }
    return numerator / denominator;
}

/* ============================================================================
 * Hungarian algorithm (Munkres)
 *
 * Implementation for rectangular cost matrices up to 16x16.
 * Standard Jonker-Volgenant / Munkres approach with row and column reduction,
 * augmenting paths via shortest-path (Dijkstra) for non-square matrices.
 *
 * We pad the matrix to square (max(n_rows, n_cols)) for simplicity.
 * ============================================================================ */

#define HUNG_MAX_DIM 32  /* Must be >= PT_MAX_DETECTIONS */

void pt_hungarian(
    const double *cost_matrix,
    int n_rows, int n_cols,
    int *row_assign,
    int *col_assign)
{
    int n, i, j;
    double cost[HUNG_MAX_DIM][HUNG_MAX_DIM];
    double u[HUNG_MAX_DIM + 1], v[HUNG_MAX_DIM + 1];
    int p[HUNG_MAX_DIM + 1], way[HUNG_MAX_DIM + 1];

    /* Initialize outputs to -1 */
    for (i = 0; i < n_rows; i++) row_assign[i] = -1;
    for (j = 0; j < n_cols; j++) col_assign[j] = -1;

    /* Pad to square matrix */
    n = (n_rows > n_cols) ? n_rows : n_cols;
    if (n == 0) return;

    /* Fill cost matrix; pad with zeros */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i < n_rows && j < n_cols) {
                cost[i][j] = cost_matrix[i * n_cols + j];
            } else {
                cost[i][j] = 0.0;
            }
        }
    }

    /*
     * Classical Hungarian (potential-based, 1-indexed internally).
     * Based on the O(n^3) implementation by Kuhn/Munkres.
     * u[] and v[] are dual variables (potentials).
     * p[j] = row assigned to column j (1-indexed; 0 = unassigned).
     */
    memset(u, 0, sizeof(u));
    memset(v, 0, sizeof(v));
    memset(p, 0, sizeof(p));

    for (i = 1; i <= n; i++) {
        double minv[HUNG_MAX_DIM + 1];
        int used[HUNG_MAX_DIM + 1];
        int j0, j1;

        p[0] = i;
        j0 = 0;

        for (j = 0; j <= n; j++) {
            minv[j] = HUGE_VAL;
            used[j] = 0;
        }

        do {
            int i0;
            double delta;

            used[j0] = 1;
            i0 = p[j0];
            delta = HUGE_VAL;
            j1 = -1;

            for (j = 1; j <= n; j++) {
                if (!used[j]) {
                    double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            /* Update potentials */
            for (j = 0; j <= n; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
        } while (p[j0] != 0);

        /* Augmenting path: update assignments */
        do {
            int j_prev = way[j0];
            p[j0] = p[j_prev];
            j0 = j_prev;
        } while (j0 != 0);
    }

    /* Extract results (convert back to 0-indexed) */
    for (j = 1; j <= n; j++) {
        int row = p[j] - 1;
        int col = j - 1;
        if (row >= 0 && row < n_rows && col < n_cols) {
            row_assign[row] = col;
            col_assign[col] = row;
        }
    }
}

/* ============================================================================
 * Cross-view matching
 * ============================================================================ */

int pt_match_cross_view(
    const PT_Detection2D detections[PT_MAX_CAMERAS][PT_MAX_DETECTIONS],
    const int detection_counts[PT_MAX_CAMERAS],
    const PT_CameraConstants *constants,
    float epipolar_threshold,
    PT_Group *out_groups,
    int max_groups)
{
    int nc = constants->num_cameras;
    int ci, cj;

    /*
     * Union-find elements: each element is (port_index * PT_MAX_DETECTIONS + detection_index).
     * Max elements = PT_MAX_CAMERAS * PT_MAX_DETECTIONS = 16 * 16 = 256.
     */
    PT_UnionFind uf;
    int total_elements = nc * PT_MAX_DETECTIONS;
    pt_uf_init(&uf, total_elements);

    /* Track which elements are actually used (have valid detections) */
    int element_active[256];
    memset(element_active, 0, sizeof(element_active));
    for (ci = 0; ci < nc; ci++) {
        int di;
        for (di = 0; di < detection_counts[ci]; di++) {
            if (detections[ci][di].valid) {
                element_active[ci * PT_MAX_DETECTIONS + di] = 1;
            }
        }
    }

    /* For each camera pair, compute cost matrix and run Hungarian */
    for (ci = 0; ci < nc; ci++) {
        int n1 = detection_counts[ci];
        if (n1 <= 0) continue;

        for (cj = ci + 1; cj < nc; cj++) {
            int n2 = detection_counts[cj];
            if (n2 <= 0) continue;

            /* Look up precomputed fundamental matrix */
            int fidx = ci * PT_MAX_CAMERAS + cj;
            if (!constants->fundamental_valid[fidx]) continue;

            const double (*F)[3] = (const double (*)[3])constants->fundamental[fidx];

            /* Build cost matrix */
            double cost[PT_MAX_DETECTIONS * PT_MAX_DETECTIONS];
            int a, b;
            for (a = 0; a < n1; a++) {
                for (b = 0; b < n2; b++) {
                    if (!detections[ci][a].valid || !detections[cj][b].valid) {
                        cost[a * n2 + b] = 1000.0;
                        continue;
                    }
                    double dist = pt_epipolar_distance(
                        F,
                        (double)detections[ci][a].com_2d[0],
                        (double)detections[ci][a].com_2d[1],
                        (double)detections[cj][b].com_2d[0],
                        (double)detections[cj][b].com_2d[1]
                    );
                    if (dist < (double)epipolar_threshold) {
                        cost[a * n2 + b] = dist;
                    } else {
                        cost[a * n2 + b] = 1000.0;
                    }
                }
            }

            /* Run Hungarian */
            int row_assign[PT_MAX_DETECTIONS];
            int col_assign[PT_MAX_DETECTIONS];
            pt_hungarian(cost, n1, n2, row_assign, col_assign);

            /* Accept matches where cost < threshold and union them */
            for (a = 0; a < n1; a++) {
                b = row_assign[a];
                if (b >= 0 && cost[a * n2 + b] < (double)epipolar_threshold) {
                    int elem_i = ci * PT_MAX_DETECTIONS + a;
                    int elem_j = cj * PT_MAX_DETECTIONS + b;
                    pt_uf_union(&uf, elem_i, elem_j);
                }
            }
        }
    }

    /*
     * Collect groups from union-find.
     *
     * Walk all active elements, find their root, and group them.
     * A group must have members from at least 2 different cameras.
     */

    /* Map root -> group index. Use a simple linear scan (at most 256 elements). */
    int group_roots[PT_MAX_GROUPS];
    int num_groups = 0;

    /* Temporary storage: which group index each root maps to */
    int root_to_group[256];
    memset(root_to_group, -1, sizeof(root_to_group));

    /* Temporary groups */
    PT_Group temp_groups[PT_MAX_GROUPS];
    memset(temp_groups, 0, sizeof(temp_groups));

    for (ci = 0; ci < nc; ci++) {
        int di;
        for (di = 0; di < detection_counts[ci]; di++) {
            int elem = ci * PT_MAX_DETECTIONS + di;
            if (!element_active[elem]) continue;

            int root = pt_uf_find(&uf, elem);
            int gi = root_to_group[root];

            if (gi < 0) {
                /* New group */
                if (num_groups >= PT_MAX_GROUPS) continue;
                gi = num_groups++;
                group_roots[gi] = root;
                root_to_group[root] = gi;
                temp_groups[gi].num_members = 0;
            }

            /* Add this detection to the group */
            if (temp_groups[gi].num_members < PT_MAX_CAMERAS) {
                int mi = temp_groups[gi].num_members;
                temp_groups[gi].members[mi].port_index = ci;
                temp_groups[gi].members[mi].detection_index = di;
                temp_groups[gi].num_members++;
            }
        }
    }

    /* Filter: require >= 2 cameras per group.
     * Also check that at least 2 *different* cameras are represented. */
    int out_count = 0;
    int gi;
    for (gi = 0; gi < num_groups; gi++) {
        if (out_count >= max_groups) break;

        PT_Group *g = &temp_groups[gi];
        if (g->num_members < 2) continue;

        /* Check for at least 2 distinct cameras */
        int first_cam = g->members[0].port_index;
        int has_different = 0;
        int mi;
        for (mi = 1; mi < g->num_members; mi++) {
            if (g->members[mi].port_index != first_cam) {
                has_different = 1;
                break;
            }
        }
        if (!has_different) continue;

        memcpy(&out_groups[out_count], g, sizeof(PT_Group));
        out_count++;
    }

    return out_count;
}
