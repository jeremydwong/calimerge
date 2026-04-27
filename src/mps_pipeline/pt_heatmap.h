/*
 * pt_heatmap.h - Heatmap decode with DARK sub-pixel refinement.
 *
 * Plain C port of CUDA kernel 5 (kernel_heatmap_decode_dark).
 * Operates on CPU — fast enough for small heatmaps (52 × 64 × 48).
 */

#ifndef PT_HEATMAP_H
#define PT_HEATMAP_H

#include "../pt_shared/pt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Decode heatmaps to 2D keypoints with DARK refinement.
 *
 * For each person, for each of 52 keypoints:
 *   1. Argmax over the 64×48 heatmap
 *   2. DARK refinement: gradient + Hessian shift for sub-pixel accuracy
 *   3. Map from heatmap coords to VitPose input coords (×4 stride)
 *   4. Apply affine transform to get original image coordinates
 *
 * heatmaps      - input: (num_persons, 52, 64, 48) fp32
 * affines        - input: (num_persons, 6) affine matrices
 * keypoints_out  - output: (num_persons, 52, 3) fp32 [x_image, y_image, conf]
 * num_persons    - number of person crops
 */
void pt_heatmap_decode(const float *heatmaps,
                       const float *affines,
                       float *keypoints_out,
                       int num_persons);

#ifdef __cplusplus
}
#endif

#endif /* PT_HEATMAP_H */
