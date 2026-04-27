/*
 * pt_heatmap.c - Heatmap DARK decode in plain C.
 *
 * Direct port of kernel_heatmap_decode_dark from pt_kernels.cu.
 * No GPU dependencies — runs on CPU.  For 52 keypoints × 64×48 heatmaps
 * this completes in <1ms on any modern CPU.
 */

#include "pt_heatmap.h"
#include <math.h>

static inline float clamp_f(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void pt_heatmap_decode(const float *heatmaps,
                       const float *affines,
                       float *keypoints_out,
                       int num_persons) {
    const int hm_w = PT_VITPOSE_HEATMAP_W;  /* 48 */
    const int hm_h = PT_VITPOSE_HEATMAP_H;  /* 64 */
    const int hm_size = hm_w * hm_h;

    for (int p = 0; p < num_persons; p++) {
        const float *a = affines + p * 6;
        float a00 = a[0];  /* scale_x */
        float a01 = a[1];  /* 0 */
        float a02 = a[2];  /* tx */
        float a10 = a[3];  /* 0 */
        float a11 = a[4];  /* scale_y */
        float a12 = a[5];  /* ty */

        for (int kp = 0; kp < PT_NUM_KEYPOINTS; kp++) {
            const float *hm = heatmaps
                             + p * PT_NUM_KEYPOINTS * hm_size
                             + kp * hm_size;

            /* Step 1: Argmax */
            float max_val = -1e30f;
            int   max_idx = 0;

            for (int i = 0; i < hm_size; i++) {
                if (hm[i] > max_val) {
                    max_val = hm[i];
                    max_idx = i;
                }
            }

            int px = max_idx % hm_w;
            int py = max_idx / hm_w;
            float confidence = max_val;

            /* Step 2: DARK sub-pixel refinement */
            float sub_x = (float)px;
            float sub_y = (float)py;

            if (px > 0 && px < hm_w - 1 && py > 0 && py < hm_h - 1) {
                float dx = (hm[py * hm_w + (px + 1)] - hm[py * hm_w + (px - 1)]) * 0.5f;
                float dy = (hm[(py + 1) * hm_w + px] - hm[(py - 1) * hm_w + px]) * 0.5f;

                float dxx = hm[py * hm_w + (px + 1)]
                           + hm[py * hm_w + (px - 1)]
                           - 2.0f * max_val;
                float dyy = hm[(py + 1) * hm_w + px]
                           + hm[(py - 1) * hm_w + px]
                           - 2.0f * max_val;

                if (fabsf(dxx) > 1e-6f)
                    sub_x -= 0.5f * dx / dxx;
                if (fabsf(dyy) > 1e-6f)
                    sub_y -= 0.5f * dy / dyy;

                sub_x = clamp_f(sub_x, 0.0f, (float)(hm_w - 1));
                sub_y = clamp_f(sub_y, 0.0f, (float)(hm_h - 1));
            }

            /* Step 3: Heatmap coords -> VitPose input coords (4x stride) */
            float vp_x = sub_x * 4.0f + 1.5f;
            float vp_y = sub_y * 4.0f + 1.5f;

            /* Step 4: Apply affine (VitPose coords -> image coords) */
            float img_x = a00 * vp_x + a01 * vp_y + a02;
            float img_y = a10 * vp_x + a11 * vp_y + a12;

            /* Write output */
            float *out = keypoints_out + (p * PT_NUM_KEYPOINTS + kp) * 3;
            out[0] = img_x;
            out[1] = img_y;
            out[2] = confidence;
        }
    }
}
