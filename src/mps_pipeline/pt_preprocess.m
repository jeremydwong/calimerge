/*
 * pt_preprocess.m - Image preprocessing using Accelerate (vImage) on macOS.
 *
 * Implements letterbox resize, detection filtering, and VitPose crop+normalize.
 * Uses vImage from the Accelerate framework for image scaling.
 * Falls back to manual bilinear interpolation for crop operations.
 */

#import <Accelerate/Accelerate.h>
#include "pt_preprocess.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * Helpers
 * ============================================================================ */

static inline float clamp_f(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Bilinear sample one channel from BGR8 image */
static inline float bilinear_sample(const uint8_t *img, int w, int h,
                                    float x, float y, int channel) {
    float fx = clamp_f(x, 0.0f, (float)(w - 1));
    float fy = clamp_f(y, 0.0f, (float)(h - 1));

    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = x0 + 1 < w ? x0 + 1 : w - 1;
    int y1 = y0 + 1 < h ? y0 + 1 : h - 1;

    float dx = fx - (float)x0;
    float dy = fy - (float)y0;

    int stride = w * 3;
    float v00 = (float)img[y0 * stride + x0 * 3 + channel];
    float v10 = (float)img[y0 * stride + x1 * 3 + channel];
    float v01 = (float)img[y1 * stride + x0 * 3 + channel];
    float v11 = (float)img[y1 * stride + x1 * 3 + channel];

    float top    = v00 + dx * (v10 - v00);
    float bottom = v01 + dx * (v11 - v01);
    return top + dy * (bottom - top);
}

/* ============================================================================
 * YOLO letterbox preprocessing
 * ============================================================================ */

void pt_preprocess_letterbox(const uint8_t *src_bgr,
                             int src_w, int src_h,
                             float *dst,
                             int dst_w, int dst_h,
                             PT_LetterboxInfo *info) {
    /* Compute letterbox parameters */
    float scale_x = (float)dst_w / (float)src_w;
    float scale_y = (float)dst_h / (float)src_h;
    float scale   = fminf(scale_x, scale_y);

    int new_w = (int)(src_w * scale + 0.5f);
    int new_h = (int)(src_h * scale + 0.5f);
    int pad_x = (dst_w - new_w) / 2;
    int pad_y = (dst_h - new_h) / 2;

    if (info) {
        info->scale = scale;
        info->pad_x = pad_x;
        info->pad_y = pad_y;
    }

    int plane_size = dst_h * dst_w;
    float pad_val = (float)PT_LETTERBOX_PAD_VALUE / 255.0f;

    /* Fill with padding value */
    for (int i = 0; i < 3 * plane_size; i++) {
        dst[i] = pad_val;
    }

    /* Resize source into the non-padded region via bilinear interpolation */
    for (int dy = 0; dy < new_h; dy++) {
        for (int dx = 0; dx < new_w; dx++) {
            /* Map destination pixel to source */
            float sx = ((float)dx + 0.5f) / scale - 0.5f;
            float sy = ((float)dy + 0.5f) / scale - 0.5f;

            float b_raw = bilinear_sample(src_bgr, src_w, src_h, sx, sy, 0);
            float g_raw = bilinear_sample(src_bgr, src_w, src_h, sx, sy, 1);
            float r_raw = bilinear_sample(src_bgr, src_w, src_h, sx, sy, 2);

            int out_x = dx + pad_x;
            int out_y = dy + pad_y;
            int pixel_idx = out_y * dst_w + out_x;

            /* CHW RGB layout */
            dst[0 * plane_size + pixel_idx] = r_raw / 255.0f;
            dst[1 * plane_size + pixel_idx] = g_raw / 255.0f;
            dst[2 * plane_size + pixel_idx] = b_raw / 255.0f;
        }
    }
}

void pt_preprocess_letterbox_batch(const uint8_t **src_ptrs,
                                   int num_images,
                                   int src_w, int src_h,
                                   float *dst,
                                   int dst_w, int dst_h,
                                   PT_LetterboxInfo *info) {
    int image_size = 3 * dst_w * dst_h;

    for (int i = 0; i < num_images; i++) {
        pt_preprocess_letterbox(src_ptrs[i], src_w, src_h,
                                dst + i * image_size,
                                dst_w, dst_h,
                                i == 0 ? info : NULL);
    }
}

/* ============================================================================
 * YOLO detection filtering
 * ============================================================================ */

void pt_preprocess_filter_detections(const float *yolo_out,
                                     float *boxes,
                                     float *scores,
                                     int *counts,
                                     int batch,
                                     float conf_thresh,
                                     const PT_LetterboxInfo *info,
                                     int orig_w, int orig_h) {
    float scale = info->scale;
    int   pad_x = info->pad_x;
    int   pad_y = info->pad_y;

    memset(counts, 0, batch * sizeof(int));

    for (int img = 0; img < batch; img++) {
        const float *img_dets = yolo_out + img * PT_YOLO_MAX_RAW_DETS * 6;
        int count = 0;

        for (int d = 0; d < PT_YOLO_MAX_RAW_DETS; d++) {
            const float *det = img_dets + d * 6;
            float conf = det[4];
            int   cls  = (int)det[5];

            if (cls != PT_COCO_PERSON_CLASS || conf < conf_thresh)
                continue;
            if (count >= PT_MAX_DETECTIONS)
                break;

            /* Undo letterbox: subtract padding, divide by scale */
            float x1 = clamp_f((det[0] - (float)pad_x) / scale, 0, (float)(orig_w - 1));
            float y1 = clamp_f((det[1] - (float)pad_y) / scale, 0, (float)(orig_h - 1));
            float x2 = clamp_f((det[2] - (float)pad_x) / scale, 0, (float)(orig_w - 1));
            float y2 = clamp_f((det[3] - (float)pad_y) / scale, 0, (float)(orig_h - 1));

            int box_base = (img * PT_MAX_DETECTIONS + count) * 4;
            boxes[box_base + 0] = x1;
            boxes[box_base + 1] = y1;
            boxes[box_base + 2] = x2;
            boxes[box_base + 3] = y2;

            scores[img * PT_MAX_DETECTIONS + count] = conf;
            count++;
        }

        counts[img] = count;
    }
}

/* ============================================================================
 * VitPose crop + resize + normalize
 * ============================================================================ */

void pt_preprocess_crop_vitpose(const uint8_t *src_bgr,
                                int src_w, int src_h,
                                const float box[4],
                                float *dst,
                                float affine[6]) {
    float x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];

    /* Box center and size */
    float cx = (x1 + x2) * 0.5f;
    float cy = (y1 + y2) * 0.5f;
    float bw = x2 - x1;
    float bh = y2 - y1;

    /* Expand by 1.25x */
    bw *= PT_VITPOSE_BOX_PAD;
    bh *= PT_VITPOSE_BOX_PAD;

    /* Enforce 192:256 = 3:4 aspect ratio */
    float aspect = (float)PT_VITPOSE_INPUT_W / (float)PT_VITPOSE_INPUT_H;  /* 0.75 */
    if (bw / bh < aspect) {
        bw = bh * aspect;
    } else {
        bh = bw / aspect;
    }

    /* Affine: maps VitPose input coords -> image coords */
    float scale_x = bw / (float)PT_VITPOSE_INPUT_W;
    float scale_y = bh / (float)PT_VITPOSE_INPUT_H;
    float tx = cx - bw * 0.5f;
    float ty = cy - bh * 0.5f;

    affine[0] = scale_x;  affine[1] = 0.0f;    affine[2] = tx;
    affine[3] = 0.0f;     affine[4] = scale_y;  affine[5] = ty;

    int plane_size = PT_VITPOSE_INPUT_H * PT_VITPOSE_INPUT_W;

    /* For each output pixel, map back to source and sample */
    for (int dy = 0; dy < PT_VITPOSE_INPUT_H; dy++) {
        for (int dx = 0; dx < PT_VITPOSE_INPUT_W; dx++) {
            float sx = scale_x * ((float)dx + 0.5f) + tx;
            float sy = scale_y * ((float)dy + 0.5f) + ty;

            float b_raw = bilinear_sample(src_bgr, src_w, src_h, sx, sy, 0);
            float g_raw = bilinear_sample(src_bgr, src_w, src_h, sx, sy, 1);
            float r_raw = bilinear_sample(src_bgr, src_w, src_h, sx, sy, 2);

            /* ImageNet normalization: (pixel/255 - mean) / std, RGB order */
            float r_norm = (r_raw / 255.0f - PT_IMAGENET_MEAN_R) / PT_IMAGENET_STD_R;
            float g_norm = (g_raw / 255.0f - PT_IMAGENET_MEAN_G) / PT_IMAGENET_STD_G;
            float b_norm = (b_raw / 255.0f - PT_IMAGENET_MEAN_B) / PT_IMAGENET_STD_B;

            int pixel_idx = dy * PT_VITPOSE_INPUT_W + dx;

            dst[0 * plane_size + pixel_idx] = r_norm;
            dst[1 * plane_size + pixel_idx] = g_norm;
            dst[2 * plane_size + pixel_idx] = b_norm;
        }
    }
}

void pt_preprocess_crop_vitpose_batch(const uint8_t *src_bgr,
                                      int src_w, int src_h,
                                      const float *boxes,
                                      int num_det,
                                      float *dst,
                                      float *affines) {
    int crop_size = 3 * PT_VITPOSE_INPUT_H * PT_VITPOSE_INPUT_W;

    for (int i = 0; i < num_det; i++) {
        pt_preprocess_crop_vitpose(src_bgr, src_w, src_h,
                                   boxes + i * 4,
                                   dst + i * crop_size,
                                   affines + i * 6);
    }
}
