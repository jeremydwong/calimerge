/*
 * pt_kernels.cu - CUDA preprocessing/postprocessing kernels for pose tracking.
 *
 * Five kernels:
 *   1. NV12 -> BGR8 colorspace conversion
 *   2. Letterbox resize + normalize for YOLO (BGR8 -> fp32 CHW RGB 640x640)
 *   3. Filter YOLO detections (threshold, undo letterbox)
 *   4. Crop + resize + normalize for VitPose (BGR8 -> fp32 CHW RGB 256x192)
 *   5. Heatmap decode with DARK refinement (heatmaps -> 2D keypoints)
 *
 * Style: plain C structs, free functions, no STL, no exceptions.
 * All launches are async on the caller's CUDA stream.
 */

#include "pt_kernels.h"
#include "pt_common.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

/* =========================================================================
 * Device helpers
 * ========================================================================= */

/* Clamp integer to [lo, hi]. */
__device__ __forceinline__ int clamp_i(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Clamp float to [lo, hi]. */
__device__ __forceinline__ float clamp_f(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

/* Bilinear sample from a BGR8 image.  Returns one channel (0=B, 1=G, 2=R).
 * Coordinates are in pixel units (float).  Clamp-to-edge. */
__device__ __forceinline__
float bilinear_sample_u8(const uint8_t *img, int w, int h, int stride,
                         float x, float y, int channel) {
    float fx = clamp_f(x, 0.0f, (float)(w - 1));
    float fy = clamp_f(y, 0.0f, (float)(h - 1));

    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = min(x0 + 1, w - 1);
    int y1 = min(y0 + 1, h - 1);

    float dx = fx - (float)x0;
    float dy = fy - (float)y0;

    float v00 = (float)img[y0 * stride + x0 * 3 + channel];
    float v10 = (float)img[y0 * stride + x1 * 3 + channel];
    float v01 = (float)img[y1 * stride + x0 * 3 + channel];
    float v11 = (float)img[y1 * stride + x1 * 3 + channel];

    float top    = v00 + dx * (v10 - v00);
    float bottom = v01 + dx * (v11 - v01);
    return top + dy * (bottom - top);
}


/* =========================================================================
 * Kernel 1: NV12 -> BGR8
 * =========================================================================
 *
 * NV12 layout:
 *   Y plane:  h rows of w bytes (luma)
 *   UV plane: h/2 rows of w bytes (interleaved Cb,Cr pairs)
 *
 * Each thread handles one pixel.
 * Grid:  (w/32, h/2), Block: (32, 2)
 */

__global__
void kernel_nv12_to_bgr(const uint8_t * __restrict__ nv12,
                         uint8_t * __restrict__ bgr,
                         int w, int h,
                         int nv12_stride, int bgr_stride) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= w || py >= h) return;

    /* Y value for this pixel */
    float y_val = (float)nv12[py * nv12_stride + px];

    /* UV pair: subsampled 2x2.  UV plane starts at offset nv12_stride * h. */
    int uv_row = py >> 1;
    int uv_col = (px & ~1);    /* round down to even */
    const uint8_t *uv_plane = nv12 + nv12_stride * h;
    float u_val = (float)uv_plane[uv_row * nv12_stride + uv_col];      /* Cb */
    float v_val = (float)uv_plane[uv_row * nv12_stride + uv_col + 1];  /* Cr */

    /* BT.601 YUV -> RGB */
    float c = y_val - 16.0f;
    float d = u_val - 128.0f;
    float e = v_val - 128.0f;

    float r = 1.164383f * c + 1.596027f * e;
    float g = 1.164383f * c - 0.391762f * d - 0.812968f * e;
    float b = 1.164383f * c + 2.017232f * d;

    /* Clamp to [0, 255] and write BGR */
    int out_idx = py * bgr_stride + px * 3;
    bgr[out_idx + 0] = (uint8_t)clamp_f(b + 0.5f, 0.0f, 255.0f);
    bgr[out_idx + 1] = (uint8_t)clamp_f(g + 0.5f, 0.0f, 255.0f);
    bgr[out_idx + 2] = (uint8_t)clamp_f(r + 0.5f, 0.0f, 255.0f);
}


/* =========================================================================
 * Kernel 2: Letterbox resize + normalize for YOLO
 * =========================================================================
 *
 * Each thread computes one pixel in the (dst_w, dst_h) output.
 * The kernel is launched once per image in the batch; the batch index
 * is passed via blockIdx.z.
 *
 * Output is fp16 CHW RGB: channel-first, BGR->RGB reorder, /255 normalize.
 * Letterbox padding areas are filled with 114/255 in fp16.
 */

__global__
void kernel_letterbox_resize_normalize(const uint8_t * const * __restrict__ src_ptrs,
                                       __half * __restrict__ dst,
                                       int num_images,
                                       int src_w, int src_h,
                                       int dst_w, int dst_h) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    int img_idx = blockIdx.z;

    if (dx >= dst_w || dy >= dst_h || img_idx >= num_images) return;

    /* Compute letterbox parameters */
    float scale_x = (float)dst_w / (float)src_w;
    float scale_y = (float)dst_h / (float)src_h;
    float scale   = fminf(scale_x, scale_y);

    int new_w = (int)(src_w * scale + 0.5f);
    int new_h = (int)(src_h * scale + 0.5f);
    int pad_x = (dst_w - new_w) / 2;
    int pad_y = (dst_h - new_h) / 2;

    /* Per-image, per-channel output offset: (img, C, H, W) layout */
    int plane_size = dst_h * dst_w;
    int img_offset = img_idx * 3 * plane_size;

    /* Check if this pixel is in the padded region */
    int rx = dx - pad_x;
    int ry = dy - pad_y;

    float r_val, g_val, b_val;

    if (rx < 0 || rx >= new_w || ry < 0 || ry >= new_h) {
        /* Padding region: gray = 114/255 */
        float pad = (float)PT_LETTERBOX_PAD_VALUE / 255.0f;
        r_val = pad;
        g_val = pad;
        b_val = pad;
    } else {
        /* Map back to source coordinates */
        float sx = ((float)rx + 0.5f) / scale - 0.5f;
        float sy = ((float)ry + 0.5f) / scale - 0.5f;

        const uint8_t *src = src_ptrs[img_idx];
        int src_stride = src_w * 3;

        /* Bilinear sample BGR channels */
        float b_raw = bilinear_sample_u8(src, src_w, src_h, src_stride, sx, sy, 0);
        float g_raw = bilinear_sample_u8(src, src_w, src_h, src_stride, sx, sy, 1);
        float r_raw = bilinear_sample_u8(src, src_w, src_h, src_stride, sx, sy, 2);

        /* Normalize to [0, 1] */
        r_val = r_raw / 255.0f;
        g_val = g_raw / 255.0f;
        b_val = b_raw / 255.0f;
    }

    /* Write as CHW RGB (channel 0 = R, channel 1 = G, channel 2 = B) */
    int pixel_idx = dy * dst_w + dx;
    dst[img_offset + 0 * plane_size + pixel_idx] = __float2half(r_val);
    dst[img_offset + 1 * plane_size + pixel_idx] = __float2half(g_val);
    dst[img_offset + 2 * plane_size + pixel_idx] = __float2half(b_val);
}


/* =========================================================================
 * Kernel 3: Filter YOLO detections
 * =========================================================================
 *
 * Input:  (batch, 300, 6) where 6 = [x1, y1, x2, y2, conf, cls]
 *         Coordinates are in 640x640 letterbox space.
 *
 * Output: Filtered boxes in original image coordinates.
 *         Only class=0 (person) with conf > threshold are kept.
 *
 * Grid: (ceil(300/256), batch), Block: (256, 1)
 */

__global__
void kernel_filter_detections(const float * __restrict__ yolo_out,
                              float * __restrict__ boxes,
                              float * __restrict__ scores,
                              int * __restrict__ counts,
                              int batch,
                              float conf_thresh,
                              float scale,
                              int pad_x, int pad_y,
                              int orig_w, int orig_h) {
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int img_idx = blockIdx.y;

    if (det_idx >= PT_YOLO_MAX_RAW_DETS || img_idx >= batch) return;

    /* Raw detection: 6 floats per detection */
    const float *det = yolo_out + img_idx * PT_YOLO_MAX_RAW_DETS * 6 + det_idx * 6;

    float conf = det[4];
    int   cls  = (int)det[5];

    if (cls != PT_COCO_PERSON_CLASS || conf < conf_thresh) return;

    /* Undo letterbox: subtract padding, then divide by scale */
    float x1 = (det[0] - (float)pad_x) / scale;
    float y1 = (det[1] - (float)pad_y) / scale;
    float x2 = (det[2] - (float)pad_x) / scale;
    float y2 = (det[3] - (float)pad_y) / scale;

    /* Clamp to original image bounds */
    x1 = clamp_f(x1, 0.0f, (float)(orig_w - 1));
    y1 = clamp_f(y1, 0.0f, (float)(orig_h - 1));
    x2 = clamp_f(x2, 0.0f, (float)(orig_w - 1));
    y2 = clamp_f(y2, 0.0f, (float)(orig_h - 1));

    /* Atomic slot allocation */
    int slot = atomicAdd(&counts[img_idx], 1);
    if (slot >= PT_MAX_DETECTIONS) {
        /* Undo the increment -- we're full */
        atomicSub(&counts[img_idx], 1);
        return;
    }

    /* Write output */
    int box_base = (img_idx * PT_MAX_DETECTIONS + slot) * 4;
    boxes[box_base + 0] = x1;
    boxes[box_base + 1] = y1;
    boxes[box_base + 2] = x2;
    boxes[box_base + 3] = y2;

    scores[img_idx * PT_MAX_DETECTIONS + slot] = conf;
}


/* =========================================================================
 * Kernel 4: Crop + resize + normalize for VitPose
 * =========================================================================
 *
 * For each detection, expands the bounding box by 1.25x, crops from the
 * source BGR8 image, resizes to 192x256 via bilinear interpolation, converts
 * BGR->RGB, and applies ImageNet normalization.
 *
 * Also stores the 2x3 affine transform matrix (crop-space -> image-space)
 * for inverse mapping during heatmap decode.
 *
 * Grid:  (ceil(192/16), ceil(256/16), num_det), Block: (16, 16)
 */

__global__
void kernel_crop_resize_normalize_vitpose(const uint8_t * __restrict__ src_bgr,
                                          int src_w, int src_h,
                                          const float * __restrict__ boxes,
                                          int num_det,
                                          float * __restrict__ dst,
                                          float * __restrict__ affine) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    int det_idx = blockIdx.z;

    if (dx >= PT_VITPOSE_INPUT_W || dy >= PT_VITPOSE_INPUT_H || det_idx >= num_det)
        return;

    /* Read bounding box */
    float x1 = boxes[det_idx * 4 + 0];
    float y1 = boxes[det_idx * 4 + 1];
    float x2 = boxes[det_idx * 4 + 2];
    float y2 = boxes[det_idx * 4 + 3];

    /* Box center and size */
    float cx = (x1 + x2) * 0.5f;
    float cy = (y1 + y2) * 0.5f;
    float bw = x2 - x1;
    float bh = y2 - y1;

    /* Expand by PT_VITPOSE_BOX_PAD (1.25x) */
    bw *= PT_VITPOSE_BOX_PAD;
    bh *= PT_VITPOSE_BOX_PAD;

    /* Enforce aspect ratio 192:256 = 3:4.  The crop should match the model's
     * aspect ratio so the resize is uniform.  Expand the shorter dimension. */
    float aspect = (float)PT_VITPOSE_INPUT_W / (float)PT_VITPOSE_INPUT_H;  /* 0.75 */
    if (bw / bh < aspect) {
        bw = bh * aspect;
    } else {
        bh = bw / aspect;
    }

    /* Affine: maps from VitPose input coords to source image coords.
     *   src_x = scale_x * dst_x + cx - bw/2
     *   src_y = scale_y * dst_y + cy - bh/2
     *
     * Affine matrix (2x3):  [ scale_x,  0,  tx ]
     *                        [  0,  scale_y, ty ]
     */
    float scale_x = bw / (float)PT_VITPOSE_INPUT_W;
    float scale_y = bh / (float)PT_VITPOSE_INPUT_H;
    float tx = cx - bw * 0.5f;
    float ty = cy - bh * 0.5f;

    /* Write affine matrix once (first thread for this detection) */
    if (dx == 0 && dy == 0) {
        float *a = affine + det_idx * 6;
        a[0] = scale_x;  a[1] = 0.0f;    a[2] = tx;
        a[3] = 0.0f;     a[4] = scale_y;  a[5] = ty;
    }

    /* Map output pixel (dx, dy) back to source image coordinates */
    float sx = scale_x * ((float)dx + 0.5f) + tx;
    float sy = scale_y * ((float)dy + 0.5f) + ty;

    int src_stride = src_w * 3;

    /* Bilinear sample BGR from source */
    float b_raw = bilinear_sample_u8(src_bgr, src_w, src_h, src_stride, sx, sy, 0);
    float g_raw = bilinear_sample_u8(src_bgr, src_w, src_h, src_stride, sx, sy, 1);
    float r_raw = bilinear_sample_u8(src_bgr, src_w, src_h, src_stride, sx, sy, 2);

    /* ImageNet normalization: (pixel/255 - mean) / std, in RGB order */
    float r_norm = (r_raw / 255.0f - PT_IMAGENET_MEAN_R) / PT_IMAGENET_STD_R;
    float g_norm = (g_raw / 255.0f - PT_IMAGENET_MEAN_G) / PT_IMAGENET_STD_G;
    float b_norm = (b_raw / 255.0f - PT_IMAGENET_MEAN_B) / PT_IMAGENET_STD_B;

    /* Write as CHW RGB: (det_idx, C, H, W) */
    int plane_size = PT_VITPOSE_INPUT_H * PT_VITPOSE_INPUT_W;
    int det_offset = det_idx * 3 * plane_size;
    int pixel_idx  = dy * PT_VITPOSE_INPUT_W + dx;

    dst[det_offset + 0 * plane_size + pixel_idx] = r_norm;
    dst[det_offset + 1 * plane_size + pixel_idx] = g_norm;
    dst[det_offset + 2 * plane_size + pixel_idx] = b_norm;
}


/* =========================================================================
 * Kernel 5: Heatmap decode with DARK refinement
 * =========================================================================
 *
 * Input:  heatmaps (num_persons, 52, 64, 48) fp32
 *         affine   (num_persons, 2, 3) fp32
 *
 * Output: keypoints (num_persons, 52, 3) fp32 [x_image, y_image, confidence]
 *
 * Per person, per keypoint:
 *   1. Find argmax in the 64x48 heatmap
 *   2. DARK refinement: gradient + hessian from 3x3 neighborhood, shift peak
 *   3. Map from heatmap coords to VitPose input coords (scale by 4)
 *   4. Apply inverse affine to get original image coordinates
 *
 * Grid: (ceil(52/32), num_persons), Block: (32, 1)
 * Each thread handles one keypoint.
 */

__global__
void kernel_heatmap_decode_dark(const float * __restrict__ heatmaps,
                                const float * __restrict__ affine,
                                float * __restrict__ keypoints_out,
                                int num_persons) {
    int kp_idx     = blockIdx.x * blockDim.x + threadIdx.x;
    int person_idx = blockIdx.y;

    if (kp_idx >= PT_NUM_KEYPOINTS || person_idx >= num_persons) return;

    /* Heatmap for this person/keypoint: (64, 48) */
    const int hm_w = PT_VITPOSE_HEATMAP_W;  /* 48 */
    const int hm_h = PT_VITPOSE_HEATMAP_H;  /* 64 */
    const int hm_size = hm_w * hm_h;

    const float *hm = heatmaps
                     + person_idx * PT_NUM_KEYPOINTS * hm_size
                     + kp_idx * hm_size;

    /* Step 1: Argmax over the 64x48 heatmap */
    float max_val = -1e30f;
    int   max_idx = 0;

    for (int i = 0; i < hm_size; i++) {
        float v = hm[i];
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }

    int px = max_idx % hm_w;  /* column (x) */
    int py = max_idx / hm_w;  /* row    (y) */

    float confidence = max_val;

    /* Subpixel refinement via DARK (Distribution-Aware coordinate Representation
     * of Keypoint).  Compute gradient and Hessian diagonal from the 3x3
     * neighborhood, then shift: offset = -0.5 * grad / hessian_diag.
     * Only apply if the peak is not on the boundary. */
    float sub_x = (float)px;
    float sub_y = (float)py;

    if (px > 0 && px < hm_w - 1 && py > 0 && py < hm_h - 1) {
        /* Gradient */
        float dx = (hm[py * hm_w + (px + 1)] - hm[py * hm_w + (px - 1)]) * 0.5f;
        float dy = (hm[(py + 1) * hm_w + px] - hm[(py - 1) * hm_w + px]) * 0.5f;

        /* Hessian diagonal (second partial derivatives) */
        float dxx = hm[py * hm_w + (px + 1)]
                   + hm[py * hm_w + (px - 1)]
                   - 2.0f * max_val;
        float dyy = hm[(py + 1) * hm_w + px]
                   + hm[(py - 1) * hm_w + px]
                   - 2.0f * max_val;

        /* Shift: -0.5 * grad / hessian_diag.  Guard against near-zero Hessian. */
        if (fabsf(dxx) > 1e-6f) {
            sub_x -= 0.5f * dx / dxx;
        }
        if (fabsf(dyy) > 1e-6f) {
            sub_y -= 0.5f * dy / dyy;
        }

        /* Clamp refined position to heatmap bounds */
        sub_x = clamp_f(sub_x, 0.0f, (float)(hm_w - 1));
        sub_y = clamp_f(sub_y, 0.0f, (float)(hm_h - 1));
    }

    /* Step 3: Map from heatmap coords to VitPose input coords.
     * The heatmap is 4x downsampled from the input (192/48=4, 256/64=4).
     * Add 0.5 for pixel center, multiply by stride, subtract 0.5. */
    float vp_x = sub_x * 4.0f + 1.5f;  /* (sub_x + 0.5) * 4.0 - 0.5 */
    float vp_y = sub_y * 4.0f + 1.5f;

    /* Step 4: Apply inverse affine to map from VitPose input coords to
     * original image coords.
     *
     * Forward affine (stored):  src = A * dst + t
     *   img_x = scale_x * vp_x + tx
     *   img_y = scale_y * vp_y + ty
     *
     * The affine we stored maps FROM VitPose coords TO image coords
     * (it was defined as the crop-to-image transform), so we apply it
     * directly. */
    const float *a = affine + person_idx * 6;
    float a00 = a[0];  /* scale_x */
    float a01 = a[1];  /* 0 (no rotation) */
    float a02 = a[2];  /* tx */
    float a10 = a[3];  /* 0 */
    float a11 = a[4];  /* scale_y */
    float a12 = a[5];  /* ty */

    float img_x = a00 * vp_x + a01 * vp_y + a02;
    float img_y = a10 * vp_x + a11 * vp_y + a12;

    /* Write output: (person_idx, kp_idx, 3) */
    float *out = keypoints_out + (person_idx * PT_NUM_KEYPOINTS + kp_idx) * 3;
    out[0] = img_x;
    out[1] = img_y;
    out[2] = confidence;
}


/* =========================================================================
 * Launch wrappers (extern "C")
 * ========================================================================= */

extern "C"
void pt_launch_nv12_to_bgr(uint8_t *nv12,
                            uint8_t *bgr,
                            int w, int h,
                            int nv12_stride,
                            int bgr_stride,
                            cudaStream_t stream) {
    dim3 block(32, 2);
    dim3 grid((w + block.x - 1) / block.x,
              (h + block.y - 1) / block.y);

    kernel_nv12_to_bgr<<<grid, block, 0, stream>>>(
        nv12, bgr, w, h, nv12_stride, bgr_stride);
}

extern "C"
void pt_launch_letterbox_batch(uint8_t **src_ptrs,
                               void *dst_fp16,
                               int num_images,
                               int src_w, int src_h,
                               int dst_w, int dst_h,
                               cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y,
              num_images);

    kernel_letterbox_resize_normalize<<<grid, block, 0, stream>>>(
        src_ptrs, (__half *)dst_fp16,
        num_images, src_w, src_h, dst_w, dst_h);
}

extern "C"
void pt_launch_filter_detections(float *yolo_out,
                                 float *boxes,
                                 float *scores,
                                 int *counts,
                                 int batch,
                                 float conf_thresh,
                                 float scale,
                                 int pad_x, int pad_y,
                                 int orig_w, int orig_h,
                                 cudaStream_t stream) {
    /* Zero the counts before launching */
    cudaMemsetAsync(counts, 0, batch * sizeof(int), stream);

    dim3 block(256);
    dim3 grid((PT_YOLO_MAX_RAW_DETS + block.x - 1) / block.x, batch);

    kernel_filter_detections<<<grid, block, 0, stream>>>(
        yolo_out, boxes, scores, counts,
        batch, conf_thresh, scale, pad_x, pad_y, orig_w, orig_h);
}

extern "C"
void pt_launch_crop_normalize_vitpose(uint8_t *src_bgr,
                                      int src_w, int src_h,
                                      float *boxes,
                                      int num_det,
                                      float *dst,
                                      float *affine,
                                      cudaStream_t stream) {
    if (num_det <= 0) return;

    dim3 block(16, 16);
    dim3 grid((PT_VITPOSE_INPUT_W + block.x - 1) / block.x,
              (PT_VITPOSE_INPUT_H + block.y - 1) / block.y,
              num_det);

    kernel_crop_resize_normalize_vitpose<<<grid, block, 0, stream>>>(
        src_bgr, src_w, src_h, boxes, num_det, dst, affine);
}

extern "C"
void pt_launch_heatmap_decode(float *heatmaps,
                              float *affine,
                              float *keypoints_out,
                              int num_persons,
                              cudaStream_t stream) {
    if (num_persons <= 0) return;

    dim3 block(32);
    dim3 grid((PT_NUM_KEYPOINTS + block.x - 1) / block.x, num_persons);

    kernel_heatmap_decode_dark<<<grid, block, 0, stream>>>(
        heatmaps, affine, keypoints_out, num_persons);
}
