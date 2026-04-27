/*
 * pt_preprocess.h - Image preprocessing for YOLO and VitPose on macOS.
 *
 * macOS equivalent of pt_kernels.h kernels 2 and 4:
 *   - Letterbox resize + normalize for YOLO  (BGR8 -> fp32 CHW RGB 640x640)
 *   - Crop + resize + normalize for VitPose  (BGR8 -> fp32 CHW RGB 256x192)
 *   - Filter YOLO detections  (threshold + undo letterbox)
 *
 * Uses Accelerate framework (vImage) for image operations on CPU.
 * On Apple Silicon unified memory, vImage operates on the same physical
 * memory as CoreML, so there is no copy overhead.
 *
 * Style: Plain C API (extern "C").
 */

#ifndef PT_PREPROCESS_H
#define PT_PREPROCESS_H

#include "../pt_shared/pt_common.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Letterbox info (returned by letterbox, consumed by filter_detections)
 * ============================================================================ */

typedef struct {
    float scale;    /* min(dst_w/src_w, dst_h/src_h) */
    int   pad_x;    /* horizontal padding (pixels) */
    int   pad_y;    /* vertical padding (pixels) */
} PT_LetterboxInfo;

/* ============================================================================
 * YOLO preprocessing
 * ============================================================================ */

/*
 * Letterbox resize + normalize a single BGR8 image to fp32 CHW RGB.
 *
 * src_bgr  - source BGR8 image (row-major, stride = src_w * 3)
 * src_w/h  - source dimensions
 * dst      - output buffer: (3, 640, 640) float32 CHW RGB, /255 normalized
 * dst_w/h  - destination dimensions (640, 640)
 * info     - output letterbox parameters (scale, pad_x, pad_y)
 */
void pt_preprocess_letterbox(const uint8_t *src_bgr,
                             int src_w, int src_h,
                             float *dst,
                             int dst_w, int dst_h,
                             PT_LetterboxInfo *info);

/*
 * Letterbox resize a batch of images.
 *
 * src_ptrs    - array of num_images BGR8 image pointers
 * num_images  - batch size
 * src_w/h     - source dimensions (all images same size)
 * dst         - output: (num_images, 3, 640, 640) float32
 * dst_w/h     - destination dimensions
 * info        - output letterbox info (same for all images in batch)
 */
void pt_preprocess_letterbox_batch(const uint8_t **src_ptrs,
                                   int num_images,
                                   int src_w, int src_h,
                                   float *dst,
                                   int dst_w, int dst_h,
                                   PT_LetterboxInfo *info);

/* ============================================================================
 * YOLO detection filtering
 * ============================================================================ */

/*
 * Filter YOLO output for person detections and undo letterbox transform.
 *
 * yolo_out    - raw YOLO output: (batch, 300, 6) [x1,y1,x2,y2,conf,cls]
 * boxes       - output: (batch, MAX_DET, 4) in original image coords
 * scores      - output: (batch, MAX_DET) confidence scores
 * counts      - output: (batch,) detection count per image
 * batch       - number of images
 * conf_thresh - minimum confidence threshold
 * info        - letterbox parameters for coordinate transform
 * orig_w/h    - original image dimensions (for clamping)
 */
void pt_preprocess_filter_detections(const float *yolo_out,
                                     float *boxes,
                                     float *scores,
                                     int *counts,
                                     int batch,
                                     float conf_thresh,
                                     const PT_LetterboxInfo *info,
                                     int orig_w, int orig_h);

/* ============================================================================
 * VitPose preprocessing
 * ============================================================================ */

/*
 * Crop, resize, and normalize for VitPose. One detection at a time.
 *
 * src_bgr  - source BGR8 image
 * src_w/h  - source dimensions
 * box      - detection box [x1, y1, x2, y2] in original image coords
 * dst      - output: (3, 256, 192) float32 CHW RGB, ImageNet-normalized
 * affine   - output: 6 floats [scale_x, 0, tx, 0, scale_y, ty]
 *            maps from VitPose input coords to original image coords
 */
void pt_preprocess_crop_vitpose(const uint8_t *src_bgr,
                                int src_w, int src_h,
                                const float box[4],
                                float *dst,
                                float affine[6]);

/*
 * Crop + normalize a batch of detections from the same source image.
 *
 * src_bgr  - source BGR8 image
 * src_w/h  - source dimensions
 * boxes    - (num_det, 4) detection boxes
 * num_det  - number of detections
 * dst      - output: (num_det, 3, 256, 192) float32
 * affines  - output: (num_det, 6) affine matrices
 */
void pt_preprocess_crop_vitpose_batch(const uint8_t *src_bgr,
                                      int src_w, int src_h,
                                      const float *boxes,
                                      int num_det,
                                      float *dst,
                                      float *affines);

#ifdef __cplusplus
}
#endif

#endif /* PT_PREPROCESS_H */
