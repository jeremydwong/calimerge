/*
 * pt_kernels.h - CUDA kernel launch wrapper declarations.
 *
 * Five preprocessing/postprocessing kernels for the pose tracking pipeline:
 *   1. NV12 -> BGR8 colorspace conversion
 *   2. Letterbox resize + normalize for YOLO  (BGR8 -> fp16 CHW RGB)
 *   3. Filter YOLO detections  (threshold + undo letterbox)
 *   4. Crop + resize + normalize for VitPose  (BGR8 -> fp32 CHW RGB)
 *   5. Heatmap decode with DARK refinement  (heatmap -> keypoints)
 *
 * All functions are extern "C" for C linkage.  All launches are async
 * on the provided CUDA stream.
 */

#ifndef PT_KERNELS_H
#define PT_KERNELS_H

#include "pt_common.h"
#include <stdint.h>

/* Forward-declare cudaStream_t so callers don't need cuda_runtime.h in pure-C
 * translation units.  The actual type is CUstream_st*. */
#ifndef __CUDA_RUNTIME_H__
typedef struct CUstream_st *cudaStream_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 1. NV12 -> BGR8 conversion (single frame).
 *
 * nv12        - source NV12 data (Y plane followed by interleaved UV plane)
 * bgr         - destination BGR8 buffer (w * h * 3)
 * w, h        - frame dimensions (width must be even)
 * nv12_stride - row stride for the NV12 Y plane in bytes
 * bgr_stride  - row stride for the BGR output in bytes (typically w * 3)
 * stream      - CUDA stream for async execution
 */
void pt_launch_nv12_to_bgr(uint8_t *nv12,
                            uint8_t *bgr,
                            int w, int h,
                            int nv12_stride,
                            int bgr_stride,
                            cudaStream_t stream);

/*
 * 2. Letterbox resize + normalize for YOLO (batched).
 *
 * Resizes each source BGR8 image into a 640x640 fp16 CHW RGB tensor with
 * letterbox padding (gray=114) and /255 normalization.
 *
 * src_ptrs    - array of num_images device pointers to BGR8 source images
 * dst_fp16    - destination __half buffer: (num_images, 3, 640, 640)
 * num_images  - number of images in the batch
 * src_w,src_h - source image dimensions (all images share the same size)
 * dst_w,dst_h - destination dimensions (640, 640)
 * stream      - CUDA stream
 */
void pt_launch_letterbox_batch(uint8_t **src_ptrs,
                               void *dst_fp16,
                               int num_images,
                               int src_w, int src_h,
                               int dst_w, int dst_h,
                               cudaStream_t stream);

/*
 * 3. Filter YOLO detections.
 *
 * Filters raw YOLO output for class=0 (person) above a confidence threshold,
 * and transforms bounding box coordinates from letterboxed 640x640 space
 * back to original image coordinates.
 *
 * yolo_out    - raw YOLO output: (batch, 300, 6) where 6=[x1,y1,x2,y2,conf,cls]
 * boxes       - output filtered boxes: (batch, MAX_DET, 4) in original coords
 * scores      - output confidence scores: (batch, MAX_DET)
 * counts      - output detection count per image: (batch,)
 * batch       - number of images
 * conf_thresh - minimum confidence threshold
 * scale       - letterbox scale factor (applied to boxes)
 * pad_x,pad_y - letterbox padding offsets in pixels
 * orig_w,orig_h - original image dimensions (for clamping)
 * stream      - CUDA stream
 */
void pt_launch_filter_detections(float *yolo_out,
                                 float *boxes,
                                 float *scores,
                                 int *counts,
                                 int batch,
                                 float conf_thresh,
                                 float scale,
                                 int pad_x, int pad_y,
                                 int orig_w, int orig_h,
                                 cudaStream_t stream);

/*
 * 4. Crop + resize + normalize for VitPose (batched across detections).
 *
 * For each detected bounding box, crops and resizes to 192x256 with 1.25x
 * box expansion, applies ImageNet normalization, and stores the 2x3 affine
 * transform matrix for later inverse mapping.
 *
 * src_bgr     - source BGR8 image (all crops come from the same source)
 * src_w,src_h - source image dimensions
 * boxes       - detection boxes (num_det, 4) [x1,y1,x2,y2] in original coords
 * num_det     - number of detections to crop
 * dst         - output fp32 buffer: (num_det, 3, 256, 192) CHW RGB
 * affine      - output affine matrices: (num_det, 2, 3) fp32
 * stream      - CUDA stream
 */
void pt_launch_crop_normalize_vitpose(uint8_t *src_bgr,
                                      int src_w, int src_h,
                                      float *boxes,
                                      int num_det,
                                      float *dst,
                                      float *affine,
                                      cudaStream_t stream);

/*
 * 5. Heatmap decode with DARK refinement.
 *
 * For each person, decodes 52 keypoints from 64x48 heatmaps using argmax
 * followed by DARK sub-pixel refinement (gradient/hessian), then applies
 * the inverse affine transform to map coordinates back to original image
 * space.
 *
 * heatmaps      - input: (num_persons, 52, 64, 48) fp32
 * affine         - input: (num_persons, 2, 3) fp32 affine matrices
 * keypoints_out  - output: (num_persons, 52, 3) fp32 [x_image, y_image, conf]
 * num_persons    - total number of person crops
 * stream         - CUDA stream
 */
void pt_launch_heatmap_decode(float *heatmaps,
                              float *affine,
                              float *keypoints_out,
                              int num_persons,
                              cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* PT_KERNELS_H */
