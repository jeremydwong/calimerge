/*
 * pt_nvdec.h - Hardware video decode via NVDEC (FFmpeg hwaccel path).
 *
 * Decodes H.264 .mp4 files directly to GPU memory using NVIDIA's hardware
 * video decoder.  Falls back to CPU decode (OpenCV VideoCapture + cudaMemcpy)
 * when FFmpeg or NVDEC headers are unavailable at compile time.
 *
 * Output format: NV12 in GPU memory (Y plane w*h, UV interleaved w*h/2).
 * The caller supplies output pointers from the preallocated PT_GpuArena.
 *
 * Style: Plain C structs + free functions.  No classes, no templates, no STL.
 */

#ifndef PT_NVDEC_H
#define PT_NVDEC_H

#include "pt_common.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward-declare cudaStream_t so callers don't need cuda_runtime.h just
 * to include this header.  The actual type is struct CUstream_st*. */
#ifndef __CUDA_RUNTIME_H__
typedef struct CUstream_st *cudaStream_t;
#endif

/* ============================================================================
 * PT_VideoDecoder - Per-video decode state
 *
 * One instance per camera video file.  Each decoder is single-threaded,
 * but multiple decoders can run in parallel from different threads.
 * ============================================================================ */

typedef struct {
    /* FFmpeg demuxer state (for reading .mp4 container) */
    void *format_ctx;           /* AVFormatContext* */
    void *codec_ctx;            /* AVCodecContext*  */
    void *hw_device_ctx;        /* AVBufferRef* for CUDA hw device */
    int   video_stream_idx;

    /* Frame info */
    int width;
    int height;
    int num_frames;
    int codec_type;             /* cudaVideoCodec_H264, etc. */

    /* Current decode position */
    int      current_frame_idx;
    int      nv12_pitch;        /* stride of NV12 output */

    /* Seek state */
    int is_open;
    int is_hwaccel;             /* 1 = NVDEC hwaccel, 0 = CPU fallback */

    /* CPU fallback state (OpenCV) */
    void *cv_capture;           /* cv::VideoCapture* when using CPU fallback */

    /* File path (for error messages) */
    char path[512];

} PT_VideoDecoder;

/* ============================================================================
 * API
 * ============================================================================ */

/*
 * pt_video_open - Open a video file for hardware-accelerated decode.
 *
 * Tries NVDEC via FFmpeg h264_cuvid first; falls back to CPU decode if
 * hardware is unavailable.  Populates width, height, num_frames.
 *
 * Returns PT_OK on success.
 * Returns PT_ERR_FILE_NOT_FOUND if the file does not exist.
 * Returns PT_ERR_NVDEC if hardware decode setup fails (falls back to CPU).
 * Returns PT_ERR_DECODE if neither hardware nor CPU decode can open the file.
 */
int pt_video_open(PT_VideoDecoder *dec, const char *path);

/*
 * pt_video_get_frame_count - Return total number of frames in the video.
 *
 * Returns -1 if the decoder is not open.
 */
int pt_video_get_frame_count(PT_VideoDecoder *dec);

/*
 * pt_video_get_dimensions - Query video width and height.
 *
 * Returns PT_OK on success, PT_ERR_NOT_INITIALIZED if decoder is not open.
 */
int pt_video_get_dimensions(PT_VideoDecoder *dec, int *out_width, int *out_height);

/*
 * pt_video_decode_frame - Decode a single frame to GPU memory.
 *
 * For sequential access (frame_index == current_frame_idx + 1): fast path,
 * just decodes the next frame without seeking.
 *
 * For random access: seeks to nearest keyframe, then decodes forward.
 *
 * Output is NV12 format in out_nv12_gpu, which must point into the
 * preallocated arena buffer.  Size required: width * height * 3/2 bytes.
 *
 * Returns PT_OK on success.
 * Returns PT_ERR_DECODE on decode failure.
 * Returns PT_ERR_NOT_INITIALIZED if decoder is not open.
 */
int pt_video_decode_frame(PT_VideoDecoder *dec,
                          int frame_index,
                          uint8_t *out_nv12_gpu,
                          cudaStream_t stream);

/*
 * pt_video_decode_batch - Decode one frame from each of N decoders.
 *
 * Processes one frame per camera for the given frame indices.  This is the
 * main entry point for the pipeline: at each sync index, we decode one
 * frame from each camera video.
 *
 * decoders[]       - array of PT_VideoDecoder, one per camera
 * num_decoders     - number of cameras / decoders
 * frame_indices[]  - frame index to decode from each camera (-1 = skip)
 * out_nv12_gpu[]   - output pointers into arena, one per camera
 * stream           - CUDA stream for async operations
 *
 * Returns PT_OK if all requested frames decoded successfully.
 * Returns PT_ERR_DECODE if any frame failed (partial results are valid).
 */
int pt_video_decode_batch(PT_VideoDecoder decoders[],
                          int num_decoders,
                          const int frame_indices[],
                          uint8_t *out_nv12_gpu[],
                          cudaStream_t stream);

/*
 * pt_video_close - Release all resources held by a decoder.
 *
 * Safe to call on a zeroed / already-closed decoder (no-op).
 */
void pt_video_close(PT_VideoDecoder *dec);

#ifdef __cplusplus
}
#endif

#endif /* PT_NVDEC_H */
