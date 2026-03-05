/*
 * pt_nvdec.cpp - Hardware video decode via NVDEC (FFmpeg hwaccel path).
 *
 * Primary path (Option B from design doc):
 *   FFmpeg h264_cuvid decoder -> av_hwframe_transfer_data -> NV12 in GPU memory.
 *   Requires FFmpeg built with --enable-cuvid --enable-cuda.
 *
 * Fallback path:
 *   OpenCV VideoCapture (CPU) -> BGR frame -> cudaMemcpy to GPU.
 *   The BGR-to-NV12 conversion is skipped; we upload BGR directly and the
 *   downstream NV12->BGR kernel handles the identity case.  (In practice
 *   the fallback writes a synthetic NV12 frame from the BGR data.)
 *
 * Compile-time feature detection:
 *   - If LIBAVFORMAT headers are found: full FFmpeg + NVDEC path.
 *   - Otherwise: OpenCV CPU fallback only.
 *   - CUDA runtime is always required (for cudaMemcpy).
 *
 * Style: Plain C functions with extern "C" linkage.  No classes, no STL.
 */

#include "pt_nvdec.h"

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

/* --------------------------------------------------------------------------
 * Compile-time feature detection
 * -------------------------------------------------------------------------- */

/* Try to detect FFmpeg headers.  If they exist, we get the full hwaccel path. */
#if __has_include(<libavformat/avformat.h>) && __has_include(<libavcodec/avcodec.h>)
    #define PT_HAVE_FFMPEG 1
#else
    #define PT_HAVE_FFMPEG 0
#endif

#if PT_HAVE_FFMPEG
    extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/avutil.h>
    #include <libavutil/imgutils.h>
    #include <libavutil/hwcontext.h>
    #include <libavutil/pixdesc.h>
    }

    /* CUDA hwcontext may or may not be available even with FFmpeg */
    #if __has_include(<libavutil/hwcontext_cuda.h>)
        extern "C" {
        #include <libavutil/hwcontext_cuda.h>
        }
        #define PT_HAVE_FFMPEG_CUDA 1
    #else
        #define PT_HAVE_FFMPEG_CUDA 0
    #endif
#else
    #define PT_HAVE_FFMPEG_CUDA 0
#endif

/* CUDA runtime -- always required */
#include <cuda_runtime.h>

/* OpenCV -- used for CPU fallback decode */
#if __has_include(<opencv2/videoio.hpp>)
    #include <opencv2/videoio.hpp>
    #include <opencv2/imgproc.hpp>
    #define PT_HAVE_OPENCV 1
#else
    #define PT_HAVE_OPENCV 0
#endif

/* --------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------- */

static void pt_log_error(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[pt_nvdec] %s: ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

/* Check if a file exists (simple fopen test). */
static int pt_file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

/* ============================================================================
 * FFmpeg + NVDEC path
 * ============================================================================ */

#if PT_HAVE_FFMPEG

/*
 * find_hw_config - Search for a CUDA hwaccel config on the given codec.
 * Returns the config if found, NULL otherwise.
 */
static const AVCodecHWConfig *find_hw_config(const AVCodec *codec) {
    for (int i = 0; ; i++) {
        const AVCodecHWConfig *config = avcodec_get_hw_config(codec, i);
        if (!config) break;
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
            config->device_type == AV_HWDEVICE_TYPE_CUDA) {
            return config;
        }
    }
    return NULL;
}

/*
 * hw_pix_fmt_callback - Tell FFmpeg which hwaccel pixel format we want.
 * Called by the codec to negotiate the output format.
 */
static enum AVPixelFormat hw_pix_fmt_callback(AVCodecContext *ctx,
                                               const enum AVPixelFormat *pix_fmts) {
    (void)ctx;
    for (const enum AVPixelFormat *p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == AV_PIX_FMT_CUDA)
            return AV_PIX_FMT_CUDA;
    }
    /* CUDA format not available; fall back to whatever FFmpeg offers. */
    return pix_fmts[0];
}

/*
 * open_ffmpeg - Open the video with FFmpeg, optionally with CUDA hwaccel.
 * Returns PT_OK on success, error code on failure.
 * Sets dec->is_hwaccel = 1 if hardware decode is active.
 */
static int open_ffmpeg(PT_VideoDecoder *dec, const char *path) {
    AVFormatContext *fmt_ctx = NULL;
    AVCodecContext *codec_ctx = NULL;
    AVBufferRef *hw_device_ctx_ref = NULL;
    int ret;

    /* Open container */
    ret = avformat_open_input(&fmt_ctx, path, NULL, NULL);
    if (ret < 0) {
        pt_log_error("open_ffmpeg", "avformat_open_input failed for '%s' (error %d)", path, ret);
        return PT_ERR_DECODE;
    }

    ret = avformat_find_stream_info(fmt_ctx, NULL);
    if (ret < 0) {
        pt_log_error("open_ffmpeg", "avformat_find_stream_info failed (error %d)", ret);
        avformat_close_input(&fmt_ctx);
        return PT_ERR_DECODE;
    }

    /* Find best video stream */
    int stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (stream_idx < 0) {
        pt_log_error("open_ffmpeg", "no video stream found in '%s'", path);
        avformat_close_input(&fmt_ctx);
        return PT_ERR_DECODE;
    }

    AVStream *stream = fmt_ctx->streams[stream_idx];
    const AVCodec *codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) {
        pt_log_error("open_ffmpeg", "no decoder found for codec id %d", stream->codecpar->codec_id);
        avformat_close_input(&fmt_ctx);
        return PT_ERR_DECODE;
    }

    /* Try to set up CUDA hwaccel */
    int hwaccel_ok = 0;
#if PT_HAVE_FFMPEG_CUDA
    const AVCodecHWConfig *hw_config = find_hw_config(codec);
    if (hw_config) {
        ret = av_hwdevice_ctx_create(&hw_device_ctx_ref, AV_HWDEVICE_TYPE_CUDA, NULL, NULL, 0);
        if (ret >= 0) {
            hwaccel_ok = 1;
        } else {
            pt_log_error("open_ffmpeg", "CUDA hw device creation failed (error %d), "
                         "will use CPU decode", ret);
        }
    }
#endif

    /* Allocate codec context */
    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        pt_log_error("open_ffmpeg", "avcodec_alloc_context3 failed");
        if (hw_device_ctx_ref) av_buffer_unref(&hw_device_ctx_ref);
        avformat_close_input(&fmt_ctx);
        return PT_ERR_DECODE;
    }

    ret = avcodec_parameters_to_context(codec_ctx, stream->codecpar);
    if (ret < 0) {
        pt_log_error("open_ffmpeg", "avcodec_parameters_to_context failed (error %d)", ret);
        avcodec_free_context(&codec_ctx);
        if (hw_device_ctx_ref) av_buffer_unref(&hw_device_ctx_ref);
        avformat_close_input(&fmt_ctx);
        return PT_ERR_DECODE;
    }

    if (hwaccel_ok) {
        codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx_ref);
        codec_ctx->get_format = hw_pix_fmt_callback;
    }

    /* Open codec */
    ret = avcodec_open2(codec_ctx, codec, NULL);
    if (ret < 0) {
        pt_log_error("open_ffmpeg", "avcodec_open2 failed (error %d)", ret);
        avcodec_free_context(&codec_ctx);
        if (hw_device_ctx_ref) av_buffer_unref(&hw_device_ctx_ref);
        avformat_close_input(&fmt_ctx);
        return PT_ERR_DECODE;
    }

    /* Populate decoder struct */
    dec->format_ctx       = fmt_ctx;
    dec->codec_ctx        = codec_ctx;
    dec->hw_device_ctx    = hw_device_ctx_ref;
    dec->video_stream_idx = stream_idx;
    dec->width            = codec_ctx->width;
    dec->height           = codec_ctx->height;
    dec->is_hwaccel       = hwaccel_ok;
    dec->current_frame_idx = -1;

    /* Frame count: try container metadata first, then duration-based estimate */
    if (stream->nb_frames > 0) {
        dec->num_frames = (int)stream->nb_frames;
    } else if (stream->duration > 0 && stream->time_base.den > 0) {
        double duration_sec = (double)stream->duration * stream->time_base.num
                            / (double)stream->time_base.den;
        double fps = 0.0;
        if (stream->avg_frame_rate.den > 0) {
            fps = (double)stream->avg_frame_rate.num / (double)stream->avg_frame_rate.den;
        } else if (stream->r_frame_rate.den > 0) {
            fps = (double)stream->r_frame_rate.num / (double)stream->r_frame_rate.den;
        }
        if (fps > 0.0) {
            dec->num_frames = (int)(duration_sec * fps + 0.5);
        } else {
            dec->num_frames = 0;  /* unknown */
        }
    } else {
        dec->num_frames = 0;  /* unknown */
    }

    dec->is_open = 1;
    dec->cv_capture = NULL;

    fprintf(stderr, "[pt_nvdec] Opened '%s': %dx%d, %d frames, %s\n",
            path, dec->width, dec->height, dec->num_frames,
            hwaccel_ok ? "NVDEC hwaccel" : "CPU decode (FFmpeg)");

    return PT_OK;
}

/*
 * decode_one_frame_ffmpeg - Decode a single frame via FFmpeg.
 *
 * Reads packets until one video frame is decoded.  If hwaccel is active,
 * transfers the frame from GPU to a CPU-side NV12 AVFrame, then uploads
 * the NV12 data to the caller's GPU buffer.
 *
 * If hwaccel is active and the decoded frame is already in GPU memory
 * (AV_PIX_FMT_CUDA), we use av_hwframe_transfer_data to get NV12 on CPU,
 * then cudaMemcpy2D it into the arena.  (A future optimization would be to
 * use cuMemcpy D2D to avoid the round-trip, but this requires accessing the
 * internal CUDA pointer from the AVFrame, which is fragile across FFmpeg
 * versions.)
 *
 * For CPU-decoded frames (AV_PIX_FMT_YUV420P typically), we convert to
 * NV12 format and upload.
 *
 * Returns PT_OK on success, PT_ERR_DECODE on failure or EOF.
 */
static int decode_one_frame_ffmpeg(PT_VideoDecoder *dec,
                                    uint8_t *out_nv12_gpu,
                                    cudaStream_t stream) {
    AVFormatContext *fmt_ctx = (AVFormatContext *)dec->format_ctx;
    AVCodecContext  *codec_ctx = (AVCodecContext *)dec->codec_ctx;
    AVPacket *pkt = av_packet_alloc();
    AVFrame  *frame = av_frame_alloc();
    AVFrame  *sw_frame = av_frame_alloc();  /* for hwaccel transfer */
    int got_frame = 0;
    int ret;

    if (!pkt || !frame || !sw_frame) {
        pt_log_error("decode_one_frame_ffmpeg", "failed to allocate packet/frame");
        av_packet_free(&pkt);
        av_frame_free(&frame);
        av_frame_free(&sw_frame);
        return PT_ERR_DECODE;
    }

    /* Read packets until we get a decoded frame */
    while (!got_frame) {
        ret = av_read_frame(fmt_ctx, pkt);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                /* Flush the decoder */
                avcodec_send_packet(codec_ctx, NULL);
                ret = avcodec_receive_frame(codec_ctx, frame);
                if (ret == 0) {
                    got_frame = 1;
                    break;
                }
            }
            /* True end of file or error */
            av_packet_free(&pkt);
            av_frame_free(&frame);
            av_frame_free(&sw_frame);
            return PT_ERR_DECODE;
        }

        if (pkt->stream_index != dec->video_stream_idx) {
            av_packet_unref(pkt);
            continue;
        }

        ret = avcodec_send_packet(codec_ctx, pkt);
        av_packet_unref(pkt);

        if (ret < 0 && ret != AVERROR(EAGAIN)) {
            pt_log_error("decode_one_frame_ffmpeg", "avcodec_send_packet error %d", ret);
            continue;  /* try next packet */
        }

        ret = avcodec_receive_frame(codec_ctx, frame);
        if (ret == 0) {
            got_frame = 1;
        } else if (ret == AVERROR(EAGAIN)) {
            continue;  /* need more packets */
        } else {
            pt_log_error("decode_one_frame_ffmpeg", "avcodec_receive_frame error %d", ret);
            av_packet_free(&pkt);
            av_frame_free(&frame);
            av_frame_free(&sw_frame);
            return PT_ERR_DECODE;
        }
    }

    /*
     * Now `frame` holds the decoded frame.  Get NV12 data into the arena.
     *
     * Case 1: hwaccel (frame->format == AV_PIX_FMT_CUDA)
     *   -> av_hwframe_transfer_data to sw_frame (NV12 on CPU)
     *   -> cudaMemcpy2DAsync to out_nv12_gpu
     *
     * Case 2: CPU decode (typically YUV420P)
     *   -> manually interleave U/V planes to create NV12
     *   -> cudaMemcpyAsync to out_nv12_gpu
     */

    int w = dec->width;
    int h = dec->height;
    int nv12_size = w * h + w * h / 2;  /* Y + UV */

    if (frame->format == AV_PIX_FMT_CUDA) {
        /* Hardware-decoded frame: transfer to CPU NV12 */
        sw_frame->format = AV_PIX_FMT_NV12;
        ret = av_hwframe_transfer_data(sw_frame, frame, 0);
        if (ret < 0) {
            pt_log_error("decode_one_frame_ffmpeg",
                         "av_hwframe_transfer_data failed (error %d)", ret);
            av_frame_free(&frame);
            av_frame_free(&sw_frame);
            av_packet_free(&pkt);
            return PT_ERR_DECODE;
        }

        /*
         * sw_frame is NV12: data[0] = Y plane, data[1] = UV plane.
         * Upload each plane to the contiguous GPU buffer.
         */
        int y_pitch  = sw_frame->linesize[0];
        int uv_pitch = sw_frame->linesize[1];
        dec->nv12_pitch = w;

        /* Y plane: copy row-by-row if pitch != width (padding) */
        if (y_pitch == w) {
            cudaMemcpyAsync(out_nv12_gpu, sw_frame->data[0],
                            (size_t)w * h, cudaMemcpyHostToDevice, stream);
        } else {
            cudaMemcpy2DAsync(out_nv12_gpu, (size_t)w,
                              sw_frame->data[0], (size_t)y_pitch,
                              (size_t)w, (size_t)h,
                              cudaMemcpyHostToDevice, stream);
        }

        /* UV plane: h/2 rows of w bytes (interleaved U,V) */
        int uv_h = h / 2;
        if (uv_pitch == w) {
            cudaMemcpyAsync(out_nv12_gpu + w * h, sw_frame->data[1],
                            (size_t)w * uv_h, cudaMemcpyHostToDevice, stream);
        } else {
            cudaMemcpy2DAsync(out_nv12_gpu + w * h, (size_t)w,
                              sw_frame->data[1], (size_t)uv_pitch,
                              (size_t)w, (size_t)uv_h,
                              cudaMemcpyHostToDevice, stream);
        }

    } else if (frame->format == AV_PIX_FMT_NV12) {
        /* CPU-decoded NV12 (some decoders output this directly) */
        int y_pitch  = frame->linesize[0];
        int uv_pitch = frame->linesize[1];
        dec->nv12_pitch = w;

        if (y_pitch == w) {
            cudaMemcpyAsync(out_nv12_gpu, frame->data[0],
                            (size_t)w * h, cudaMemcpyHostToDevice, stream);
        } else {
            cudaMemcpy2DAsync(out_nv12_gpu, (size_t)w,
                              frame->data[0], (size_t)y_pitch,
                              (size_t)w, (size_t)h,
                              cudaMemcpyHostToDevice, stream);
        }

        int uv_h = h / 2;
        if (uv_pitch == w) {
            cudaMemcpyAsync(out_nv12_gpu + w * h, frame->data[1],
                            (size_t)w * uv_h, cudaMemcpyHostToDevice, stream);
        } else {
            cudaMemcpy2DAsync(out_nv12_gpu + w * h, (size_t)w,
                              frame->data[1], (size_t)uv_pitch,
                              (size_t)w, (size_t)uv_h,
                              cudaMemcpyHostToDevice, stream);
        }

    } else if (frame->format == AV_PIX_FMT_YUV420P) {
        /*
         * CPU-decoded YUV420P: three separate planes Y, U, V.
         * Convert to NV12 by interleaving U and V into a single UV plane.
         *
         * We allocate a temporary CPU buffer for the NV12 data, build it,
         * then upload in one shot.
         */
        uint8_t *nv12_buf = (uint8_t *)malloc((size_t)nv12_size);
        if (!nv12_buf) {
            pt_log_error("decode_one_frame_ffmpeg", "malloc failed for NV12 conversion");
            av_frame_free(&frame);
            av_frame_free(&sw_frame);
            av_packet_free(&pkt);
            return PT_ERR_DECODE;
        }

        /* Copy Y plane */
        int y_pitch = frame->linesize[0];
        for (int row = 0; row < h; row++) {
            memcpy(nv12_buf + row * w, frame->data[0] + row * y_pitch, (size_t)w);
        }

        /* Interleave U and V into UV plane */
        int u_pitch = frame->linesize[1];
        int v_pitch = frame->linesize[2];
        int uv_h = h / 2;
        int uv_w = w / 2;
        uint8_t *uv_dst = nv12_buf + w * h;

        for (int row = 0; row < uv_h; row++) {
            const uint8_t *u_row = frame->data[1] + row * u_pitch;
            const uint8_t *v_row = frame->data[2] + row * v_pitch;
            uint8_t *dst = uv_dst + row * w;
            for (int col = 0; col < uv_w; col++) {
                dst[col * 2 + 0] = u_row[col];
                dst[col * 2 + 1] = v_row[col];
            }
        }

        dec->nv12_pitch = w;
        cudaMemcpyAsync(out_nv12_gpu, nv12_buf, (size_t)nv12_size,
                         cudaMemcpyHostToDevice, stream);

        /* Must sync before freeing the CPU buffer */
        cudaStreamSynchronize(stream);
        free(nv12_buf);

    } else {
        /*
         * Unexpected pixel format.  Log it and fail gracefully.
         */
        pt_log_error("decode_one_frame_ffmpeg",
                      "unsupported pixel format %d (%s)",
                      frame->format,
                      av_get_pix_fmt_name((AVPixelFormat)frame->format));
        av_frame_free(&frame);
        av_frame_free(&sw_frame);
        av_packet_free(&pkt);
        return PT_ERR_DECODE;
    }

    dec->current_frame_idx++;

    av_frame_free(&frame);
    av_frame_free(&sw_frame);
    av_packet_free(&pkt);
    return PT_OK;
}

/*
 * seek_ffmpeg - Seek to a target frame index.
 *
 * Seeks to the nearest keyframe at or before target_frame, then decodes
 * forward to the exact frame.  Updates dec->current_frame_idx.
 */
static int seek_ffmpeg(PT_VideoDecoder *dec, int target_frame) {
    AVFormatContext *fmt_ctx = (AVFormatContext *)dec->format_ctx;
    AVCodecContext  *codec_ctx = (AVCodecContext *)dec->codec_ctx;
    AVStream *stream = fmt_ctx->streams[dec->video_stream_idx];

    /* Convert frame index to stream timestamp */
    int64_t target_ts;
    if (stream->avg_frame_rate.num > 0 && stream->avg_frame_rate.den > 0 &&
        stream->time_base.num > 0 && stream->time_base.den > 0) {
        /* ts = frame_index * (time_base_den / fps) */
        double fps = (double)stream->avg_frame_rate.num / (double)stream->avg_frame_rate.den;
        double time_sec = (double)target_frame / fps;
        target_ts = (int64_t)(time_sec * stream->time_base.den / stream->time_base.num);
    } else {
        /* Fallback: assume 30 fps */
        double time_sec = (double)target_frame / 30.0;
        target_ts = (int64_t)(time_sec * stream->time_base.den / stream->time_base.num);
    }

    int ret = av_seek_frame(fmt_ctx, dec->video_stream_idx, target_ts, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        pt_log_error("seek_ffmpeg", "av_seek_frame failed (error %d)", ret);
        return PT_ERR_DECODE;
    }

    /* Flush codec buffers after seek */
    avcodec_flush_buffers(codec_ctx);

    /*
     * We don't know exactly which frame the seek landed on.
     * We'll set current_frame_idx to target_frame - 1 so the caller's
     * subsequent sequential decode loop works correctly.
     *
     * For the keyframe-to-target decode loop, we need to discard frames
     * until we reach the target.  But since we don't have reliable PTS-to-
     * frame-index mapping in all containers, we estimate:
     *
     * Heuristic: seek typically lands on a keyframe before target.
     * We decode frames discarding them until we've decoded enough to reach
     * target_frame.  We track this via a simple counter.
     *
     * This is imperfect but sufficient for the common case where videos
     * have regular keyframe intervals (every 1-2 seconds).
     */

    /* Decode and discard frames until we reach the target.
     * We use the PTS to determine frame index where possible. */
    AVPacket *pkt = av_packet_alloc();
    AVFrame  *frame = av_frame_alloc();
    AVFrame  *sw_frame = av_frame_alloc();

    if (!pkt || !frame || !sw_frame) {
        av_packet_free(&pkt);
        av_frame_free(&frame);
        av_frame_free(&sw_frame);
        return PT_ERR_DECODE;
    }

    int decoded_count = 0;
    int max_discard = target_frame + 1;  /* absolute max to prevent infinite loop */
    int current_pts_frame = -1;

    while (decoded_count < max_discard) {
        ret = av_read_frame(fmt_ctx, pkt);
        if (ret < 0) break;

        if (pkt->stream_index != dec->video_stream_idx) {
            av_packet_unref(pkt);
            continue;
        }

        ret = avcodec_send_packet(codec_ctx, pkt);
        av_packet_unref(pkt);
        if (ret < 0 && ret != AVERROR(EAGAIN)) continue;

        while (1) {
            ret = avcodec_receive_frame(codec_ctx, frame);
            if (ret != 0) break;

            /* Estimate current frame index from PTS */
            if (frame->pts != AV_NOPTS_VALUE && stream->avg_frame_rate.num > 0) {
                double fps = (double)stream->avg_frame_rate.num
                           / (double)stream->avg_frame_rate.den;
                double time_sec = (double)frame->pts * stream->time_base.num
                                / (double)stream->time_base.den;
                current_pts_frame = (int)(time_sec * fps + 0.5);
            } else {
                current_pts_frame++;
            }

            av_frame_unref(frame);

            if (current_pts_frame >= target_frame - 1) {
                /* We've reached (or passed) one frame before target.
                 * Set state so the next decode_one_frame call returns target. */
                dec->current_frame_idx = target_frame - 1;
                av_packet_free(&pkt);
                av_frame_free(&frame);
                av_frame_free(&sw_frame);
                return PT_OK;
            }

            decoded_count++;
        }
    }

    /* If we get here, we decoded everything up to EOF without finding target.
     * This means target_frame is beyond the video. */
    dec->current_frame_idx = target_frame - 1;
    av_packet_free(&pkt);
    av_frame_free(&frame);
    av_frame_free(&sw_frame);
    return PT_OK;
}

/*
 * close_ffmpeg - Release all FFmpeg resources.
 */
static void close_ffmpeg(PT_VideoDecoder *dec) {
    if (dec->codec_ctx) {
        avcodec_free_context((AVCodecContext **)&dec->codec_ctx);
        dec->codec_ctx = NULL;
    }
    if (dec->hw_device_ctx) {
        av_buffer_unref((AVBufferRef **)&dec->hw_device_ctx);
        dec->hw_device_ctx = NULL;
    }
    if (dec->format_ctx) {
        avformat_close_input((AVFormatContext **)&dec->format_ctx);
        dec->format_ctx = NULL;
    }
}

#endif /* PT_HAVE_FFMPEG */

/* ============================================================================
 * OpenCV CPU fallback path
 * ============================================================================ */

#if PT_HAVE_OPENCV

/*
 * BGR-to-NV12 conversion on CPU.
 *
 * NV12 = Y plane (w*h bytes) + UV interleaved plane (w*h/2 bytes).
 * Standard BT.601 conversion:
 *   Y  =  0.299*R + 0.587*G + 0.114*B
 *   U  = -0.169*R - 0.331*G + 0.500*B + 128
 *   V  =  0.500*R - 0.419*G - 0.081*B + 128
 *
 * We subsample U/V by 2x2 averaging.
 */
static void bgr_to_nv12_cpu(const uint8_t *bgr, int w, int h, int bgr_stride,
                              uint8_t *nv12_y, uint8_t *nv12_uv) {
    /* Y plane */
    for (int row = 0; row < h; row++) {
        const uint8_t *src = bgr + row * bgr_stride;
        uint8_t *dst = nv12_y + row * w;
        for (int col = 0; col < w; col++) {
            int b = src[col * 3 + 0];
            int g = src[col * 3 + 1];
            int r = src[col * 3 + 2];
            int y = (( 66 * r + 129 * g +  25 * b + 128) >> 8) + 16;
            if (y < 0) y = 0; if (y > 255) y = 255;
            dst[col] = (uint8_t)y;
        }
    }

    /* UV plane (subsampled 2x2) */
    int uv_h = h / 2;
    int uv_w = w / 2;
    for (int row = 0; row < uv_h; row++) {
        const uint8_t *src0 = bgr + (row * 2 + 0) * bgr_stride;
        const uint8_t *src1 = bgr + (row * 2 + 1) * bgr_stride;
        uint8_t *dst = nv12_uv + row * w;
        for (int col = 0; col < uv_w; col++) {
            /* Average 2x2 block */
            int px = col * 2;
            int b = (src0[px*3+0] + src0[(px+1)*3+0] + src1[px*3+0] + src1[(px+1)*3+0]) / 4;
            int g = (src0[px*3+1] + src0[(px+1)*3+1] + src1[px*3+1] + src1[(px+1)*3+1]) / 4;
            int r = (src0[px*3+2] + src0[(px+1)*3+2] + src1[px*3+2] + src1[(px+1)*3+2]) / 4;
            int u = ((-38 * r -  74 * g + 112 * b + 128) >> 8) + 128;
            int v = ((112 * r -  94 * g -  18 * b + 128) >> 8) + 128;
            if (u < 0) u = 0; if (u > 255) u = 255;
            if (v < 0) v = 0; if (v > 255) v = 255;
            dst[col * 2 + 0] = (uint8_t)u;
            dst[col * 2 + 1] = (uint8_t)v;
        }
    }
}

static int open_opencv(PT_VideoDecoder *dec, const char *path) {
    cv::VideoCapture *cap = new (std::nothrow) cv::VideoCapture();
    if (!cap) {
        pt_log_error("open_opencv", "failed to allocate VideoCapture");
        return PT_ERR_DECODE;
    }

    if (!cap->open(path)) {
        pt_log_error("open_opencv", "VideoCapture failed to open '%s'", path);
        delete cap;
        return PT_ERR_DECODE;
    }

    dec->cv_capture = cap;
    dec->width      = (int)cap->get(cv::CAP_PROP_FRAME_WIDTH);
    dec->height     = (int)cap->get(cv::CAP_PROP_FRAME_HEIGHT);
    dec->num_frames = (int)cap->get(cv::CAP_PROP_FRAME_COUNT);
    dec->is_hwaccel = 0;
    dec->is_open    = 1;
    dec->current_frame_idx = -1;
    dec->nv12_pitch = dec->width;

    fprintf(stderr, "[pt_nvdec] Opened '%s': %dx%d, %d frames, CPU fallback (OpenCV)\n",
            path, dec->width, dec->height, dec->num_frames);

    return PT_OK;
}

static int decode_one_frame_opencv(PT_VideoDecoder *dec,
                                    uint8_t *out_nv12_gpu,
                                    cudaStream_t stream) {
    cv::VideoCapture *cap = (cv::VideoCapture *)dec->cv_capture;
    if (!cap) return PT_ERR_NOT_INITIALIZED;

    cv::Mat bgr_frame;
    if (!cap->read(bgr_frame)) {
        return PT_ERR_DECODE;  /* EOF or error */
    }

    int w = bgr_frame.cols;
    int h = bgr_frame.rows;
    int nv12_size = w * h + w * h / 2;

    /* Allocate temporary NV12 buffer on CPU */
    uint8_t *nv12_buf = (uint8_t *)malloc((size_t)nv12_size);
    if (!nv12_buf) {
        pt_log_error("decode_one_frame_opencv", "malloc failed for NV12 buffer");
        return PT_ERR_DECODE;
    }

    bgr_to_nv12_cpu(bgr_frame.data, w, h, (int)bgr_frame.step[0],
                     nv12_buf, nv12_buf + w * h);

    cudaMemcpyAsync(out_nv12_gpu, nv12_buf, (size_t)nv12_size,
                     cudaMemcpyHostToDevice, stream);

    /* Must sync before freeing CPU buffer */
    cudaStreamSynchronize(stream);
    free(nv12_buf);

    dec->current_frame_idx++;
    return PT_OK;
}

static int seek_opencv(PT_VideoDecoder *dec, int target_frame) {
    cv::VideoCapture *cap = (cv::VideoCapture *)dec->cv_capture;
    if (!cap) return PT_ERR_NOT_INITIALIZED;

    if (!cap->set(cv::CAP_PROP_POS_FRAMES, (double)target_frame)) {
        pt_log_error("seek_opencv", "seek to frame %d failed", target_frame);
        return PT_ERR_DECODE;
    }

    dec->current_frame_idx = target_frame - 1;
    return PT_OK;
}

static void close_opencv(PT_VideoDecoder *dec) {
    if (dec->cv_capture) {
        cv::VideoCapture *cap = (cv::VideoCapture *)dec->cv_capture;
        cap->release();
        delete cap;
        dec->cv_capture = NULL;
    }
}

#endif /* PT_HAVE_OPENCV */

/* ============================================================================
 * Public API
 * ============================================================================ */

extern "C" {

int pt_video_open(PT_VideoDecoder *dec, const char *path) {
    if (!dec || !path) return PT_ERR_INVALID_PARAM;

    /* Zero the struct */
    memset(dec, 0, sizeof(PT_VideoDecoder));
    strncpy(dec->path, path, sizeof(dec->path) - 1);
    dec->path[sizeof(dec->path) - 1] = '\0';

    /* Check file exists */
    if (!pt_file_exists(path)) {
        pt_log_error("pt_video_open", "file not found: '%s'", path);
        return PT_ERR_FILE_NOT_FOUND;
    }

#if PT_HAVE_FFMPEG
    /* Try FFmpeg (with or without hwaccel) */
    int ret = open_ffmpeg(dec, path);
    if (ret == PT_OK) return PT_OK;

    /* FFmpeg failed; try OpenCV fallback */
    pt_log_error("pt_video_open", "FFmpeg open failed, trying OpenCV fallback");
#endif

#if PT_HAVE_OPENCV
    int ret_cv = open_opencv(dec, path);
    if (ret_cv == PT_OK) return PT_OK;
#endif

    /* Nothing worked */
    pt_log_error("pt_video_open",
                 "all decode backends failed for '%s'. "
                 "Compile with FFmpeg and/or OpenCV support.", path);
    return PT_ERR_DECODE;
}

int pt_video_get_frame_count(PT_VideoDecoder *dec) {
    if (!dec || !dec->is_open) return -1;
    return dec->num_frames;
}

int pt_video_get_dimensions(PT_VideoDecoder *dec, int *out_width, int *out_height) {
    if (!dec || !dec->is_open) return PT_ERR_NOT_INITIALIZED;
    if (out_width)  *out_width  = dec->width;
    if (out_height) *out_height = dec->height;
    return PT_OK;
}

int pt_video_decode_frame(PT_VideoDecoder *dec,
                          int frame_index,
                          uint8_t *out_nv12_gpu,
                          cudaStream_t stream) {
    if (!dec || !dec->is_open) return PT_ERR_NOT_INITIALIZED;
    if (!out_nv12_gpu) return PT_ERR_INVALID_PARAM;
    if (frame_index < 0) return PT_ERR_INVALID_PARAM;

    /*
     * Sequential fast path: if requesting the next frame, just decode it.
     * Random access: seek first, then decode.
     */
    int need_seek = (frame_index != dec->current_frame_idx + 1);

    if (need_seek && frame_index != 0) {
        int seek_ret;

#if PT_HAVE_FFMPEG
        if (dec->format_ctx) {
            seek_ret = seek_ffmpeg(dec, frame_index);
        } else
#endif
#if PT_HAVE_OPENCV
        if (dec->cv_capture) {
            seek_ret = seek_opencv(dec, frame_index);
        } else
#endif
        {
            pt_log_error("pt_video_decode_frame", "no backend available for seek");
            return PT_ERR_NOT_INITIALIZED;
        }

        if (seek_ret != PT_OK) return seek_ret;
    } else if (frame_index == 0 && dec->current_frame_idx != -1) {
        /* Rewind to beginning */
#if PT_HAVE_FFMPEG
        if (dec->format_ctx) {
            int seek_ret = seek_ffmpeg(dec, 0);
            if (seek_ret != PT_OK) return seek_ret;
        } else
#endif
#if PT_HAVE_OPENCV
        if (dec->cv_capture) {
            int seek_ret = seek_opencv(dec, 0);
            if (seek_ret != PT_OK) return seek_ret;
        } else
#endif
        {
            return PT_ERR_NOT_INITIALIZED;
        }
    }

    /* Decode one frame */
#if PT_HAVE_FFMPEG
    if (dec->format_ctx) {
        return decode_one_frame_ffmpeg(dec, out_nv12_gpu, stream);
    }
#endif

#if PT_HAVE_OPENCV
    if (dec->cv_capture) {
        return decode_one_frame_opencv(dec, out_nv12_gpu, stream);
    }
#endif

    return PT_ERR_NOT_INITIALIZED;
}

int pt_video_decode_batch(PT_VideoDecoder decoders[],
                          int num_decoders,
                          const int frame_indices[],
                          uint8_t *out_nv12_gpu[],
                          cudaStream_t stream) {
    if (!decoders || !frame_indices || !out_nv12_gpu)
        return PT_ERR_INVALID_PARAM;
    if (num_decoders <= 0 || num_decoders > PT_MAX_CAMERAS)
        return PT_ERR_INVALID_PARAM;

    int any_error = 0;

    /*
     * Decode one frame from each camera.
     *
     * For maximum NVDEC throughput, we interleave the decode submissions:
     * submit all seeks first, then decode all frames.  This keeps the
     * hardware pipeline full.
     *
     * In practice, with the FFmpeg hwaccel path, the send_packet/receive_frame
     * calls are synchronous per-decoder, so the interleaving benefit is
     * limited.  The real parallelism comes from running decoders on
     * different threads (future work).
     *
     * For now, we process sequentially but skip cameras with frame_index == -1.
     */

    for (int i = 0; i < num_decoders; i++) {
        if (frame_indices[i] < 0) {
            /* No frame for this camera at this sync index; skip. */
            continue;
        }

        int ret = pt_video_decode_frame(&decoders[i], frame_indices[i],
                                         out_nv12_gpu[i], stream);
        if (ret != PT_OK) {
            pt_log_error("pt_video_decode_batch",
                         "decode failed for camera %d, frame %d (error %d)",
                         i, frame_indices[i], ret);
            any_error = 1;
        }
    }

    return any_error ? PT_ERR_DECODE : PT_OK;
}

void pt_video_close(PT_VideoDecoder *dec) {
    if (!dec) return;
    if (!dec->is_open) return;

#if PT_HAVE_FFMPEG
    if (dec->format_ctx) {
        close_ffmpeg(dec);
    }
#endif

#if PT_HAVE_OPENCV
    if (dec->cv_capture) {
        close_opencv(dec);
    }
#endif

    dec->is_open = 0;
}

} /* extern "C" */
