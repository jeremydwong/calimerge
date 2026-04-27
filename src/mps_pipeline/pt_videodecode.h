/*
 * pt_videodecode.h - Hardware-accelerated video decode for batch mode on macOS.
 *
 * Uses AVAssetReader + AVAssetReaderTrackOutput for H.264/HEVC decode.
 * Hardware acceleration is automatic on Apple Silicon via VideoToolbox.
 *
 * For streaming mode (live cameras), this module is not used — frames come
 * directly from cm_capture_synced().
 *
 * Style: Plain C API (extern "C"), implemented in Objective-C (.m).
 */

#ifndef PT_VIDEODECODE_H
#define PT_VIDEODECODE_H

#include "../pt_shared/pt_common.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle for a video reader */
typedef struct PT_VideoReader PT_VideoReader;

/*
 * Open a video file for reading.
 *
 * Frames are decoded to BGRA and converted to BGR8 on read.
 * The reader delivers frames in decode order (presentation time).
 *
 * Returns PT_OK on success, PT_ERR_FILE_NOT_FOUND if file doesn't exist.
 */
int pt_video_open(PT_VideoReader **out,
                  const char *video_path,
                  int *out_width, int *out_height,
                  double *out_fps, int *out_frame_count);

/*
 * Read the next frame as BGR8.
 *
 * bgr_out must point to a buffer of at least width * height * 3 bytes.
 * Returns PT_OK on success, PT_ERR_EOF when no more frames.
 */
int pt_video_read_frame(PT_VideoReader *reader, uint8_t *bgr_out);

/*
 * Close the video reader and free resources.
 */
void pt_video_close(PT_VideoReader *reader);

/* Error code for end of file */
#define PT_ERR_EOF 100

#ifdef __cplusplus
}
#endif

#endif /* PT_VIDEODECODE_H */
