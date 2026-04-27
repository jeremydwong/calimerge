/*
 * pt_videodecode.m - Video decode via AVAssetReader on macOS.
 *
 * Uses AVFoundation's AVAssetReader for hardware-accelerated decode.
 * Output is BGRA from VideoToolbox, converted to BGR8 for the pipeline.
 */

#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#include "pt_videodecode.h"
#include <string.h>
#include <stdio.h>

struct PT_VideoReader {
    void *asset_reader;       /* AVAssetReader*             */
    void *track_output;       /* AVAssetReaderTrackOutput*  */
    int   width;
    int   height;
    int   frame_count;
    double fps;
};

int pt_video_open(PT_VideoReader **out,
                  const char *video_path,
                  int *out_width, int *out_height,
                  double *out_fps, int *out_frame_count) {
    if (!out || !video_path) return PT_ERR_INVALID_ARGS;

    @autoreleasepool {
        NSString *path = [NSString stringWithUTF8String:video_path];
        NSURL *url = [NSURL fileURLWithPath:path];

        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
            fprintf(stderr, "[pt_videodecode] File not found: %s\n", video_path);
            return PT_ERR_FILE_NOT_FOUND;
        }

        AVAsset *asset = [AVAsset assetWithURL:url];
        NSArray *tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
        if (tracks.count == 0) {
            fprintf(stderr, "[pt_videodecode] No video tracks in: %s\n", video_path);
            return PT_ERR_FILE_NOT_FOUND;
        }

        AVAssetTrack *track = tracks[0];
        CGSize size = track.naturalSize;
        float fps = track.nominalFrameRate;
        CMTimeRange range = track.timeRange;
        int frame_count = (int)(CMTimeGetSeconds(range.duration) * fps + 0.5);

        NSError *error = nil;
        AVAssetReader *reader = [AVAssetReader assetReaderWithAsset:asset error:&error];
        if (!reader) {
            fprintf(stderr, "[pt_videodecode] AVAssetReader failed: %s\n",
                    error.localizedDescription.UTF8String);
            return PT_ERR_ENGINE_BUILD;
        }

        /* Request BGRA output (hardware-decoded) */
        NSDictionary *settings = @{
            (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)
        };

        AVAssetReaderTrackOutput *output =
            [AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:track
                                                      outputSettings:settings];
        output.alwaysCopiesSampleData = NO;  /* zero-copy when possible */

        [reader addOutput:output];

        if (![reader startReading]) {
            fprintf(stderr, "[pt_videodecode] Failed to start reading: %s\n",
                    reader.error.localizedDescription.UTF8String);
            return PT_ERR_ENGINE_BUILD;
        }

        /* Allocate handle */
        PT_VideoReader *r = calloc(1, sizeof(PT_VideoReader));
        r->asset_reader = (__bridge_retained void *)reader;
        r->track_output = (__bridge_retained void *)output;
        r->width  = (int)size.width;
        r->height = (int)size.height;
        r->fps    = fps;
        r->frame_count = frame_count;

        if (out_width)       *out_width = r->width;
        if (out_height)      *out_height = r->height;
        if (out_fps)         *out_fps = r->fps;
        if (out_frame_count) *out_frame_count = r->frame_count;

        *out = r;
    }

    return PT_OK;
}

int pt_video_read_frame(PT_VideoReader *reader, uint8_t *bgr_out) {
    if (!reader || !bgr_out) return PT_ERR_INVALID_ARGS;

    @autoreleasepool {
        AVAssetReaderTrackOutput *output =
            (__bridge AVAssetReaderTrackOutput *)reader->track_output;

        CMSampleBufferRef sample = [output copyNextSampleBuffer];
        if (!sample) {
            return PT_ERR_EOF;
        }

        CVImageBufferRef img = CMSampleBufferGetImageBuffer(sample);
        CVPixelBufferLockBaseAddress(img, kCVPixelBufferLock_ReadOnly);

        int w = (int)CVPixelBufferGetWidth(img);
        int h = (int)CVPixelBufferGetHeight(img);
        size_t stride = CVPixelBufferGetBytesPerRow(img);
        const uint8_t *base = (const uint8_t *)CVPixelBufferGetBaseAddress(img);

        /* Convert BGRA -> BGR */
        for (int y = 0; y < h; y++) {
            const uint8_t *src_row = base + y * stride;
            uint8_t *dst_row = bgr_out + y * w * 3;
            for (int x = 0; x < w; x++) {
                dst_row[x * 3 + 0] = src_row[x * 4 + 0];  /* B */
                dst_row[x * 3 + 1] = src_row[x * 4 + 1];  /* G */
                dst_row[x * 3 + 2] = src_row[x * 4 + 2];  /* R */
            }
        }

        CVPixelBufferUnlockBaseAddress(img, kCVPixelBufferLock_ReadOnly);
        CFRelease(sample);
    }

    return PT_OK;
}

void pt_video_close(PT_VideoReader *reader) {
    if (!reader) return;

    @autoreleasepool {
        if (reader->track_output) {
            AVAssetReaderTrackOutput *output =
                (__bridge_transfer AVAssetReaderTrackOutput *)reader->track_output;
            (void)output;
        }
        if (reader->asset_reader) {
            AVAssetReader *r =
                (__bridge_transfer AVAssetReader *)reader->asset_reader;
            [r cancelReading];
            (void)r;
        }
    }

    free(reader);
}
