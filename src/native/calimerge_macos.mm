/*
 * calimerge_macos.mm
 *
 * macOS implementation using AVFoundation.
 * Objective-C++ file (.mm) to interface with Apple frameworks.
 *
 * Design: Each camera runs continuous capture in its own thread.
 * Frames store BOTH their native camera timestamp (CMSampleBuffer PTS) and
 * an arrival timestamp from a common clock (mach_absolute_time).
 *
 * Clock Synchronization Strategy:
 * - Each camera has its own clock domain for PTS timestamps
 * - We measure the offset between each camera's clock and a common reference
 * - When synchronizing frames, we apply offsets to compare camera timestamps
 * - This preserves the camera's native timing while enabling cross-camera sync
 */

#include "calimerge_platform.h"

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#include <pthread.h>
#include <mach/mach_time.h>
#include <string.h>
#include <stdlib.h>

/* ============================================================================
 * Mach Time Conversion (global, thread-safe after first call)
 * ============================================================================ */

static mach_timebase_info_data_t g_timebase_info = {0, 0};

static uint64_t get_timestamp_ns(void) {
    if (g_timebase_info.denom == 0) {
        mach_timebase_info(&g_timebase_info);
    }
    uint64_t mach_time = mach_absolute_time();
    return mach_time * g_timebase_info.numer / g_timebase_info.denom;
}

/* ============================================================================
 * Ring Buffer for Frame Storage
 * ============================================================================ */

#define RING_BUFFER_SIZE 8  /* Keep last N frames per camera */

typedef struct {
    uint8_t *pixels;
    int width, height;
    int stride;
    uint64_t camera_pts_ns;     /* Camera's own timestamp (from CMSampleBuffer PTS) */
    uint64_t arrival_ns;        /* Common clock timestamp when frame arrived */
    bool valid;
} RingFrame;

typedef struct {
    RingFrame frames[RING_BUFFER_SIZE];
    int write_index;        /* Next slot to write */
    int frame_count;        /* Total frames written (for detecting new frames) */
    pthread_mutex_t mutex;
    pthread_cond_t cond;    /* Signaled when new frame arrives */
} FrameRingBuffer;

static void ring_buffer_init(FrameRingBuffer *rb) {
    memset(rb, 0, sizeof(FrameRingBuffer));
    pthread_mutex_init(&rb->mutex, NULL);
    pthread_cond_init(&rb->cond, NULL);
}

static void ring_buffer_destroy(FrameRingBuffer *rb) {
    pthread_mutex_lock(&rb->mutex);
    for (int i = 0; i < RING_BUFFER_SIZE; i++) {
        free(rb->frames[i].pixels);
        rb->frames[i].pixels = NULL;
    }
    pthread_mutex_unlock(&rb->mutex);
    pthread_mutex_destroy(&rb->mutex);
    pthread_cond_destroy(&rb->cond);
}

/* Push a new frame (caller provides allocated pixels, ring buffer takes ownership) */
static void ring_buffer_push(FrameRingBuffer *rb, uint8_t *pixels, int width, int height,
                             uint64_t camera_pts_ns, uint64_t arrival_ns) {
    pthread_mutex_lock(&rb->mutex);

    int idx = rb->write_index;

    /* Free old frame if present */
    free(rb->frames[idx].pixels);

    /* Store new frame */
    rb->frames[idx].pixels = pixels;
    rb->frames[idx].width = width;
    rb->frames[idx].height = height;
    rb->frames[idx].stride = width * 3;
    rb->frames[idx].camera_pts_ns = camera_pts_ns;
    rb->frames[idx].arrival_ns = arrival_ns;
    rb->frames[idx].valid = true;

    rb->write_index = (rb->write_index + 1) % RING_BUFFER_SIZE;
    rb->frame_count++;

    /* Signal waiters */
    pthread_cond_broadcast(&rb->cond);
    pthread_mutex_unlock(&rb->mutex);
}

/* Get the most recent frame (copies data, caller must free pixels) */
static bool ring_buffer_get_latest(FrameRingBuffer *rb, CM_Frame *out, int64_t clock_offset_ns) {
    pthread_mutex_lock(&rb->mutex);

    if (rb->frame_count == 0) {
        pthread_mutex_unlock(&rb->mutex);
        return false;
    }

    /* Most recent is at (write_index - 1 + SIZE) % SIZE */
    int idx = (rb->write_index - 1 + RING_BUFFER_SIZE) % RING_BUFFER_SIZE;
    RingFrame *rf = &rb->frames[idx];

    if (!rf->valid || !rf->pixels) {
        pthread_mutex_unlock(&rb->mutex);
        return false;
    }

    int size = rf->width * rf->height * 3;
    out->pixels = (uint8_t *)malloc(size);
    memcpy(out->pixels, rf->pixels, size);
    out->width = rf->width;
    out->height = rf->height;
    out->stride = rf->stride;
    out->timestamp_ns = rf->camera_pts_ns;          /* Camera's native PTS */
    out->arrival_ns = rf->arrival_ns;               /* Common clock arrival time */
    out->corrected_ns = (uint64_t)((int64_t)rf->camera_pts_ns + clock_offset_ns);  /* Offset-corrected */

    pthread_mutex_unlock(&rb->mutex);
    return true;
}

/*
 * Get frame closest to target timestamp using CORRECTED PTS (Option B).
 * The target_common_ns is in the common clock domain.
 * The clock_offset_ns converts this camera's PTS to common domain.
 *
 * For each frame: corrected_pts = camera_pts + clock_offset
 * Find frame where corrected_pts is closest to target_common_ns.
 */
static bool ring_buffer_get_closest_corrected(
    FrameRingBuffer *rb,
    uint64_t target_common_ns,      /* Target time in common clock domain */
    int64_t clock_offset_ns,        /* This camera's offset: corrected = pts + offset */
    CM_Frame *out
) {
    pthread_mutex_lock(&rb->mutex);

    if (rb->frame_count == 0) {
        pthread_mutex_unlock(&rb->mutex);
        return false;
    }

    int best_idx = -1;
    uint64_t best_diff = UINT64_MAX;
    int available = (rb->frame_count < RING_BUFFER_SIZE) ? rb->frame_count : RING_BUFFER_SIZE;

    for (int i = 0; i < available; i++) {
        int idx = (rb->write_index - 1 - i + RING_BUFFER_SIZE) % RING_BUFFER_SIZE;
        RingFrame *rf = &rb->frames[idx];

        if (!rf->valid || !rf->pixels) continue;

        /* Convert camera PTS to common clock domain */
        uint64_t corrected_pts = (uint64_t)((int64_t)rf->camera_pts_ns + clock_offset_ns);

        /* Calculate difference from target */
        uint64_t diff = (corrected_pts > target_common_ns)
            ? (corrected_pts - target_common_ns)
            : (target_common_ns - corrected_pts);

        if (diff < best_diff) {
            best_diff = diff;
            best_idx = idx;
        }
    }

    if (best_idx < 0) {
        pthread_mutex_unlock(&rb->mutex);
        return false;
    }

    RingFrame *rf = &rb->frames[best_idx];
    int size = rf->width * rf->height * 3;
    out->pixels = (uint8_t *)malloc(size);
    memcpy(out->pixels, rf->pixels, size);
    out->width = rf->width;
    out->height = rf->height;
    out->stride = rf->stride;
    out->timestamp_ns = rf->camera_pts_ns;          /* Camera's native PTS */
    out->arrival_ns = rf->arrival_ns;               /* Common clock arrival time */
    out->corrected_ns = (uint64_t)((int64_t)rf->camera_pts_ns + clock_offset_ns);

    pthread_mutex_unlock(&rb->mutex);
    return true;
}

/* Get latest arrival timestamp (common clock domain) without copying frame */
static uint64_t ring_buffer_get_latest_arrival(FrameRingBuffer *rb) {
    pthread_mutex_lock(&rb->mutex);

    if (rb->frame_count == 0) {
        pthread_mutex_unlock(&rb->mutex);
        return 0;
    }

    int idx = (rb->write_index - 1 + RING_BUFFER_SIZE) % RING_BUFFER_SIZE;
    uint64_t ts = rb->frames[idx].arrival_ns;

    pthread_mutex_unlock(&rb->mutex);
    return ts;
}

/* Get latest camera PTS timestamp (camera's native clock) */
static uint64_t ring_buffer_get_latest_camera_pts(FrameRingBuffer *rb) {
    pthread_mutex_lock(&rb->mutex);

    if (rb->frame_count == 0) {
        pthread_mutex_unlock(&rb->mutex);
        return 0;
    }

    int idx = (rb->write_index - 1 + RING_BUFFER_SIZE) % RING_BUFFER_SIZE;
    uint64_t ts = rb->frames[idx].camera_pts_ns;

    pthread_mutex_unlock(&rb->mutex);
    return ts;
}

/* Get both timestamps from the latest frame (for clock offset calculation) */
static bool ring_buffer_get_latest_timestamps(FrameRingBuffer *rb, uint64_t *out_camera_pts, uint64_t *out_arrival) {
    pthread_mutex_lock(&rb->mutex);

    if (rb->frame_count == 0) {
        pthread_mutex_unlock(&rb->mutex);
        return false;
    }

    int idx = (rb->write_index - 1 + RING_BUFFER_SIZE) % RING_BUFFER_SIZE;
    RingFrame *rf = &rb->frames[idx];

    if (!rf->valid) {
        pthread_mutex_unlock(&rb->mutex);
        return false;
    }

    *out_camera_pts = rf->camera_pts_ns;
    *out_arrival = rf->arrival_ns;

    pthread_mutex_unlock(&rb->mutex);
    return true;
}

/* Wait for a new frame with timeout (returns frame count at return) */
static int ring_buffer_wait_for_frame(FrameRingBuffer *rb, int last_count, int timeout_ms) {
    pthread_mutex_lock(&rb->mutex);

    if (rb->frame_count > last_count) {
        int count = rb->frame_count;
        pthread_mutex_unlock(&rb->mutex);
        return count;
    }

    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_nsec += timeout_ms * 1000000LL;
    while (timeout.tv_nsec >= 1000000000) {
        timeout.tv_sec += 1;
        timeout.tv_nsec -= 1000000000;
    }

    pthread_cond_timedwait(&rb->cond, &rb->mutex, &timeout);
    int count = rb->frame_count;
    pthread_mutex_unlock(&rb->mutex);
    return count;
}

/* ============================================================================
 * Platform-Specific Handle
 * ============================================================================ */

typedef struct {
    /* AVFoundation objects */
    void *session;          /* AVCaptureSession */
    void *device;           /* AVCaptureDevice */
    void *input;            /* AVCaptureDeviceInput */
    void *output;           /* AVCaptureVideoDataOutput */
    void *delegate;         /* Our frame delegate object */
    void *capture_queue;    /* dispatch_queue_t */

    /* Ring buffer for continuous capture */
    FrameRingBuffer ring_buffer;

    /* Clock synchronization:
     * offset_ns = arrival_ns - camera_pts_ns
     * To convert camera_pts to common clock: camera_pts_ns + offset_ns = arrival_ns (approx)
     *
     * We measure this offset over several frames at startup and take the median
     * to get a stable estimate.
     */
    int64_t clock_offset_ns;        /* Signed: can be positive or negative */
    bool clock_offset_valid;

    /* State */
    bool is_open;
    int camera_index;
} MacOSCameraHandle;

/* ============================================================================
 * Frame Capture Delegate
 * ============================================================================ */

@interface CMFrameDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
{
    MacOSCameraHandle *handle;
}
- (instancetype)initWithHandle:(MacOSCameraHandle *)h;
@end

@implementation CMFrameDelegate

- (instancetype)initWithHandle:(MacOSCameraHandle *)h {
    self = [super init];
    if (self) {
        handle = h;
    }
    return self;
}

- (void)captureOutput:(AVCaptureOutput *)output
        didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
        fromConnection:(AVCaptureConnection *)connection {

    /*
     * Record BOTH timestamps:
     * 1. camera_pts_ns: The camera's native timestamp from CMSampleBuffer
     *    This is the time the camera says the frame was captured.
     * 2. arrival_ns: Our common clock (mach_absolute_time) when we received it.
     *    This is used for cross-camera synchronization.
     */
    uint64_t arrival_ns = get_timestamp_ns();

    /* Get camera's native PTS */
    CMTime pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
    uint64_t camera_pts_ns = 0;
    if (CMTIME_IS_VALID(pts)) {
        /* Convert CMTime to nanoseconds */
        camera_pts_ns = (uint64_t)(CMTimeGetSeconds(pts) * 1e9);
    } else {
        /* Fallback to arrival time if PTS is invalid */
        camera_pts_ns = arrival_ns;
    }

    /* Get pixel buffer */
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (!imageBuffer) return;

    CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    uint8_t *baseAddress = (uint8_t *)CVPixelBufferGetBaseAddress(imageBuffer);

    /* Allocate and convert BGRA -> BGR */
    int bgr_size = (int)(width * height * 3);
    uint8_t *bgr_pixels = (uint8_t *)malloc(bgr_size);

    /* Optimized conversion: process 4 pixels at a time where possible */
    for (size_t y = 0; y < height; y++) {
        uint8_t *src_row = baseAddress + y * bytesPerRow;
        uint8_t *dst_row = bgr_pixels + y * width * 3;

        size_t x = 0;
        /* Process bulk of pixels */
        for (; x + 4 <= width; x += 4) {
            /* Pixel 0 */
            dst_row[0] = src_row[0];  /* B */
            dst_row[1] = src_row[1];  /* G */
            dst_row[2] = src_row[2];  /* R */
            /* Pixel 1 */
            dst_row[3] = src_row[4];
            dst_row[4] = src_row[5];
            dst_row[5] = src_row[6];
            /* Pixel 2 */
            dst_row[6] = src_row[8];
            dst_row[7] = src_row[9];
            dst_row[8] = src_row[10];
            /* Pixel 3 */
            dst_row[9] = src_row[12];
            dst_row[10] = src_row[13];
            dst_row[11] = src_row[14];

            src_row += 16;
            dst_row += 12;
        }
        /* Handle remaining pixels */
        for (; x < width; x++) {
            dst_row[0] = src_row[0];
            dst_row[1] = src_row[1];
            dst_row[2] = src_row[2];
            src_row += 4;
            dst_row += 3;
        }
    }

    CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);

    /* Push to ring buffer (takes ownership of pixels) */
    ring_buffer_push(&handle->ring_buffer, bgr_pixels, (int)width, (int)height, camera_pts_ns, arrival_ns);
}

- (void)captureOutput:(AVCaptureOutput *)output
        didDropSampleBuffer:(CMSampleBufferRef)sampleBuffer
        fromConnection:(AVCaptureConnection *)connection {
    /* Frame dropped by AVFoundation - nothing we can do */
}

@end

/* ============================================================================
 * Clock Offset Calibration
 * ============================================================================ */

/*
 * Calibrate clock offset for a camera.
 * Collects several samples and uses the median for stability.
 * Call this after camera has started capturing (give it a moment to warm up).
 */
#define CLOCK_OFFSET_SAMPLES 10

static void calibrate_clock_offset(MacOSCameraHandle *handle) {
    int64_t offsets[CLOCK_OFFSET_SAMPLES];
    int sample_count = 0;
    int last_frame_count = 0;

    for (int attempt = 0; attempt < CLOCK_OFFSET_SAMPLES * 3 && sample_count < CLOCK_OFFSET_SAMPLES; attempt++) {
        /* Wait for a new frame */
        int current_count = ring_buffer_wait_for_frame(&handle->ring_buffer, last_frame_count, 100);
        if (current_count <= last_frame_count) continue;
        last_frame_count = current_count;

        uint64_t camera_pts, arrival;
        if (ring_buffer_get_latest_timestamps(&handle->ring_buffer, &camera_pts, &arrival)) {
            /* Offset = arrival - camera_pts
             * If positive: camera clock is behind common clock
             * If negative: camera clock is ahead of common clock
             */
            offsets[sample_count++] = (int64_t)arrival - (int64_t)camera_pts;
        }
    }

    if (sample_count < 3) {
        /* Not enough samples, use zero offset */
        handle->clock_offset_ns = 0;
        handle->clock_offset_valid = false;
        return;
    }

    /* Simple insertion sort for small array */
    for (int i = 1; i < sample_count; i++) {
        int64_t key = offsets[i];
        int j = i - 1;
        while (j >= 0 && offsets[j] > key) {
            offsets[j + 1] = offsets[j];
            j--;
        }
        offsets[j + 1] = key;
    }

    /* Use median */
    handle->clock_offset_ns = offsets[sample_count / 2];
    handle->clock_offset_valid = true;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

static bool g_initialized = false;

int cm_init(void) {
    if (g_initialized) return CM_OK;

    /* Initialize timebase for timestamp conversion */
    if (g_timebase_info.denom == 0) {
        mach_timebase_info(&g_timebase_info);
    }

    g_initialized = true;
    return CM_OK;
}

void cm_shutdown(void) {
    g_initialized = false;
}

/* ============================================================================
 * Camera Enumeration
 * ============================================================================ */

int cm_enumerate_cameras(CM_Camera *out_cameras, int max_cameras) {
    if (!g_initialized) {
        if (cm_init() != CM_OK) return CM_ERROR_INIT_FAILED;
    }
    if (!out_cameras || max_cameras <= 0) return CM_ERROR_INVALID_PARAM;

    @autoreleasepool {
        AVCaptureDeviceDiscoverySession *discovery = [AVCaptureDeviceDiscoverySession
            discoverySessionWithDeviceTypes:@[
                AVCaptureDeviceTypeBuiltInWideAngleCamera,
                AVCaptureDeviceTypeExternal
            ]
            mediaType:AVMediaTypeVideo
            position:AVCaptureDevicePositionUnspecified];

        NSArray<AVCaptureDevice *> *devices = discovery.devices;
        int count = 0;

        for (AVCaptureDevice *device in devices) {
            if (count >= max_cameras) break;

            CM_Camera *cam = &out_cameras[count];
            memset(cam, 0, sizeof(CM_Camera));

            const char *uid = [device.uniqueID UTF8String];
            strncpy(cam->serial_number, uid ? uid : "unknown", CM_SERIAL_LEN - 1);

            const char *name = [device.localizedName UTF8String];
            strncpy(cam->display_name, name ? name : "Unknown Camera", CM_NAME_LEN - 1);

            cam->device_index = count;
            cam->enabled = true;
            cam->platform_handle = NULL;

            /* Query supported resolutions */
            cam->supported_resolution_count = 0;
            static const CM_Resolution test_resolutions[] = {
                {640, 480}, {1280, 720}, {1920, 1080}
            };

            for (AVCaptureDeviceFormat *format in device.formats) {
                CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription);

                for (int r = 0; r < CM_RES_COUNT; r++) {
                    if (dims.width == test_resolutions[r].width &&
                        dims.height == test_resolutions[r].height) {
                        bool found = false;
                        for (int i = 0; i < cam->supported_resolution_count; i++) {
                            if (cam->supported_resolutions[i].width == dims.width &&
                                cam->supported_resolutions[i].height == dims.height) {
                                found = true;
                                break;
                            }
                        }
                        if (!found && cam->supported_resolution_count < CM_RES_COUNT) {
                            cam->supported_resolutions[cam->supported_resolution_count].width = dims.width;
                            cam->supported_resolutions[cam->supported_resolution_count].height = dims.height;
                            cam->supported_resolution_count++;
                        }
                    }
                }
            }

            if (cam->supported_resolution_count > 0) {
                int best = cam->supported_resolution_count - 1;
                cam->width = cam->supported_resolutions[best].width;
                cam->height = cam->supported_resolutions[best].height;
            } else {
                cam->width = 640;
                cam->height = 480;
            }

            cam->fps = 30;
            cam->rotation = 0;
            cam->exposure = 0;

            count++;
        }

        return count;
    }
}

int cm_get_camera_serial(int device_index, char *out_serial, int max_len) {
    if (!out_serial || max_len <= 0) return CM_ERROR_INVALID_PARAM;

    CM_Camera cameras[CM_MAX_CAMERAS];
    int count = cm_enumerate_cameras(cameras, CM_MAX_CAMERAS);

    if (device_index < 0 || device_index >= count) {
        return CM_ERROR_NO_CAMERA;
    }

    strncpy(out_serial, cameras[device_index].serial_number, max_len - 1);
    out_serial[max_len - 1] = '\0';
    return CM_OK;
}

/* ============================================================================
 * Camera Control
 * ============================================================================ */

int cm_open_camera(CM_Camera *camera) {
    if (!camera) return CM_ERROR_INVALID_PARAM;
    if (camera->platform_handle) return CM_OK;

    @autoreleasepool {
        AVCaptureDeviceDiscoverySession *discovery = [AVCaptureDeviceDiscoverySession
            discoverySessionWithDeviceTypes:@[
                AVCaptureDeviceTypeBuiltInWideAngleCamera,
                AVCaptureDeviceTypeExternal
            ]
            mediaType:AVMediaTypeVideo
            position:AVCaptureDevicePositionUnspecified];

        AVCaptureDevice *device = nil;
        NSString *targetSerial = [NSString stringWithUTF8String:camera->serial_number];

        for (AVCaptureDevice *d in discovery.devices) {
            if ([d.uniqueID isEqualToString:targetSerial]) {
                device = d;
                break;
            }
        }

        if (!device) return CM_ERROR_NO_CAMERA;

        MacOSCameraHandle *handle = (MacOSCameraHandle *)calloc(1, sizeof(MacOSCameraHandle));
        if (!handle) return CM_ERROR_OPEN_FAILED;

        ring_buffer_init(&handle->ring_buffer);
        handle->camera_index = camera->device_index;

        AVCaptureSession *session = [[AVCaptureSession alloc] init];
        handle->session = (__bridge_retained void *)session;
        handle->device = (__bridge_retained void *)device;

        if (camera->width >= 1920 && camera->height >= 1080) {
            session.sessionPreset = AVCaptureSessionPreset1920x1080;
        } else if (camera->width >= 1280 && camera->height >= 720) {
            session.sessionPreset = AVCaptureSessionPreset1280x720;
        } else {
            session.sessionPreset = AVCaptureSessionPreset640x480;
        }

        NSError *error = nil;
        AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
        if (!input || error) {
            ring_buffer_destroy(&handle->ring_buffer);
            free(handle);
            return CM_ERROR_OPEN_FAILED;
        }
        handle->input = (__bridge_retained void *)input;

        if (![session canAddInput:input]) {
            ring_buffer_destroy(&handle->ring_buffer);
            free(handle);
            return CM_ERROR_OPEN_FAILED;
        }
        [session addInput:input];

        AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
        output.alwaysDiscardsLateVideoFrames = YES;
        output.videoSettings = @{
            (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)
        };

        CMFrameDelegate *delegate = [[CMFrameDelegate alloc] initWithHandle:handle];
        handle->delegate = (__bridge_retained void *)delegate;

        /* Use high-priority queue for frame capture */
        dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
            DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
        dispatch_queue_t queue = dispatch_queue_create("com.calimerge.capture", attr);
        handle->capture_queue = (__bridge_retained void *)queue;

        [output setSampleBufferDelegate:delegate queue:queue];
        handle->output = (__bridge_retained void *)output;

        if (![session canAddOutput:output]) {
            ring_buffer_destroy(&handle->ring_buffer);
            free(handle);
            return CM_ERROR_OPEN_FAILED;
        }
        [session addOutput:output];

        /* Configure frame rate */
        if ([device lockForConfiguration:&error]) {
            for (AVCaptureDeviceFormat *format in device.formats) {
                CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription);
                if (dims.width == camera->width && dims.height == camera->height) {
                    for (AVFrameRateRange *range in format.videoSupportedFrameRateRanges) {
                        if (range.minFrameRate <= camera->fps && range.maxFrameRate >= camera->fps) {
                            device.activeFormat = format;
                            device.activeVideoMinFrameDuration = CMTimeMake(1, camera->fps);
                            device.activeVideoMaxFrameDuration = CMTimeMake(1, camera->fps);
                            goto configured;
                        }
                    }
                }
            }
            configured:
            [device unlockForConfiguration];
        }

        [session startRunning];
        handle->is_open = true;
        camera->platform_handle = handle;

        /*
         * Calibrate clock offset:
         * Wait briefly for camera to start producing frames, then measure
         * the offset between camera timestamps and our common clock.
         */
        usleep(200000);  /* 200ms warmup */
        calibrate_clock_offset(handle);

        return CM_OK;
    }
}

void cm_close_camera(CM_Camera *camera) {
    if (!camera || !camera->platform_handle) return;

    MacOSCameraHandle *handle = (MacOSCameraHandle *)camera->platform_handle;

    @autoreleasepool {
        AVCaptureSession *session = (__bridge_transfer AVCaptureSession *)handle->session;
        [session stopRunning];

        (void)(__bridge_transfer AVCaptureDevice *)handle->device;
        (void)(__bridge_transfer AVCaptureDeviceInput *)handle->input;
        (void)(__bridge_transfer AVCaptureVideoDataOutput *)handle->output;
        (void)(__bridge_transfer CMFrameDelegate *)handle->delegate;
        (void)(__bridge_transfer dispatch_queue_t)handle->capture_queue;
    }

    ring_buffer_destroy(&handle->ring_buffer);
    free(handle);
    camera->platform_handle = NULL;
}

int cm_set_resolution(CM_Camera *camera, int width, int height) {
    if (!camera) return CM_ERROR_INVALID_PARAM;
    camera->width = width;
    camera->height = height;
    if (camera->platform_handle) {
        cm_close_camera(camera);
        return cm_open_camera(camera);
    }
    return CM_OK;
}

int cm_set_fps(CM_Camera *camera, int fps) {
    if (!camera) return CM_ERROR_INVALID_PARAM;
    camera->fps = fps;

    if (camera->platform_handle) {
        MacOSCameraHandle *handle = (MacOSCameraHandle *)camera->platform_handle;
        @autoreleasepool {
            AVCaptureDevice *device = (__bridge AVCaptureDevice *)handle->device;
            NSError *error = nil;
            if ([device lockForConfiguration:&error]) {
                device.activeVideoMinFrameDuration = CMTimeMake(1, fps);
                device.activeVideoMaxFrameDuration = CMTimeMake(1, fps);
                [device unlockForConfiguration];
            }
        }
    }
    return CM_OK;
}

int cm_set_exposure(CM_Camera *camera, int exposure) {
    if (!camera) return CM_ERROR_INVALID_PARAM;
    camera->exposure = exposure;
    /* Note: Manual exposure not available on macOS AVFoundation for most webcams */
    return CM_OK;
}

/* ============================================================================
 * Frame Capture
 * ============================================================================ */

int cm_capture_frame(CM_Camera *camera, CM_Frame *out_frame) {
    if (!camera || !out_frame) return CM_ERROR_INVALID_PARAM;
    if (!camera->platform_handle) return CM_ERROR_NO_CAMERA;

    MacOSCameraHandle *handle = (MacOSCameraHandle *)camera->platform_handle;

    /* Try to get latest frame, wait if none available */
    if (!ring_buffer_get_latest(&handle->ring_buffer, out_frame, handle->clock_offset_ns)) {
        /* Wait up to 100ms for a frame */
        ring_buffer_wait_for_frame(&handle->ring_buffer, 0, 100);

        if (!ring_buffer_get_latest(&handle->ring_buffer, out_frame, handle->clock_offset_ns)) {
            return CM_ERROR_CAPTURE_FAILED;
        }
    }

    out_frame->camera_index = camera->device_index;
    return CM_OK;
}

void cm_release_frame(CM_Frame *frame) {
    if (frame && frame->pixels) {
        free(frame->pixels);
        frame->pixels = NULL;
    }
}

uint64_t cm_get_latest_timestamp(CM_Camera *camera) {
    if (!camera || !camera->platform_handle) return 0;
    MacOSCameraHandle *handle = (MacOSCameraHandle *)camera->platform_handle;
    return ring_buffer_get_latest_camera_pts(&handle->ring_buffer);
}

/* ============================================================================
 * Multi-Camera Synchronization
 * ============================================================================ */

int cm_capture_synced(CM_Camera *cameras, int camera_count, CM_SyncedFrameSet *out) {
    if (!cameras || !out || camera_count <= 0) return CM_ERROR_INVALID_PARAM;

    memset(out, 0, sizeof(CM_SyncedFrameSet));

    /*
     * Synchronization strategy (Option B - Clock-Offset Corrected PTS):
     *
     * Each camera has its own clock domain for PTS timestamps. We measure
     * the clock offset (arrival_ns - camera_pts_ns) at startup and use it
     * to convert camera timestamps to a common clock domain.
     *
     * 1. Get latest CORRECTED timestamp from each camera
     *    (corrected = camera_pts + clock_offset)
     * 2. Compute the mean corrected timestamp as target
     * 3. For each camera, find frame with corrected timestamp closest to target
     * 4. Return frames with all timestamps (raw PTS, arrival, corrected)
     *
     * This uses the camera's native timing (more precise) while enabling
     * cross-camera comparison via offset correction.
     */

    /* Step 1: Collect latest CORRECTED timestamps */
    uint64_t corrected_times[CM_MAX_CAMERAS] = {0};
    int64_t offsets[CM_MAX_CAMERAS] = {0};
    uint64_t sum_corrected = 0;
    int valid_count = 0;

    for (int i = 0; i < camera_count && i < CM_MAX_CAMERAS; i++) {
        if (!cameras[i].platform_handle) {
            out->dropped_mask |= (1 << i);
            continue;
        }

        MacOSCameraHandle *handle = (MacOSCameraHandle *)cameras[i].platform_handle;
        offsets[i] = handle->clock_offset_ns;

        uint64_t camera_pts = ring_buffer_get_latest_camera_pts(&handle->ring_buffer);
        if (camera_pts > 0) {
            /* Convert to common clock domain using measured offset */
            corrected_times[i] = (uint64_t)((int64_t)camera_pts + handle->clock_offset_ns);
            sum_corrected += corrected_times[i];
            valid_count++;
        } else {
            out->dropped_mask |= (1 << i);
        }
    }

    if (valid_count == 0) {
        return CM_ERROR_CAPTURE_FAILED;
    }

    /* Step 2: Compute target time (mean of corrected timestamps) */
    uint64_t target_corrected = sum_corrected / valid_count;

    /* Step 3: Get frame closest to target (using corrected timestamps for matching) */
    for (int i = 0; i < camera_count && i < CM_MAX_CAMERAS; i++) {
        if (out->dropped_mask & (1 << i)) continue;

        MacOSCameraHandle *handle = (MacOSCameraHandle *)cameras[i].platform_handle;

        if (ring_buffer_get_closest_corrected(&handle->ring_buffer, target_corrected,
                                               handle->clock_offset_ns, &out->frames[i])) {
            out->frames[i].camera_index = i;
            out->frame_count++;
        } else {
            out->dropped_mask |= (1 << i);
        }
    }

    static uint64_t sync_counter = 0;
    out->sync_index = sync_counter++;

    return CM_OK;
}

void cm_release_synced(CM_SyncedFrameSet *frameset) {
    if (!frameset) return;
    for (int i = 0; i < CM_MAX_CAMERAS; i++) {
        cm_release_frame(&frameset->frames[i]);
    }
}
