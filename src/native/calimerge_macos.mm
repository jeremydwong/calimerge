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
 *
 * UVC Exposure Control:
 * - AVFoundation doesn't expose exposure control for external USB cameras
 * - We use IOKit to send UVC control requests directly to USB cameras
 * - This works for UVC-compliant webcams (most external USB webcams)
 */

#include "calimerge_platform.h"

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <IOKit/usb/IOUSBLib.h>
#import <IOKit/IOCFPlugIn.h>
#include <pthread.h>
#include <mach/mach_time.h>
#include <string.h>
#include <stdlib.h>

/* ============================================================================
 * UVC (USB Video Class) Control Definitions
 * From USB Video Class 1.5 Specification
 * ============================================================================ */

/* UVC Control Interface selectors */
#define UVC_CT_CONTROL_UNDEFINED                0x00
#define UVC_CT_SCANNING_MODE_CONTROL            0x01
#define UVC_CT_AE_MODE_CONTROL                  0x02  /* Auto Exposure Mode */
#define UVC_CT_AE_PRIORITY_CONTROL              0x03
#define UVC_CT_EXPOSURE_TIME_ABSOLUTE_CONTROL   0x04  /* Exposure Time */
#define UVC_CT_EXPOSURE_TIME_RELATIVE_CONTROL   0x05
#define UVC_CT_FOCUS_ABSOLUTE_CONTROL           0x06
#define UVC_CT_FOCUS_RELATIVE_CONTROL           0x07
#define UVC_CT_FOCUS_AUTO_CONTROL               0x08
#define UVC_CT_IRIS_ABSOLUTE_CONTROL            0x09
#define UVC_CT_IRIS_RELATIVE_CONTROL            0x0A
#define UVC_CT_ZOOM_ABSOLUTE_CONTROL            0x0B
#define UVC_CT_ZOOM_RELATIVE_CONTROL            0x0C
#define UVC_CT_PANTILT_ABSOLUTE_CONTROL         0x0D
#define UVC_CT_PANTILT_RELATIVE_CONTROL         0x0E
#define UVC_CT_ROLL_ABSOLUTE_CONTROL            0x0F
#define UVC_CT_ROLL_RELATIVE_CONTROL            0x10
#define UVC_CT_PRIVACY_CONTROL                  0x11

/* UVC request types */
#define UVC_SET_CUR     0x01
#define UVC_GET_CUR     0x81
#define UVC_GET_MIN     0x82
#define UVC_GET_MAX     0x83
#define UVC_GET_RES     0x84
#define UVC_GET_LEN     0x85
#define UVC_GET_INFO    0x86
#define UVC_GET_DEF     0x87

/* UVC Interface types */
#define UVC_INTERFACE_CONTROL   0
#define UVC_INTERFACE_STREAMING 1

/* UVC Auto Exposure Mode bits */
#define UVC_AE_MODE_MANUAL          0x01  /* Manual exposure time, manual iris */
#define UVC_AE_MODE_AUTO            0x02  /* Auto exposure time, auto iris */
#define UVC_AE_MODE_SHUTTER_PRIORITY 0x04 /* Manual exposure time, auto iris */
#define UVC_AE_MODE_APERTURE_PRIORITY 0x08 /* Auto exposure time, manual iris */

/* Camera Terminal Unit ID - typically 1 for most UVC cameras */
#define UVC_INPUT_TERMINAL_ID   1

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

#define RING_BUFFER_SIZE 64  /* Keep last N frames per camera (2s at 30fps) */

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

static void ring_buffer_flush(FrameRingBuffer *rb) {
    pthread_mutex_lock(&rb->mutex);
    for (int i = 0; i < RING_BUFFER_SIZE; i++) {
        free(rb->frames[i].pixels);
        rb->frames[i].pixels = NULL;
        rb->frames[i].valid = false;
    }
    rb->write_index = 0;
    /* Keep frame_count so waiters don't get confused, just clear the data */
    pthread_mutex_unlock(&rb->mutex);
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

    @autoreleasepool {
        /* Guard: skip if camera is being shut down */
        if (!handle->is_open) return;

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

/*
 * Try to read the real USB iSerialNumber descriptor for the camera identified
 * by an AVFoundation uniqueID string.
 *
 * AVFoundation uniqueID format for external USB cameras:
 *   "0xLLLLLLLLVVVVPPPP"  (64-bit hex)
 *   bits 63-32 : USB locationID  (bus + port topology)
 *   bits 31-16 : USB vendorID
 *   bits 15-0  : USB productID
 *
 * Strategy:
 *   1. Parse the hex value and extract locationID, vendorID, productID.
 *   2. Walk IOKit USB device tree; match on locationID + VID + PID
 *      (locationID uniquely identifies port, so two identical cameras on
 *      different ports are distinguished correctly).
 *   3. Read the "USB Serial Number" IOKit property — this is the raw
 *      iSerialNumber string from the USB descriptor, identical to what
 *      Windows reads from its symbolic link.
 *   4. Return the serial string, or nil if the camera has no serial /
 *      is not a USB device.
 */
static NSString *get_usb_serial_for_avf_camera(NSString *avfUniqueID) {
    if (!avfUniqueID || ![avfUniqueID hasPrefix:@"0x"]) {
        return nil; /* built-in FaceTime camera — no USB serial */
    }

    /* Parse the 64-bit AVF uniqueID */
    unsigned long long avfID = 0;
    NSScanner *scanner = [NSScanner scannerWithString:avfUniqueID];
    [scanner scanHexLongLong:&avfID];

    uint32_t locationID = (uint32_t)((avfID >> 32) & 0xFFFFFFFF);
    uint16_t vendorID   = (uint16_t)((avfID >> 16) & 0xFFFF);
    uint16_t productID  = (uint16_t)( avfID        & 0xFFFF);

    fprintf(stderr, "[serial] AVF uniqueID %s  loc=0x%08x vid=0x%04x pid=0x%04x\n",
            [avfUniqueID UTF8String], locationID, vendorID, productID);

    CFMutableDictionaryRef matching = IOServiceMatching(kIOUSBDeviceClassName);
    if (!matching) return nil;

    io_iterator_t iterator;
    kern_return_t kr = IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator);
    if (kr != KERN_SUCCESS) return nil;

    NSString *result = nil;
    io_service_t service;

    while ((service = IOIteratorNext(iterator))) {
        /* Read locationID, vendorID, productID from IOKit */
        CFNumberRef locRef = (CFNumberRef)IORegistryEntryCreateCFProperty(
            service, CFSTR("locationID"), kCFAllocatorDefault, 0);
        CFNumberRef vidRef = (CFNumberRef)IORegistryEntryCreateCFProperty(
            service, CFSTR("idVendor"), kCFAllocatorDefault, 0);
        CFNumberRef pidRef = (CFNumberRef)IORegistryEntryCreateCFProperty(
            service, CFSTR("idProduct"), kCFAllocatorDefault, 0);

        uint32_t loc = 0;
        int vid = 0, pid = 0;
        if (locRef) { CFNumberGetValue(locRef, kCFNumberSInt32Type, &loc); CFRelease(locRef); }
        if (vidRef) { CFNumberGetValue(vidRef, kCFNumberIntType, &vid);    CFRelease(vidRef); }
        if (pidRef) { CFNumberGetValue(pidRef, kCFNumberIntType, &pid);    CFRelease(pidRef); }

        bool vidpid_match = ((int)vendorID == vid && (int)productID == pid);

        /* Prefer locationID match; fall back to VID+PID only when locationID
         * is zero (shouldn't happen for real USB devices, but be defensive). */
        bool matched = vidpid_match &&
                       (locationID == 0 || loc == locationID);

        if (matched) {
            fprintf(stderr, "[serial] matched IOKit device loc=0x%08x vid=0x%04x pid=0x%04x\n",
                    loc, vid, pid);
            CFStringRef serialRef = (CFStringRef)IORegistryEntryCreateCFProperty(
                service, CFSTR("USB Serial Number"), kCFAllocatorDefault, 0);
            if (serialRef) {
                NSString *s = (__bridge_transfer NSString *)serialRef;
                if (s.length > 0) {
                    fprintf(stderr, "[serial] USB iSerialNumber: %s\n", [s UTF8String]);
                    result = s;
                } else {
                    fprintf(stderr, "[serial] USB Serial Number property is empty\n");
                }
            } else {
                fprintf(stderr, "[serial] no USB Serial Number property on this device\n");
            }
            IOObjectRelease(service);
            break;
        }

        IOObjectRelease(service);
    }

    IOObjectRelease(iterator);
    if (!result) {
        fprintf(stderr, "[serial] no IOKit match found for %s — using AVF uniqueID\n",
                [avfUniqueID UTF8String]);
    }
    return result;
}

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

            /* Prefer the real USB iSerialNumber so serial numbers match
             * the Windows implementation.  Fall back to AVF uniqueID for
             * cameras that have no hardware serial (e.g. built-in webcam). */
            NSString *usbSerial = get_usb_serial_for_avf_camera(device.uniqueID);
            if (usbSerial) {
                strncpy(cam->serial_number, [usbSerial UTF8String], CM_SERIAL_LEN - 1);
                fprintf(stderr, "[enum] camera %d '%s'  serial=%s  (from USB iSerialNumber)\n",
                        count, device.localizedName.UTF8String ?: "?", cam->serial_number);
            } else {
                const char *uid = [device.uniqueID UTF8String];
                strncpy(cam->serial_number, uid ? uid : "unknown", CM_SERIAL_LEN - 1);
                fprintf(stderr, "[enum] camera %d '%s'  serial=%s  (from AVF uniqueID)\n",
                        count, device.localizedName.UTF8String ?: "?", cam->serial_number);
            }

            const char *name = [device.localizedName UTF8String];
            strncpy(cam->display_name, name ? name : "Unknown Camera", CM_NAME_LEN - 1);

            cam->device_index = count;
            cam->enabled = true;
            cam->platform_handle = NULL;

            /* Enumerate all supported (width, height, fps) tuples.
             * Each AVCaptureDeviceFormat has a resolution and one or more
             * AVFrameRateRange entries.  We extract the max fps from each
             * range and deduplicate by (w, h, fps). */
            cam->supported_format_count = 0;

            for (AVCaptureDeviceFormat *format in device.formats) {
                CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription);
                int w = dims.width;
                int h = dims.height;

                for (AVFrameRateRange *range in format.videoSupportedFrameRateRanges) {
                    int fps = (int)range.maxFrameRate;

                    /* Deduplicate */
                    bool found = false;
                    for (int i = 0; i < cam->supported_format_count; i++) {
                        if (cam->supported_formats[i].width == w &&
                            cam->supported_formats[i].height == h &&
                            cam->supported_formats[i].fps == fps) {
                            found = true;
                            break;
                        }
                    }
                    if (!found && cam->supported_format_count < CM_MAX_FORMATS) {
                        cam->supported_formats[cam->supported_format_count].width = w;
                        cam->supported_formats[cam->supported_format_count].height = h;
                        cam->supported_formats[cam->supported_format_count].fps = fps;
                        cam->supported_format_count++;
                    }
                }
            }

            /*
             * Default to camera's current activeFormat if available,
             * else try 1280x720@30, else use first supported format.
             */
            bool default_set = false;

            /* First, try to read the camera's current activeFormat */
            AVCaptureDeviceFormat *currentFormat = device.activeFormat;
            if (currentFormat) {
                CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(currentFormat.formatDescription);
                int curr_w = dims.width;
                int curr_h = dims.height;
                /* Get current FPS from the first frame rate range */
                int curr_fps = 30;  /* Default fallback */
                for (AVFrameRateRange *range in currentFormat.videoSupportedFrameRateRanges) {
                    curr_fps = (int)range.maxFrameRate;
                    break;
                }
                /* Verify this format is in our supported list */
                for (int i = 0; i < cam->supported_format_count; i++) {
                    if (cam->supported_formats[i].width == curr_w &&
                        cam->supported_formats[i].height == curr_h) {
                        cam->width = curr_w;
                        cam->height = curr_h;
                        cam->fps = curr_fps;
                        default_set = true;
                        break;
                    }
                }
            }

            /* Fallback: try 1280x720@30 if available */
            if (!default_set) {
                for (int i = 0; i < cam->supported_format_count; i++) {
                    if (cam->supported_formats[i].width == 1280 &&
                        cam->supported_formats[i].height == 720 &&
                        cam->supported_formats[i].fps == 30) {
                        cam->width = 1280;
                        cam->height = 720;
                        cam->fps = 30;
                        default_set = true;
                        break;
                    }
                }
            }

            /* Final fallback: use first supported format */
            if (!default_set && cam->supported_format_count > 0) {
                cam->width = cam->supported_formats[0].width;
                cam->height = cam->supported_formats[0].height;
                cam->fps = cam->supported_formats[0].fps;
            } else if (!default_set) {
                cam->width = 640;
                cam->height = 480;
                cam->fps = 30;
            }

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

        /*
         * Do NOT set a sessionPreset (e.g. Preset1280x720).  That would
         * override any manual activeFormat changes.  On macOS the default
         * AVCaptureSessionPresetHigh allows manual activeFormat control.
         * (AVCaptureSessionPresetInputPriority is iOS-only.)
         */

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

        /*
         * Set initial format: find AVCaptureDeviceFormat matching (w, h, fps).
         *
         * macOS has no AVCaptureSessionPresetInputPriority, so the session
         * will override our activeFormat when it starts — UNLESS we keep the
         * device locked through startRunning.  We lock, set format, start
         * the session, THEN unlock.
         */
        if ([device lockForConfiguration:&error]) {
            for (AVCaptureDeviceFormat *format in device.formats) {
                CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription);
                if (dims.width != camera->width || dims.height != camera->height) continue;

                for (AVFrameRateRange *range in format.videoSupportedFrameRateRanges) {
                    if (range.minFrameRate <= camera->fps && range.maxFrameRate >= camera->fps) {
                        device.activeFormat = format;
                        device.activeVideoMinFrameDuration = CMTimeMake(1, camera->fps);
                        device.activeVideoMaxFrameDuration = CMTimeMake(1, camera->fps);
                        goto configured;
                    }
                }
            }
            configured:
            [session startRunning];
            [device unlockForConfiguration];
        } else {
            [session startRunning];
        }
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
    handle->is_open = false;

    @autoreleasepool {
        AVCaptureSession *session = (__bridge AVCaptureSession *)handle->session;
        [session stopRunning];

        /*
         * Drain the capture queue: stopRunning prevents NEW frames from being
         * delivered, but a delegate callback may still be executing. Dispatch
         * a synchronous no-op to the capture queue to wait for it to finish.
         */
        dispatch_queue_t queue = (__bridge dispatch_queue_t)handle->capture_queue;
        if (queue) {
            dispatch_sync(queue, ^{});
        }

        /* Now safe to release AVFoundation objects */
        (void)(__bridge_transfer AVCaptureSession *)handle->session;
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

/*
 * cm_set_format — Set camera resolution + fps atomically.
 *
 * Finds the best matching AVCaptureDeviceFormat for the requested (w, h, fps).
 * Only stops/restarts the capture session when the underlying format must change.
 * If only the fps changes and the current format supports it, just adjusts
 * the frame duration without touching the session.
 */
int cm_set_format(CM_Camera *camera, int width, int height, int fps) {
    NSLog(@"[DEBUG native] cm_set_format called: %dx%d@%d", width, height, fps);

    if (!camera) {
        NSLog(@"[DEBUG native]   camera is NULL!");
        return CM_ERROR_INVALID_PARAM;
    }

    NSLog(@"[DEBUG native]   Camera: %s (index %d)", camera->display_name, camera->device_index);
    NSLog(@"[DEBUG native]   Current: %dx%d@%d", camera->width, camera->height, camera->fps);

    camera->width = width;
    camera->height = height;
    camera->fps = fps;

    /* If camera isn't open yet, just store the values */
    if (!camera->platform_handle) {
        NSLog(@"[DEBUG native]   platform_handle is NULL - camera not open yet, storing values");
        return CM_OK;
    }

    MacOSCameraHandle *handle = (MacOSCameraHandle *)camera->platform_handle;
    NSLog(@"[DEBUG native]   platform_handle: %p", handle);

    @autoreleasepool {
        AVCaptureDevice *device = (__bridge AVCaptureDevice *)handle->device;
        AVCaptureSession *session = (__bridge AVCaptureSession *)handle->session;
        NSError *error = nil;

        NSLog(@"[DEBUG native]   Device: %@ (uniqueID: %@)", device.localizedName, device.uniqueID);

        /* Check if current activeFormat already matches (w, h) and supports fps */
        AVCaptureDeviceFormat *currentFormat = device.activeFormat;
        CMVideoDimensions currentDims = CMVideoFormatDescriptionGetDimensions(currentFormat.formatDescription);
        bool current_matches_res = (currentDims.width == width && currentDims.height == height);
        bool current_supports_fps = false;

        NSLog(@"[DEBUG native]   Current activeFormat: %dx%d", currentDims.width, currentDims.height);

        AVFrameRateRange *matchingRange = nil;
        if (current_matches_res) {
            for (AVFrameRateRange *range in currentFormat.videoSupportedFrameRateRanges) {
                NSLog(@"[DEBUG native]   FPS range: %.1f - %.1f", range.minFrameRate, range.maxFrameRate);
                if (range.minFrameRate <= fps && range.maxFrameRate >= fps) {
                    current_supports_fps = true;
                    matchingRange = range;
                    break;
                }
            }
        }

        NSLog(@"[DEBUG native]   current_matches_res: %d, current_supports_fps: %d",
              current_matches_res, current_supports_fps);

        if (current_matches_res && current_supports_fps && matchingRange) {
            /* Fast path: just change frame duration, no session restart */
            NSLog(@"[DEBUG native]   FAST PATH - just changing frame duration");
            /*
             * Use the EXACT frame duration from the range, not CMTimeMake(1, fps).
             * Some cameras (especially USB cameras) require the precise CMTime values
             * they report in their supported ranges.
             */
            CMTime exactDuration = matchingRange.minFrameDuration;
            NSLog(@"[DEBUG native]   Using exact duration from range: %lld/%d",
                  exactDuration.value, exactDuration.timescale);
            if ([device lockForConfiguration:&error]) {
                device.activeVideoMinFrameDuration = exactDuration;
                device.activeVideoMaxFrameDuration = exactDuration;
                [device unlockForConfiguration];
                NSLog(@"[DEBUG native]   Frame duration set successfully");
            } else {
                NSLog(@"[DEBUG native]   lockForConfiguration failed: %@", error);
            }
            return CM_OK;
        }

        /* Slow path: need a different format — stop session, switch, restart */
        NSLog(@"[DEBUG native]   SLOW PATH - need different format");
        ring_buffer_flush(&handle->ring_buffer);
        [session stopRunning];
        NSLog(@"[DEBUG native]   Session stopped");

        /* Drain capture queue so no delegate callbacks are in flight */
        dispatch_queue_t queue = (__bridge dispatch_queue_t)handle->capture_queue;
        if (queue) {
            dispatch_sync(queue, ^{});
        }

        if (![device lockForConfiguration:&error]) {
            NSLog(@"[DEBUG native]   lockForConfiguration FAILED: %@", error);
            [session startRunning];
            return CM_ERROR_OPEN_FAILED;
        }

        /* Search for format matching (w, h, fps) */
        bool found = false;
        NSLog(@"[DEBUG native]   Searching %lu formats...", (unsigned long)[device.formats count]);
        for (AVCaptureDeviceFormat *format in device.formats) {
            CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription);
            if (dims.width != width || dims.height != height) continue;

            NSLog(@"[DEBUG native]   Found matching resolution: %dx%d", dims.width, dims.height);
            for (AVFrameRateRange *range in format.videoSupportedFrameRateRanges) {
                double fps_d = (double)fps;
                double min_fps = range.minFrameRate;
                double max_fps = range.maxFrameRate;
                NSLog(@"[DEBUG native]     FPS range: %.1f - %.1f (requested fps=%.1f)",
                      min_fps, max_fps, fps_d);
                bool min_ok = min_fps <= fps_d + 0.1;  /* Allow small floating point tolerance */
                bool max_ok = max_fps >= fps_d - 0.1;
                NSLog(@"[DEBUG native]     Check: min_ok=%d, max_ok=%d", min_ok, max_ok);
                if (min_ok && max_ok) {
                    /*
                     * Use the EXACT frame duration from the range, not CMTimeMake(1, fps).
                     * USB cameras often require precise CMTime values that match their
                     * supported ranges exactly (e.g., 1000000/30000030 not 1/30).
                     */
                    CMTime exactDuration = range.minFrameDuration;
                    NSLog(@"[DEBUG native]     Setting activeFormat with exact duration: %lld/%d",
                          exactDuration.value, exactDuration.timescale);
                    device.activeFormat = format;
                    device.activeVideoMinFrameDuration = exactDuration;
                    device.activeVideoMaxFrameDuration = exactDuration;
                    found = true;
                    break;
                } else {
                    NSLog(@"[DEBUG native]     Does NOT match");
                }
            }
            if (found) break;
        }

        NSLog(@"[DEBUG native]   Format found: %d", found);

        /*
         * On macOS, keep device locked while starting the session to
         * prevent the session from overriding our activeFormat choice.
         */
        [session startRunning];
        [device unlockForConfiguration];
        NSLog(@"[DEBUG native]   Session restarted");

        if (!found) {
            return CM_ERROR_NOT_SUPPORTED;
        }

        /*
         * Wait for frames at the new resolution.
         * After a format change, the ring buffer was flushed and frame_count
         * was preserved. We need to wait for NEW frames (frame_count to increase
         * beyond what it was before the session restarted).
         *
         * The flush cleared pixels but kept frame_count. We'll get the current
         * count and wait for it to increase, which indicates a new valid frame.
         */
        int current_count = handle->ring_buffer.frame_count;
        NSLog(@"[DEBUG native]   Waiting for frames at new resolution (current count: %d)...", current_count);

        /* Wait for at least 2 new frames (ensures we skip any stale frames) */
        int waited_count = ring_buffer_wait_for_frame(&handle->ring_buffer, current_count + 1, 500);
        NSLog(@"[DEBUG native]   Wait complete (count now: %d)", waited_count);

        return CM_OK;
    }
}

int cm_set_resolution(CM_Camera *camera, int width, int height) {
    if (!camera) return CM_ERROR_INVALID_PARAM;
    return cm_set_format(camera, width, height, camera->fps);
}

int cm_set_fps(CM_Camera *camera, int fps) {
    if (!camera) return CM_ERROR_INVALID_PARAM;
    return cm_set_format(camera, camera->width, camera->height, fps);
}

/* ============================================================================
 * UVC Exposure Control via IOKit
 *
 * Since AVFoundation doesn't support manual exposure for external USB cameras
 * on macOS, we use IOKit to send UVC (USB Video Class) control requests
 * directly to the camera hardware.
 *
 * The workflow:
 * 1. Find the IOUSBHostDevice matching our camera's uniqueID
 * 2. Get the control interface (interface 0, typically the Video Control interface)
 * 3. Send USB control requests to set exposure parameters
 * ============================================================================ */

/* UVC control request structure */
typedef struct {
    uint8_t  bmRequestType;
    uint8_t  bRequest;
    uint16_t wValue;       /* Control selector << 8 */
    uint16_t wIndex;       /* Interface number | (Unit ID << 8) */
    uint16_t wLength;
} UVCControlRequest;

/*
 * Send a UVC control request to a USB device via IOKit.
 * Returns true on success, false on failure.
 */
static bool send_uvc_control(io_service_t usbDevice, uint8_t request, uint8_t selector,
                             uint8_t unit, uint8_t interface, uint8_t *data, uint16_t length) {
    IOCFPlugInInterface **plugInInterface = NULL;
    IOUSBDeviceInterface **deviceInterface = NULL;
    SInt32 score;
    kern_return_t kr;
    IOReturn result;
    bool success = false;

    /* Create plugin interface for USB device */
    kr = IOCreatePlugInInterfaceForService(usbDevice,
                                            kIOUSBDeviceUserClientTypeID,
                                            kIOCFPlugInInterfaceID,
                                            &plugInInterface,
                                            &score);
    if (kr != KERN_SUCCESS || !plugInInterface) {
        NSLog(@"[UVC] Failed to create plugin interface: %x", kr);
        return false;
    }

    /* Get USB device interface */
    HRESULT hr = (*plugInInterface)->QueryInterface(plugInInterface,
                                                     CFUUIDGetUUIDBytes(kIOUSBDeviceInterfaceID),
                                                     (LPVOID *)&deviceInterface);
    (*plugInInterface)->Release(plugInInterface);

    if (hr != 0 || !deviceInterface) {
        NSLog(@"[UVC] Failed to get device interface: %x", (int)hr);
        return false;
    }

    /* Open device */
    result = (*deviceInterface)->USBDeviceOpen(deviceInterface);
    if (result != kIOReturnSuccess) {
        NSLog(@"[UVC] Failed to open device: %x", result);
        (*deviceInterface)->Release(deviceInterface);
        return false;
    }

    /*
     * Build UVC control request:
     * - bmRequestType: 0x21 for SET (host-to-device, class, interface)
     *                  0xA1 for GET (device-to-host, class, interface)
     * - bRequest: UVC_SET_CUR (0x01) or UVC_GET_* (0x81-0x87)
     * - wValue: (Control Selector << 8) | 0x00
     * - wIndex: (Unit ID << 8) | Interface Number
     */
    IOUSBDevRequest devRequest;
    devRequest.bmRequestType = (request & 0x80) ? 0xA1 : 0x21;  /* GET vs SET */
    devRequest.bRequest = request;
    devRequest.wValue = (uint16_t)((selector << 8) | 0x00);
    devRequest.wIndex = (uint16_t)((unit << 8) | interface);
    devRequest.wLength = length;
    devRequest.pData = data;
    devRequest.wLenDone = 0;

    result = (*deviceInterface)->DeviceRequest(deviceInterface, &devRequest);
    if (result == kIOReturnSuccess) {
        success = true;
        NSLog(@"[UVC] Control request succeeded: selector=0x%02x, unit=%d, len=%d",
              selector, unit, length);
    } else {
        NSLog(@"[UVC] Control request failed: %x (selector=0x%02x, unit=%d)",
              result, selector, unit);
    }

    (*deviceInterface)->USBDeviceClose(deviceInterface);
    (*deviceInterface)->Release(deviceInterface);

    return success;
}

/*
 * Find the IOKit USB device corresponding to an AVFoundation camera.
 *
 * AVFoundation uniqueID format for USB cameras:
 *   "0xLLLLLLLLVVVVPPPP" where:
 *   - LLLLLLLL = location ID (but byte-swapped/encoded oddly)
 *   - VVVV = vendor ID
 *   - PPPP = product ID
 *
 * Example: "0x11000000525a4b1"
 *   - This encodes location + vendor 0x0525 + product 0xa4b1
 *
 * We match by extracting vendor/product from the serial and comparing.
 * Returns 0 if not found.
 */
static io_service_t find_usb_device_for_camera(const char *camera_serial) {
    io_service_t usbDevice = 0;
    io_iterator_t iterator;
    kern_return_t kr;

    NSString *serialStr = [NSString stringWithUTF8String:camera_serial];
    bool isUSBCamera = [serialStr hasPrefix:@"0x"];

    if (!isUSBCamera) {
        /* Built-in camera - AVFoundation handles it, no UVC needed */
        return 0;
    }

    /* Parse the hex value from AVFoundation uniqueID */
    unsigned long long avfID = 0;
    NSScanner *scanner = [NSScanner scannerWithString:serialStr];
    [scanner scanHexLongLong:&avfID];

    /*
     * Extract vendor and product ID from AVFoundation uniqueID.
     * The format appears to be: location(32) | vendor(16) | product(16)
     * But the encoding varies. Let's try to extract the lower bits.
     *
     * For "0x11000000525a4b1" = 0x011000000525a4b1:
     *   vendor = 0x0525, product = 0xa4b1
     */
    uint16_t target_vendor = (uint16_t)((avfID >> 16) & 0xFFFF);
    uint16_t target_product = (uint16_t)(avfID & 0xFFFF);

    NSLog(@"[UVC] Looking for USB device: vendor=0x%04x product=0x%04x (from AVF ID 0x%llx)",
          target_vendor, target_product, avfID);

    /* Create matching dictionary for USB devices */
    CFMutableDictionaryRef matchingDict = IOServiceMatching(kIOUSBDeviceClassName);
    if (!matchingDict) {
        NSLog(@"[UVC] Failed to create USB matching dictionary");
        return 0;
    }

    kr = IOServiceGetMatchingServices(kIOMainPortDefault, matchingDict, &iterator);
    if (kr != KERN_SUCCESS) {
        NSLog(@"[UVC] Failed to get USB services: %x", kr);
        return 0;
    }

    /* Iterate USB devices looking for matching vendor/product */
    io_service_t service;
    while ((service = IOIteratorNext(iterator))) {
        /* Get vendor ID */
        CFNumberRef vendorRef = (CFNumberRef)IORegistryEntryCreateCFProperty(
            service, CFSTR("idVendor"), kCFAllocatorDefault, 0);
        CFNumberRef productRef = (CFNumberRef)IORegistryEntryCreateCFProperty(
            service, CFSTR("idProduct"), kCFAllocatorDefault, 0);

        if (vendorRef && productRef) {
            int vendor = 0, product = 0;
            CFNumberGetValue(vendorRef, kCFNumberIntType, &vendor);
            CFNumberGetValue(productRef, kCFNumberIntType, &product);

            /* Get product name for logging */
            CFStringRef nameRef = (CFStringRef)IORegistryEntryCreateCFProperty(
                service, CFSTR("USB Product Name"), kCFAllocatorDefault, 0);

            NSLog(@"[UVC]   Checking device: vendor=0x%04x product=0x%04x name=%@",
                  vendor, product, nameRef ? (__bridge NSString *)nameRef : @"(unknown)");

            if (nameRef) CFRelease(nameRef);

            /* Match on vendor and product ID */
            if (vendor == target_vendor && product == target_product) {
                CFStringRef matchName = (CFStringRef)IORegistryEntryCreateCFProperty(
                    service, CFSTR("USB Product Name"), kCFAllocatorDefault, 0);
                NSLog(@"[UVC] Found matching USB device: %@", matchName ? (__bridge NSString *)matchName : @"(unknown)");
                if (matchName) CFRelease(matchName);

                CFRelease(vendorRef);
                CFRelease(productRef);
                usbDevice = service;
                break;
            }
        }

        if (vendorRef) CFRelease(vendorRef);
        if (productRef) CFRelease(productRef);
        IOObjectRelease(service);
    }

    IOObjectRelease(iterator);

    if (usbDevice == 0) {
        NSLog(@"[UVC] No matching USB device found for %s", camera_serial);
    }

    return usbDevice;
}

/*
 * Set UVC auto-exposure mode.
 * mode: UVC_AE_MODE_MANUAL (0x01), UVC_AE_MODE_AUTO (0x02), etc.
 */
static bool set_uvc_auto_exposure_mode(io_service_t usbDevice, uint8_t mode) {
    uint8_t data[1] = { mode };
    return send_uvc_control(usbDevice, UVC_SET_CUR, UVC_CT_AE_MODE_CONTROL,
                            UVC_INPUT_TERMINAL_ID, UVC_INTERFACE_CONTROL, data, 1);
}

/*
 * Set UVC absolute exposure time.
 * The value is in 100µs units (e.g., 100 = 10ms exposure).
 */
static bool set_uvc_exposure_time(io_service_t usbDevice, uint32_t exposure_100us) {
    uint8_t data[4];
    data[0] = (uint8_t)(exposure_100us & 0xFF);
    data[1] = (uint8_t)((exposure_100us >> 8) & 0xFF);
    data[2] = (uint8_t)((exposure_100us >> 16) & 0xFF);
    data[3] = (uint8_t)((exposure_100us >> 24) & 0xFF);
    return send_uvc_control(usbDevice, UVC_SET_CUR, UVC_CT_EXPOSURE_TIME_ABSOLUTE_CONTROL,
                            UVC_INPUT_TERMINAL_ID, UVC_INTERFACE_CONTROL, data, 4);
}

/*
 * Get UVC exposure time range (min, max, current).
 */
static bool get_uvc_exposure_range(io_service_t usbDevice, uint32_t *out_min,
                                    uint32_t *out_max, uint32_t *out_current) {
    uint8_t data[4] = {0};
    bool ok = true;

    if (out_min) {
        ok = send_uvc_control(usbDevice, UVC_GET_MIN, UVC_CT_EXPOSURE_TIME_ABSOLUTE_CONTROL,
                              UVC_INPUT_TERMINAL_ID, UVC_INTERFACE_CONTROL, data, 4);
        if (ok) *out_min = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
    }

    if (out_max && ok) {
        ok = send_uvc_control(usbDevice, UVC_GET_MAX, UVC_CT_EXPOSURE_TIME_ABSOLUTE_CONTROL,
                              UVC_INPUT_TERMINAL_ID, UVC_INTERFACE_CONTROL, data, 4);
        if (ok) *out_max = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
    }

    if (out_current && ok) {
        ok = send_uvc_control(usbDevice, UVC_GET_CUR, UVC_CT_EXPOSURE_TIME_ABSOLUTE_CONTROL,
                              UVC_INPUT_TERMINAL_ID, UVC_INTERFACE_CONTROL, data, 4);
        if (ok) *out_current = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
    }

    return ok;
}

/*
 * Convert GUI exposure value (log2 scale, -7 to 0 typical) to UVC exposure time.
 * GUI value is log2(seconds), e.g.:
 *   -7 = 1/128s ≈ 7.8ms
 *   -4 = 1/16s ≈ 62.5ms
 *   -1 = 0.5s
 *    0 = 1s
 *
 * UVC uses 100µs units, so:
 *   exposure_100us = 2^gui_value * 1000000 / 100 = 2^gui_value * 10000
 */
static uint32_t gui_exposure_to_uvc(int gui_exposure) {
    double seconds = pow(2.0, (double)gui_exposure);
    uint32_t exposure_100us = (uint32_t)(seconds * 10000.0);
    /* Clamp to reasonable range: 1 (100µs) to 100000 (10s) */
    if (exposure_100us < 1) exposure_100us = 1;
    if (exposure_100us > 100000) exposure_100us = 100000;
    return exposure_100us;
}

int cm_set_exposure(CM_Camera *camera, int exposure) {
    if (!camera) return CM_ERROR_INVALID_PARAM;
    camera->exposure = exposure;

    /*
     * Try UVC control for USB cameras.
     * AVFoundation doesn't support exposure on macOS, but we can send
     * UVC control requests directly to USB webcams via IOKit.
     */
    io_service_t usbDevice = find_usb_device_for_camera(camera->serial_number);
    if (usbDevice == 0) {
        /* Not a USB camera or couldn't find it - likely built-in, nothing we can do */
        NSLog(@"[UVC] Camera %s: not a USB camera or not found, skipping exposure control",
              camera->serial_number);
        return CM_OK;
    }

    /* First, disable auto-exposure to allow manual control */
    if (!set_uvc_auto_exposure_mode(usbDevice, UVC_AE_MODE_MANUAL)) {
        NSLog(@"[UVC] Failed to disable auto-exposure for %s", camera->serial_number);
        /* Continue anyway - some cameras may not support AE mode control */
    }

    /* Get exposure range for logging/debugging */
    uint32_t exp_min = 0, exp_max = 0, exp_current = 0;
    if (get_uvc_exposure_range(usbDevice, &exp_min, &exp_max, &exp_current)) {
        NSLog(@"[UVC] Exposure range: min=%u, max=%u, current=%u (100µs units)",
              exp_min, exp_max, exp_current);
    }

    /* Convert GUI value to UVC exposure time and set it */
    uint32_t uvc_exposure = gui_exposure_to_uvc(exposure);
    NSLog(@"[UVC] Setting exposure: GUI=%d -> UVC=%u (100µs units, %.2fms)",
          exposure, uvc_exposure, uvc_exposure * 0.1);

    if (!set_uvc_exposure_time(usbDevice, uvc_exposure)) {
        NSLog(@"[UVC] Failed to set exposure time for %s", camera->serial_number);
        IOObjectRelease(usbDevice);
        return CM_ERROR_NOT_SUPPORTED;
    }

    IOObjectRelease(usbDevice);
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
