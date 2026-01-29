/*
 * calimerge_win32.cpp
 *
 * Windows implementation using Media Foundation.
 *
 * Design: Each camera runs continuous capture in its own thread.
 * Frames store BOTH their native camera timestamp (MF sample PTS) and
 * an arrival timestamp from a common clock (QueryPerformanceCounter).
 *
 * Clock Synchronization Strategy:
 * - Each camera has its own clock domain for PTS timestamps
 * - We measure the offset between each camera's clock and a common reference
 * - When synchronizing frames, we apply offsets to compare camera timestamps
 * - This preserves the camera's native timing while enabling cross-camera sync
 */

#include "calimerge_platform.h"

#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <wchar.h>

#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "ole32.lib")

/* ============================================================================
 * QPC Timestamp (global, thread-safe after first call)
 * ============================================================================ */

static LARGE_INTEGER g_qpc_freq = {0};

static uint64_t get_timestamp_ns(void) {
    if (g_qpc_freq.QuadPart == 0) {
        QueryPerformanceFrequency(&g_qpc_freq);
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (uint64_t)((now.QuadPart * 1000000000ULL) / g_qpc_freq.QuadPart);
}

/* ============================================================================
 * Ring Buffer for Frame Storage
 * ============================================================================ */

#define RING_BUFFER_SIZE 8  /* Keep last N frames per camera */

typedef struct {
    uint8_t *pixels;
    int width, height;
    int stride;
    uint64_t camera_pts_ns;     /* Camera's own timestamp (from MF sample PTS) */
    uint64_t arrival_ns;        /* Common clock timestamp when frame arrived */
    bool valid;
} RingFrame;

typedef struct {
    RingFrame frames[RING_BUFFER_SIZE];
    int write_index;        /* Next slot to write */
    int frame_count;        /* Total frames written (for detecting new frames) */
    CRITICAL_SECTION cs;
    CONDITION_VARIABLE cv;  /* Signaled when new frame arrives */
} FrameRingBuffer;

static void ring_buffer_init(FrameRingBuffer *rb) {
    memset(rb, 0, sizeof(FrameRingBuffer));
    InitializeCriticalSection(&rb->cs);
    InitializeConditionVariable(&rb->cv);
}

static void ring_buffer_destroy(FrameRingBuffer *rb) {
    EnterCriticalSection(&rb->cs);
    for (int i = 0; i < RING_BUFFER_SIZE; i++) {
        free(rb->frames[i].pixels);
        rb->frames[i].pixels = NULL;
    }
    LeaveCriticalSection(&rb->cs);
    DeleteCriticalSection(&rb->cs);
}

/* Push a new frame (caller provides allocated pixels, ring buffer takes ownership) */
static void ring_buffer_push(FrameRingBuffer *rb, uint8_t *pixels, int width, int height,
                             uint64_t camera_pts_ns, uint64_t arrival_ns) {
    EnterCriticalSection(&rb->cs);

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
    WakeAllConditionVariable(&rb->cv);
    LeaveCriticalSection(&rb->cs);
}

/* Get the most recent frame (copies data, caller must free pixels) */
static bool ring_buffer_get_latest(FrameRingBuffer *rb, CM_Frame *out, int64_t clock_offset_ns) {
    EnterCriticalSection(&rb->cs);

    if (rb->frame_count == 0) {
        LeaveCriticalSection(&rb->cs);
        return false;
    }

    /* Most recent is at (write_index - 1 + SIZE) % SIZE */
    int idx = (rb->write_index - 1 + RING_BUFFER_SIZE) % RING_BUFFER_SIZE;
    RingFrame *rf = &rb->frames[idx];

    if (!rf->valid || !rf->pixels) {
        LeaveCriticalSection(&rb->cs);
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

    LeaveCriticalSection(&rb->cs);
    return true;
}

/*
 * Get frame closest to target timestamp using CORRECTED PTS.
 * The target_common_ns is in the common clock domain.
 * The clock_offset_ns converts this camera's PTS to common domain.
 */
static bool ring_buffer_get_closest_corrected(
    FrameRingBuffer *rb,
    uint64_t target_common_ns,
    int64_t clock_offset_ns,
    CM_Frame *out
) {
    EnterCriticalSection(&rb->cs);

    if (rb->frame_count == 0) {
        LeaveCriticalSection(&rb->cs);
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
        LeaveCriticalSection(&rb->cs);
        return false;
    }

    RingFrame *rf = &rb->frames[best_idx];
    int size = rf->width * rf->height * 3;
    out->pixels = (uint8_t *)malloc(size);
    memcpy(out->pixels, rf->pixels, size);
    out->width = rf->width;
    out->height = rf->height;
    out->stride = rf->stride;
    out->timestamp_ns = rf->camera_pts_ns;
    out->arrival_ns = rf->arrival_ns;
    out->corrected_ns = (uint64_t)((int64_t)rf->camera_pts_ns + clock_offset_ns);

    LeaveCriticalSection(&rb->cs);
    return true;
}

/* Get latest arrival timestamp (common clock domain) without copying frame */
static uint64_t ring_buffer_get_latest_arrival(FrameRingBuffer *rb) {
    EnterCriticalSection(&rb->cs);

    if (rb->frame_count == 0) {
        LeaveCriticalSection(&rb->cs);
        return 0;
    }

    int idx = (rb->write_index - 1 + RING_BUFFER_SIZE) % RING_BUFFER_SIZE;
    uint64_t ts = rb->frames[idx].arrival_ns;

    LeaveCriticalSection(&rb->cs);
    return ts;
}

/* Get latest camera PTS timestamp (camera's native clock) */
static uint64_t ring_buffer_get_latest_camera_pts(FrameRingBuffer *rb) {
    EnterCriticalSection(&rb->cs);

    if (rb->frame_count == 0) {
        LeaveCriticalSection(&rb->cs);
        return 0;
    }

    int idx = (rb->write_index - 1 + RING_BUFFER_SIZE) % RING_BUFFER_SIZE;
    uint64_t ts = rb->frames[idx].camera_pts_ns;

    LeaveCriticalSection(&rb->cs);
    return ts;
}

/* Get both timestamps from the latest frame (for clock offset calculation) */
static bool ring_buffer_get_latest_timestamps(FrameRingBuffer *rb, uint64_t *out_camera_pts, uint64_t *out_arrival) {
    EnterCriticalSection(&rb->cs);

    if (rb->frame_count == 0) {
        LeaveCriticalSection(&rb->cs);
        return false;
    }

    int idx = (rb->write_index - 1 + RING_BUFFER_SIZE) % RING_BUFFER_SIZE;
    RingFrame *rf = &rb->frames[idx];

    if (!rf->valid) {
        LeaveCriticalSection(&rb->cs);
        return false;
    }

    *out_camera_pts = rf->camera_pts_ns;
    *out_arrival = rf->arrival_ns;

    LeaveCriticalSection(&rb->cs);
    return true;
}

/* Wait for a new frame with timeout (returns frame count at return) */
static int ring_buffer_wait_for_frame(FrameRingBuffer *rb, int last_count, int timeout_ms) {
    EnterCriticalSection(&rb->cs);

    if (rb->frame_count > last_count) {
        int count = rb->frame_count;
        LeaveCriticalSection(&rb->cs);
        return count;
    }

    SleepConditionVariableCS(&rb->cv, &rb->cs, (DWORD)timeout_ms);
    int count = rb->frame_count;
    LeaveCriticalSection(&rb->cs);
    return count;
}

/* ============================================================================
 * Pixel Format Conversion
 * ============================================================================ */

/* Clamp integer to [0, 255] */
static inline uint8_t clamp_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

/* BGRA -> BGR (same as macOS, strip alpha channel) */
static void convert_bgra_to_bgr(const uint8_t *src, uint8_t *dst, int width, int height, int src_stride) {
    for (int y = 0; y < height; y++) {
        const uint8_t *src_row = src + y * src_stride;
        uint8_t *dst_row = dst + y * width * 3;

        int x = 0;
        /* Process 4 pixels at a time */
        for (; x + 4 <= width; x += 4) {
            /* Pixel 0 */
            dst_row[0]  = src_row[0];   /* B */
            dst_row[1]  = src_row[1];   /* G */
            dst_row[2]  = src_row[2];   /* R */
            /* Pixel 1 */
            dst_row[3]  = src_row[4];
            dst_row[4]  = src_row[5];
            dst_row[5]  = src_row[6];
            /* Pixel 2 */
            dst_row[6]  = src_row[8];
            dst_row[7]  = src_row[9];
            dst_row[8]  = src_row[10];
            /* Pixel 3 */
            dst_row[9]  = src_row[12];
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
}

/*
 * NV12 -> BGR
 * NV12: Y plane (width*height bytes), then UV plane (width*height/2 bytes, interleaved U/V)
 * Each 2x2 block of Y pixels shares one U and one V value.
 */
static void convert_nv12_to_bgr(const uint8_t *src, uint8_t *dst, int width, int height, int src_stride) {
    const uint8_t *y_plane = src;
    const uint8_t *uv_plane = src + src_stride * height;

    for (int row = 0; row < height; row++) {
        const uint8_t *y_row = y_plane + row * src_stride;
        const uint8_t *uv_row = uv_plane + (row / 2) * src_stride;
        uint8_t *dst_row = dst + row * width * 3;

        for (int col = 0; col < width; col++) {
            int Y = y_row[col];
            int U = uv_row[(col & ~1)] - 128;
            int V = uv_row[(col & ~1) + 1] - 128;

            /* YUV -> BGR (BT.601 coefficients) */
            int R = Y + ((1436 * V) >> 10);
            int G = Y - ((352 * U + 731 * V) >> 10);
            int B = Y + ((1815 * U) >> 10);

            dst_row[col * 3 + 0] = clamp_u8(B);
            dst_row[col * 3 + 1] = clamp_u8(G);
            dst_row[col * 3 + 2] = clamp_u8(R);
        }
    }
}

/*
 * YUY2 -> BGR
 * YUY2 (packed): Y0 U0 Y1 V0 Y2 U1 Y3 V1 ...
 * Each pair of Y pixels shares one U and one V value.
 */
static void convert_yuy2_to_bgr(const uint8_t *src, uint8_t *dst, int width, int height, int src_stride) {
    for (int row = 0; row < height; row++) {
        const uint8_t *src_row = src + row * src_stride;
        uint8_t *dst_row = dst + row * width * 3;

        for (int col = 0; col < width; col += 2) {
            int Y0 = src_row[col * 2 + 0];
            int U  = src_row[col * 2 + 1] - 128;
            int Y1 = src_row[col * 2 + 2];
            int V  = src_row[col * 2 + 3] - 128;

            /* YUV -> BGR (BT.601 coefficients) */
            int R0 = Y0 + ((1436 * V) >> 10);
            int G0 = Y0 - ((352 * U + 731 * V) >> 10);
            int B0 = Y0 + ((1815 * U) >> 10);

            dst_row[col * 3 + 0] = clamp_u8(B0);
            dst_row[col * 3 + 1] = clamp_u8(G0);
            dst_row[col * 3 + 2] = clamp_u8(R0);

            if (col + 1 < width) {
                int R1 = Y1 + ((1436 * V) >> 10);
                int G1 = Y1 - ((352 * U + 731 * V) >> 10);
                int B1 = Y1 + ((1815 * U) >> 10);

                dst_row[(col + 1) * 3 + 0] = clamp_u8(B1);
                dst_row[(col + 1) * 3 + 1] = clamp_u8(G1);
                dst_row[(col + 1) * 3 + 2] = clamp_u8(R1);
            }
        }
    }
}

/* Generic conversion dispatcher */
static uint8_t *convert_to_bgr(const uint8_t *src, int width, int height,
                                int src_stride, const GUID *format) {
    uint8_t *bgr = (uint8_t *)malloc(width * height * 3);
    if (!bgr) return NULL;

    if (IsEqualGUID(*format, MFVideoFormat_RGB32)) {
        convert_bgra_to_bgr(src, bgr, width, height, src_stride);
    } else if (IsEqualGUID(*format, MFVideoFormat_NV12)) {
        convert_nv12_to_bgr(src, bgr, width, height, src_stride);
    } else if (IsEqualGUID(*format, MFVideoFormat_YUY2)) {
        convert_yuy2_to_bgr(src, bgr, width, height, src_stride);
    } else {
        /* Unsupported format - fill with gray */
        memset(bgr, 128, width * height * 3);
    }

    return bgr;
}

/* ============================================================================
 * Serial Number Extraction
 * ============================================================================ */

/*
 * Extract serial number from a Windows device symbolic link.
 *
 * USB symbolic links have the format:
 *   \\?\usb#vid_XXXX&pid_XXXX#SERIAL#{GUID}
 *
 * We extract the SERIAL portion between the 2nd and 3rd '#' delimiters.
 * If the device doesn't have a USB serial (e.g., integrated webcam),
 * we fall back to using a hash of the full symbolic link path.
 */
static void extract_serial_from_symbolic_link(const wchar_t *symlink,
                                               char *out_serial, int max_len) {
    if (!symlink || !out_serial || max_len <= 0) return;

    /* Find '#' delimiters */
    const wchar_t *hash1 = wcschr(symlink, L'#');
    const wchar_t *hash2 = hash1 ? wcschr(hash1 + 1, L'#') : NULL;
    const wchar_t *hash3 = hash2 ? wcschr(hash2 + 1, L'#') : NULL;

    if (hash2 && hash3 && (hash3 - hash2 - 1) > 0) {
        /* Extract serial between 2nd and 3rd '#' */
        int serial_len = (int)(hash3 - hash2 - 1);
        if (serial_len >= max_len) serial_len = max_len - 1;

        for (int i = 0; i < serial_len; i++) {
            wchar_t wc = hash2[1 + i];
            out_serial[i] = (wc < 128) ? (char)wc : '?';
        }
        out_serial[serial_len] = '\0';

        /* Check if serial is actually a serial (not just "&0" or similar junk) */
        if (serial_len > 2 && out_serial[0] != '&') {
            return;  /* Good serial */
        }
    }

    /* Fallback: use full symbolic link as identifier (truncated) */
    int len = (int)wcslen(symlink);
    if (len >= max_len) len = max_len - 1;

    for (int i = 0; i < len; i++) {
        wchar_t wc = symlink[i];
        /* Replace non-ASCII and path separators with underscores */
        if (wc < 32 || wc > 126 || wc == '\\' || wc == '?') {
            out_serial[i] = '_';
        } else {
            out_serial[i] = (char)wc;
        }
    }
    out_serial[len] = '\0';
}

/* ============================================================================
 * Platform-Specific Handle
 * ============================================================================ */

typedef struct {
    /* Media Foundation objects */
    IMFSourceReader *source_reader;
    IMFMediaSource  *media_source;
    IMFActivate     *activate;        /* Keep for cleanup */

    /* Capture thread */
    HANDLE           capture_thread;
    volatile bool    capture_running;

    /* Ring buffer for continuous capture */
    FrameRingBuffer  ring_buffer;

    /* Clock synchronization */
    int64_t          clock_offset_ns;
    bool             clock_offset_valid;

    /* Native pixel format from camera */
    GUID             native_format;

    /* State */
    bool             is_open;
    int              camera_index;
    int              width, height;
} Win32CameraHandle;

/* ============================================================================
 * Clock Offset Calibration
 * ============================================================================ */

#define CLOCK_OFFSET_SAMPLES 10

static void calibrate_clock_offset(Win32CameraHandle *handle) {
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
            offsets[sample_count++] = (int64_t)arrival - (int64_t)camera_pts;
        }
    }

    if (sample_count < 3) {
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
 * Capture Thread
 * ============================================================================ */

static DWORD WINAPI capture_thread_proc(LPVOID param) {
    Win32CameraHandle *handle = (Win32CameraHandle *)param;

    /* Per-thread COM initialization */
    HRESULT hr_com = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    bool com_initialized = SUCCEEDED(hr_com);

    while (handle->capture_running) {
        IMFSample *sample = NULL;
        DWORD stream_index = 0;
        DWORD flags = 0;
        LONGLONG timestamp_100ns = 0;

        HRESULT hr = handle->source_reader->ReadSample(
            (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM,
            0,              /* No flags */
            &stream_index,
            &flags,
            &timestamp_100ns,
            &sample
        );

        if (FAILED(hr) || !sample) {
            if (sample) sample->Release();
            if (!handle->capture_running) break;
            continue;
        }

        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
            sample->Release();
            break;
        }

        uint64_t arrival_ns = get_timestamp_ns();
        /* MF timestamps are in 100-nanosecond units */
        uint64_t camera_pts_ns = (uint64_t)(timestamp_100ns * 100);

        /* Extract pixel data */
        IMFMediaBuffer *buffer = NULL;
        hr = sample->ConvertToContiguousBuffer(&buffer);
        if (FAILED(hr) || !buffer) {
            sample->Release();
            continue;
        }

        BYTE *raw_data = NULL;
        DWORD max_len = 0, cur_len = 0;
        hr = buffer->Lock(&raw_data, &max_len, &cur_len);
        if (FAILED(hr) || !raw_data) {
            buffer->Release();
            sample->Release();
            continue;
        }

        /* Compute source stride */
        int src_stride;
        if (IsEqualGUID(handle->native_format, MFVideoFormat_RGB32)) {
            src_stride = handle->width * 4;
        } else if (IsEqualGUID(handle->native_format, MFVideoFormat_NV12)) {
            src_stride = handle->width;  /* Y plane stride = width */
        } else if (IsEqualGUID(handle->native_format, MFVideoFormat_YUY2)) {
            src_stride = handle->width * 2;
        } else {
            src_stride = handle->width * 4;  /* Default assumption */
        }

        /* Convert to BGR */
        uint8_t *bgr = convert_to_bgr(raw_data, handle->width, handle->height,
                                        src_stride, &handle->native_format);

        buffer->Unlock();
        buffer->Release();
        sample->Release();

        if (bgr) {
            /* Push to ring buffer (takes ownership of bgr) */
            ring_buffer_push(&handle->ring_buffer, bgr, handle->width, handle->height,
                             camera_pts_ns, arrival_ns);
        }
    }

    if (com_initialized) {
        CoUninitialize();
    }

    return 0;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

static bool g_initialized = false;
static bool g_com_initialized = false;

int cm_init(void) {
    if (g_initialized) return CM_OK;

    /* Initialize QPC frequency */
    if (g_qpc_freq.QuadPart == 0) {
        QueryPerformanceFrequency(&g_qpc_freq);
    }

    /*
     * Initialize COM for Media Foundation.
     * PySide6 may have already initialized COM as STA on the main thread.
     * Handle RPC_E_CHANGED_MODE gracefully - we can still use MF.
     */
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (SUCCEEDED(hr)) {
        g_com_initialized = true;
    } else if (hr == RPC_E_CHANGED_MODE) {
        /* COM already initialized with different threading model (STA from Qt/PySide6) */
        /* We can still use MF, capture threads will init their own COM */
        g_com_initialized = false;
    } else {
        return CM_ERROR_INIT_FAILED;
    }

    /* Start Media Foundation */
    hr = MFStartup(MF_VERSION);
    if (FAILED(hr)) {
        if (g_com_initialized) CoUninitialize();
        return CM_ERROR_INIT_FAILED;
    }

    g_initialized = true;
    return CM_OK;
}

void cm_shutdown(void) {
    if (!g_initialized) return;

    MFShutdown();

    if (g_com_initialized) {
        CoUninitialize();
        g_com_initialized = false;
    }

    g_initialized = false;
}

/* ============================================================================
 * Camera Enumeration
 * ============================================================================ */

/*
 * Helper: Find the best matching media type for a given resolution.
 * Returns the index of the media type, or -1 if not found.
 */
static int find_media_type_for_resolution(IMFSourceReader *reader, int target_w, int target_h, GUID *out_format) {
    int best_index = -1;

    for (DWORD i = 0; ; i++) {
        IMFMediaType *type = NULL;
        HRESULT hr = reader->GetNativeMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, i, &type);
        if (FAILED(hr)) break;

        UINT32 w = 0, h = 0;
        MFGetAttributeSize(type, MF_MT_FRAME_SIZE, &w, &h);

        if ((int)w == target_w && (int)h == target_h) {
            GUID subtype = {0};
            type->GetGUID(MF_MT_SUBTYPE, &subtype);

            /* Prefer RGB32, then NV12, then YUY2 */
            if (IsEqualGUID(subtype, MFVideoFormat_RGB32)) {
                if (out_format) *out_format = subtype;
                best_index = (int)i;
                type->Release();
                break;  /* Best possible */
            } else if (IsEqualGUID(subtype, MFVideoFormat_NV12) ||
                       IsEqualGUID(subtype, MFVideoFormat_YUY2)) {
                if (best_index < 0) {
                    if (out_format) *out_format = subtype;
                    best_index = (int)i;
                }
            }
        }

        type->Release();
    }

    return best_index;
}

int cm_enumerate_cameras(CM_Camera *out_cameras, int max_cameras) {
    if (!g_initialized) {
        if (cm_init() != CM_OK) return CM_ERROR_INIT_FAILED;
    }
    if (!out_cameras || max_cameras <= 0) return CM_ERROR_INVALID_PARAM;

    /* Create attribute store for video capture devices */
    IMFAttributes *attrs = NULL;
    HRESULT hr = MFCreateAttributes(&attrs, 1);
    if (FAILED(hr)) return CM_ERROR_INIT_FAILED;

    hr = attrs->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                         MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    if (FAILED(hr)) {
        attrs->Release();
        return CM_ERROR_INIT_FAILED;
    }

    /* Enumerate video capture devices */
    IMFActivate **devices = NULL;
    UINT32 device_count = 0;
    hr = MFEnumDeviceSources(attrs, &devices, &device_count);
    attrs->Release();

    if (FAILED(hr)) return CM_ERROR_INIT_FAILED;

    int count = 0;

    for (UINT32 d = 0; d < device_count && count < max_cameras; d++) {
        CM_Camera *cam = &out_cameras[count];
        memset(cam, 0, sizeof(CM_Camera));

        /* Get display name */
        WCHAR *friendly_name = NULL;
        UINT32 name_len = 0;
        hr = devices[d]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
                                             &friendly_name, &name_len);
        if (SUCCEEDED(hr) && friendly_name) {
            WideCharToMultiByte(CP_UTF8, 0, friendly_name, -1,
                                cam->display_name, CM_NAME_LEN - 1, NULL, NULL);
            CoTaskMemFree(friendly_name);
        } else {
            strncpy(cam->display_name, "Unknown Camera", CM_NAME_LEN - 1);
        }

        /* Get symbolic link for serial number extraction */
        WCHAR *symlink = NULL;
        UINT32 symlink_len = 0;
        hr = devices[d]->GetAllocatedString(
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
            &symlink, &symlink_len);
        if (SUCCEEDED(hr) && symlink) {
            extract_serial_from_symbolic_link(symlink, cam->serial_number, CM_SERIAL_LEN);
            CoTaskMemFree(symlink);
        } else {
            _snprintf(cam->serial_number, CM_SERIAL_LEN - 1, "cam_%d", d);
        }

        cam->device_index = count;
        cam->enabled = true;
        cam->platform_handle = NULL;

        /* Probe supported resolutions by temporarily activating the source */
        IMFMediaSource *source = NULL;
        hr = devices[d]->ActivateObject(__uuidof(IMFMediaSource), (void **)&source);
        if (SUCCEEDED(hr) && source) {
            IMFSourceReader *reader = NULL;
            hr = MFCreateSourceReaderFromMediaSource(source, NULL, &reader);
            if (SUCCEEDED(hr) && reader) {
                static const CM_Resolution test_resolutions[] = {
                    {640, 480}, {1280, 720}, {1920, 1080}
                };

                cam->supported_resolution_count = 0;

                for (int r = 0; r < CM_RES_COUNT; r++) {
                    GUID dummy_format;
                    if (find_media_type_for_resolution(reader,
                            test_resolutions[r].width,
                            test_resolutions[r].height,
                            &dummy_format) >= 0) {
                        cam->supported_resolutions[cam->supported_resolution_count].width =
                            test_resolutions[r].width;
                        cam->supported_resolutions[cam->supported_resolution_count].height =
                            test_resolutions[r].height;
                        cam->supported_resolution_count++;
                    }
                }

                reader->Release();
            }

            source->Shutdown();
            source->Release();
        }

        /* Default to best supported resolution */
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

    /* Release device list */
    for (UINT32 d = 0; d < device_count; d++) {
        devices[d]->Release();
    }
    CoTaskMemFree(devices);

    return count;
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

    /* Re-enumerate to find the device by serial number */
    IMFAttributes *attrs = NULL;
    HRESULT hr = MFCreateAttributes(&attrs, 1);
    if (FAILED(hr)) return CM_ERROR_OPEN_FAILED;

    hr = attrs->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                         MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    if (FAILED(hr)) {
        attrs->Release();
        return CM_ERROR_OPEN_FAILED;
    }

    IMFActivate **devices = NULL;
    UINT32 device_count = 0;
    hr = MFEnumDeviceSources(attrs, &devices, &device_count);
    attrs->Release();

    if (FAILED(hr)) return CM_ERROR_OPEN_FAILED;

    /* Find matching device by serial number */
    IMFActivate *target_activate = NULL;

    for (UINT32 d = 0; d < device_count; d++) {
        WCHAR *symlink = NULL;
        UINT32 symlink_len = 0;
        hr = devices[d]->GetAllocatedString(
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
            &symlink, &symlink_len);

        if (SUCCEEDED(hr) && symlink) {
            char serial[CM_SERIAL_LEN] = {0};
            extract_serial_from_symbolic_link(symlink, serial, CM_SERIAL_LEN);
            CoTaskMemFree(symlink);

            if (strcmp(serial, camera->serial_number) == 0) {
                target_activate = devices[d];
                target_activate->AddRef();
                break;
            }
        }
    }

    /* Release device list */
    for (UINT32 d = 0; d < device_count; d++) {
        devices[d]->Release();
    }
    CoTaskMemFree(devices);

    if (!target_activate) return CM_ERROR_NO_CAMERA;

    /* Allocate handle */
    Win32CameraHandle *handle = (Win32CameraHandle *)calloc(1, sizeof(Win32CameraHandle));
    if (!handle) {
        target_activate->Release();
        return CM_ERROR_OPEN_FAILED;
    }

    ring_buffer_init(&handle->ring_buffer);
    handle->camera_index = camera->device_index;
    handle->width = camera->width;
    handle->height = camera->height;

    /* Activate media source */
    hr = target_activate->ActivateObject(__uuidof(IMFMediaSource),
                                          (void **)&handle->media_source);
    if (FAILED(hr) || !handle->media_source) {
        ring_buffer_destroy(&handle->ring_buffer);
        free(handle);
        target_activate->Release();
        return CM_ERROR_OPEN_FAILED;
    }

    handle->activate = target_activate;

    /* Create source reader */
    hr = MFCreateSourceReaderFromMediaSource(handle->media_source, NULL,
                                              &handle->source_reader);
    if (FAILED(hr) || !handle->source_reader) {
        handle->media_source->Shutdown();
        handle->media_source->Release();
        handle->activate->Release();
        ring_buffer_destroy(&handle->ring_buffer);
        free(handle);
        return CM_ERROR_OPEN_FAILED;
    }

    /* Find and set the best media type for requested resolution */
    GUID best_format = MFVideoFormat_RGB32;
    int type_index = find_media_type_for_resolution(handle->source_reader,
                                                      camera->width, camera->height,
                                                      &best_format);

    if (type_index >= 0) {
        IMFMediaType *type = NULL;
        hr = handle->source_reader->GetNativeMediaType(
            (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, (DWORD)type_index, &type);
        if (SUCCEEDED(hr) && type) {
            handle->source_reader->SetCurrentMediaType(
                (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, NULL, type);
            type->Release();
        }
        handle->native_format = best_format;
    } else {
        /* No exact match found - try to set output format via reader */
        IMFMediaType *output_type = NULL;
        MFCreateMediaType(&output_type);
        if (output_type) {
            output_type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
            output_type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
            MFSetAttributeSize(output_type, MF_MT_FRAME_SIZE, camera->width, camera->height);

            hr = handle->source_reader->SetCurrentMediaType(
                (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, NULL, output_type);
            output_type->Release();

            if (SUCCEEDED(hr)) {
                handle->native_format = MFVideoFormat_RGB32;
            } else {
                /* Fall back to whatever the camera gives us */
                IMFMediaType *current = NULL;
                handle->source_reader->GetCurrentMediaType(
                    (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, &current);
                if (current) {
                    current->GetGUID(MF_MT_SUBTYPE, &handle->native_format);
                    UINT32 w, h;
                    MFGetAttributeSize(current, MF_MT_FRAME_SIZE, &w, &h);
                    handle->width = (int)w;
                    handle->height = (int)h;
                    camera->width = (int)w;
                    camera->height = (int)h;
                    current->Release();
                }
            }
        }
    }

    /* Start capture thread */
    handle->capture_running = true;
    handle->capture_thread = CreateThread(NULL, 0, capture_thread_proc, handle, 0, NULL);
    if (!handle->capture_thread) {
        handle->source_reader->Release();
        handle->media_source->Shutdown();
        handle->media_source->Release();
        handle->activate->Release();
        ring_buffer_destroy(&handle->ring_buffer);
        free(handle);
        return CM_ERROR_OPEN_FAILED;
    }

    handle->is_open = true;
    camera->platform_handle = handle;

    /* Warmup and calibrate clock offset */
    Sleep(200);
    calibrate_clock_offset(handle);

    return CM_OK;
}

void cm_close_camera(CM_Camera *camera) {
    if (!camera || !camera->platform_handle) return;

    Win32CameraHandle *handle = (Win32CameraHandle *)camera->platform_handle;

    /* Stop capture thread */
    handle->capture_running = false;
    if (handle->capture_thread) {
        WaitForSingleObject(handle->capture_thread, 2000);
        CloseHandle(handle->capture_thread);
        handle->capture_thread = NULL;
    }

    /* Release Media Foundation objects */
    if (handle->source_reader) {
        handle->source_reader->Release();
        handle->source_reader = NULL;
    }
    if (handle->media_source) {
        handle->media_source->Shutdown();
        handle->media_source->Release();
        handle->media_source = NULL;
    }
    if (handle->activate) {
        handle->activate->Release();
        handle->activate = NULL;
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
        /* Need to close and reopen to change FPS with MF source reader */
        cm_close_camera(camera);
        return cm_open_camera(camera);
    }
    return CM_OK;
}

int cm_set_exposure(CM_Camera *camera, int exposure) {
    if (!camera) return CM_ERROR_INVALID_PARAM;
    camera->exposure = exposure;
    /* Note: Manual exposure via IAMVideoProcAmp not implemented yet */
    return CM_OK;
}

/* ============================================================================
 * Frame Capture
 * ============================================================================ */

int cm_capture_frame(CM_Camera *camera, CM_Frame *out_frame) {
    if (!camera || !out_frame) return CM_ERROR_INVALID_PARAM;
    if (!camera->platform_handle) return CM_ERROR_NO_CAMERA;

    Win32CameraHandle *handle = (Win32CameraHandle *)camera->platform_handle;

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
    Win32CameraHandle *handle = (Win32CameraHandle *)camera->platform_handle;
    return ring_buffer_get_latest_camera_pts(&handle->ring_buffer);
}

/* ============================================================================
 * Multi-Camera Synchronization
 * ============================================================================ */

int cm_capture_synced(CM_Camera *cameras, int camera_count, CM_SyncedFrameSet *out) {
    if (!cameras || !out || camera_count <= 0) return CM_ERROR_INVALID_PARAM;

    memset(out, 0, sizeof(CM_SyncedFrameSet));

    /*
     * Synchronization strategy (Clock-Offset Corrected PTS):
     *
     * Each camera has its own clock domain for PTS timestamps. We measure
     * the clock offset (arrival_ns - camera_pts_ns) at startup and use it
     * to convert camera timestamps to a common clock domain.
     *
     * 1. Get latest CORRECTED timestamp from each camera
     * 2. Compute the mean corrected timestamp as target
     * 3. For each camera, find frame with corrected timestamp closest to target
     * 4. Return frames with all timestamps (raw PTS, arrival, corrected)
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

        Win32CameraHandle *handle = (Win32CameraHandle *)cameras[i].platform_handle;
        offsets[i] = handle->clock_offset_ns;

        uint64_t camera_pts = ring_buffer_get_latest_camera_pts(&handle->ring_buffer);
        if (camera_pts > 0) {
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

        Win32CameraHandle *handle = (Win32CameraHandle *)cameras[i].platform_handle;

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
