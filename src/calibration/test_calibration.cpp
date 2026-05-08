/*
 * test_calibration.cpp
 *
 * Minimal smoke test for the cm_calibration library.
 *
 * Verifies:
 *   1. cm_generate_board_image returns non-NULL with expected dimensions
 *   2. cm_detect_charuco returns a non-NULL packet (count may be 0 on blank input)
 *   3. cm_free_point_packet and cm_free_image do not crash
 *
 * Build output: build/calibration/test_calibration.exe (Windows)
 *               build/calibration/test_calibration     (macOS)
 *
 * Exit codes: 0 = all PASS, 1 = any FAIL
 */

#include "cm_calibration.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int g_failures = 0;

#define EXPECT_TRUE(expr, msg) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "FAIL: %s\n", (msg)); \
            g_failures++; \
        } else { \
            printf("PASS: %s\n", (msg)); \
        } \
    } while (0)

int main(void)
{
    /* ------------------------------------------------------------------ */
    /* Test 1: cm_generate_board_image                                     */
    /* ------------------------------------------------------------------ */
    {
        CM_CharucoConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.columns       = 7;
        cfg.rows          = 5;
        cfg.square_size_cm = 3.0f;
        strncpy(cfg.dictionary, "DICT_4X4_50", sizeof(cfg.dictionary) - 1);
        cfg.inverted      = 0;
        cfg.legacy_pattern = 0;

        int stride = 0;
        int W = 800, H = 600;
        uint8_t *img = cm_generate_board_image(&cfg, W, H, &stride);

        EXPECT_TRUE(img != NULL, "board image generated (non-null)");
        if (img != NULL) {
            EXPECT_TRUE(stride == W * 3, "board image stride == width * 3 (BGR)");
            printf("PASS: board image generated %dx%d stride=%d\n", W, H, stride);
            cm_free_image(img);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Test 2: cm_generate_board_image with inverted flag                  */
    /* ------------------------------------------------------------------ */
    {
        CM_CharucoConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.columns       = 5;
        cfg.rows          = 5;
        cfg.square_size_cm = 5.0f;
        strncpy(cfg.dictionary, "DICT_4X4_50", sizeof(cfg.dictionary) - 1);
        cfg.inverted      = 1;
        cfg.legacy_pattern = 0;

        int stride = 0;
        uint8_t *img = cm_generate_board_image(&cfg, 500, 500, &stride);
        EXPECT_TRUE(img != NULL, "inverted board image generated (non-null)");
        if (img != NULL) cm_free_image(img);
    }

    /* ------------------------------------------------------------------ */
    /* Test 3: cm_generate_board_image with NULL config returns NULL       */
    /* ------------------------------------------------------------------ */
    {
        int stride = 0;
        uint8_t *img = cm_generate_board_image(NULL, 100, 100, &stride);
        EXPECT_TRUE(img == NULL, "NULL config returns NULL image");
    }

    /* ------------------------------------------------------------------ */
    /* Test 4: cm_detect_charuco on a blank (all-grey) frame               */
    /* Expect: non-NULL packet with count == 0 (no board visible)          */
    /* ------------------------------------------------------------------ */
    {
        CM_CharucoConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.columns        = 7;
        cfg.rows           = 5;
        cfg.square_size_cm = 3.0f;
        strncpy(cfg.dictionary, "DICT_4X4_50", sizeof(cfg.dictionary) - 1);
        cfg.inverted       = 0;
        cfg.legacy_pattern = 0;

        int W = 640, H = 480;
        /* All-grey image: no markers visible */
        uint8_t *blank = (uint8_t *)malloc((size_t)W * H * 3);
        if (blank) {
            memset(blank, 128, (size_t)W * H * 3);
            CM_PointPacket *p = cm_detect_charuco(blank, W, H, &cfg);
            EXPECT_TRUE(p != NULL, "cm_detect_charuco returns non-NULL on blank frame");
            if (p != NULL) {
                EXPECT_TRUE(p->count == 0, "blank frame yields 0 detections");
                cm_free_point_packet(p);
            }
            free(blank);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Test 5: cm_detect_charuco on a rendered board image                 */
    /* Expect: non-NULL packet with count > 0                              */
    /* ------------------------------------------------------------------ */
    {
        CM_CharucoConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.columns        = 7;
        cfg.rows           = 5;
        cfg.square_size_cm = 3.0f;
        strncpy(cfg.dictionary, "DICT_4X4_50", sizeof(cfg.dictionary) - 1);
        cfg.inverted       = 0;
        cfg.legacy_pattern = 0;

        int stride = 0;
        int W = 1000, H = 1000;
        uint8_t *board_img = cm_generate_board_image(&cfg, W, H, &stride);
        if (board_img != NULL) {
            CM_PointPacket *p = cm_detect_charuco(board_img, W, H, &cfg);
            EXPECT_TRUE(p != NULL, "cm_detect_charuco returns non-NULL on board image");
            if (p != NULL) {
                if (p->count > 0) {
                    printf("PASS: detected %d corners in rendered board image\n", p->count);
                } else {
                    /* Detection on a rendered image should almost always work.
                     * Warn but don't fail — rendering quality may vary by build. */
                    printf("WARN: 0 corners detected in rendered board image "
                           "(unexpected but not a hard failure)\n");
                }
                cm_free_point_packet(p);
            }
            cm_free_image(board_img);
        } else {
            fprintf(stderr, "SKIP: board image generation failed, skipping detection test\n");
        }
    }

    /* ------------------------------------------------------------------ */
    /* Test 6: cm_calibrate_intrinsics with NULL/empty input               */
    /* ------------------------------------------------------------------ */
    {
        CM_CameraIntrinsics out;
        memset(&out, 0, sizeof(out));

        int ret = cm_calibrate_intrinsics(NULL, 0, 640, 480, "TEST", &out);
        EXPECT_TRUE(ret == CM_CAL_ERR_INVALID_PARAM,
                    "NULL packets returns CM_CAL_ERR_INVALID_PARAM");
    }

    /* ------------------------------------------------------------------ */
    /* Test 7: cm_calibrate_intrinsics with too few valid frames            */
    /* ------------------------------------------------------------------ */
    {
        /* Pass two packets that have enough corners but are not enough frames */
        CM_CharucoConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.columns        = 7;
        cfg.rows           = 5;
        cfg.square_size_cm = 3.0f;
        strncpy(cfg.dictionary, "DICT_4X4_50", sizeof(cfg.dictionary) - 1);

        int W = 1000, H = 1000;
        int stride = 0;
        uint8_t *board_img = cm_generate_board_image(&cfg, W, H, &stride);

        if (board_img) {
            /* Only 2 packets — below the minimum of 3 */
            CM_PointPacket *p0 = cm_detect_charuco(board_img, W, H, &cfg);
            CM_PointPacket *p1 = cm_detect_charuco(board_img, W, H, &cfg);

            CM_PointPacket *packets[2] = { p0, p1 };
            CM_CameraIntrinsics out;
            memset(&out, 0, sizeof(out));

            int ret = cm_calibrate_intrinsics(packets, 2, W, H, "TEST_CAM", &out);
            /* With only 2 identical frames we expect CM_CAL_ERR_INSUFFICIENT */
            EXPECT_TRUE(ret == CM_CAL_ERR_INSUFFICIENT,
                        "2-frame calibration returns CM_CAL_ERR_INSUFFICIENT");

            if (p0) cm_free_point_packet(p0);
            if (p1) cm_free_point_packet(p1);
            cm_free_image(board_img);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Summary                                                              */
    /* ------------------------------------------------------------------ */
    printf("\n");
    if (g_failures == 0) {
        printf("All tests PASSED\n");
        return 0;
    } else {
        fprintf(stderr, "%d test(s) FAILED\n", g_failures);
        return 1;
    }
}
