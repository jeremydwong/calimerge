/*
 * charuco.cpp
 *
 * C++ implementation of ChArUco board utilities.
 * Port of src/calimerge/calibration/charuco.py
 *
 * Included as a translation unit by calibration_unity.cpp — do not compile
 * this file directly.
 */

#include "cm_calibration.h"

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>   /* ArucoDetector, CharucoDetector, CharucoBoard (OpenCV 4.7+) */

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

/* ============================================================================
 * Internal helpers
 * ============================================================================ */

/* Map Python ARUCO_DICTIONARIES string names to OpenCV predefined type enum. */
static cv::aruco::PredefinedDictionaryType
dict_name_to_type(const char *name)
{
    /* Default: DICT_4X4_50 */
    if (!name || name[0] == '\0') return cv::aruco::DICT_4X4_50;

    std::string s(name);

    if (s == "DICT_4X4_50")         return cv::aruco::DICT_4X4_50;
    if (s == "DICT_4X4_100")        return cv::aruco::DICT_4X4_100;
    if (s == "DICT_4X4_250")        return cv::aruco::DICT_4X4_250;
    if (s == "DICT_4X4_1000")       return cv::aruco::DICT_4X4_1000;
    if (s == "DICT_5X5_50")         return cv::aruco::DICT_5X5_50;
    if (s == "DICT_5X5_100")        return cv::aruco::DICT_5X5_100;
    if (s == "DICT_5X5_250")        return cv::aruco::DICT_5X5_250;
    if (s == "DICT_5X5_1000")       return cv::aruco::DICT_5X5_1000;
    if (s == "DICT_6X6_50")         return cv::aruco::DICT_6X6_50;
    if (s == "DICT_6X6_100")        return cv::aruco::DICT_6X6_100;
    if (s == "DICT_6X6_250")        return cv::aruco::DICT_6X6_250;
    if (s == "DICT_6X6_1000")       return cv::aruco::DICT_6X6_1000;
    if (s == "DICT_7X7_50")         return cv::aruco::DICT_7X7_50;
    if (s == "DICT_7X7_100")        return cv::aruco::DICT_7X7_100;
    if (s == "DICT_7X7_250")        return cv::aruco::DICT_7X7_250;
    if (s == "DICT_7X7_1000")       return cv::aruco::DICT_7X7_1000;
    if (s == "DICT_ARUCO_ORIGINAL") return cv::aruco::DICT_ARUCO_ORIGINAL;
    if (s == "DICT_APRILTAG_16h5")  return cv::aruco::DICT_APRILTAG_16h5;
    if (s == "DICT_APRILTAG_25h9")  return cv::aruco::DICT_APRILTAG_25h9;
    if (s == "DICT_APRILTAG_36h10") return cv::aruco::DICT_APRILTAG_36h10;
    if (s == "DICT_APRILTAG_36h11") return cv::aruco::DICT_APRILTAG_36h11;

    /* Unknown string: fall back to default */
    return cv::aruco::DICT_4X4_50;
}

/*
 * Create a cv::aruco::CharucoBoard from CM_CharucoConfig.
 * square_size_m = cfg->square_size_cm / 100.0f
 * marker_size_m = square_size_m * 0.75f
 */
static cv::aruco::CharucoBoard
make_charuco_board(const CM_CharucoConfig *cfg)
{
    auto dict_type = dict_name_to_type(cfg->dictionary);
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dict_type);

    float square_m = cfg->square_size_cm / 100.0f;
    float marker_m = square_m * 0.75f;

    cv::aruco::CharucoBoard board(
        cv::Size(cfg->columns, cfg->rows),
        square_m,
        marker_m,
        dictionary
    );

    board.setLegacyPattern(cfg->legacy_pattern != 0);
    return board;
}

/*
 * Allocate a CM_PointPacket large enough to hold `count` entries.
 * All four data arrays are packed into a single malloc for simple ownership.
 * Returns NULL on allocation failure.
 */
static CM_PointPacket *
alloc_point_packet(int count)
{
    CM_PointPacket *p = (CM_PointPacket *)malloc(sizeof(CM_PointPacket));
    if (!p) return NULL;

    if (count <= 0) {
        p->img_loc    = NULL;
        p->obj_loc    = NULL;
        p->confidence = NULL;
        p->point_id   = NULL;
        p->count      = 0;
        return p;
    }

    /* Single allocation: 2 floats (img) + 3 floats (obj) + 1 float (conf) per corner,
     * plus 1 int (id) per corner. */
    size_t float_bytes = (size_t)count * (2 + 3 + 1) * sizeof(float);
    size_t int_bytes   = (size_t)count * sizeof(int);
    size_t total       = float_bytes + int_bytes;

    uint8_t *buf = (uint8_t *)malloc(total);
    if (!buf) {
        free(p);
        return NULL;
    }

    float *fp       = (float *)buf;
    p->img_loc      = fp;                           /* count * 2 floats */
    p->obj_loc      = fp + count * 2;               /* count * 3 floats */
    p->confidence   = fp + count * 2 + count * 3;  /* count * 1 float  */
    p->point_id     = (int *)(fp + count * 2 + count * 3 + count * 1);
    p->count        = count;
    return p;
}

/*
 * Internal: detect ChArUco corners in a grayscale image.
 * Fills ids_out and corners_out.
 * Returns true if at least one corner was found.
 *
 * Mirrors Python _find_corners():
 *   - Uses modern ArucoDetector + CharucoDetector API
 *   - Falls back to legacy API if the modern one is unavailable
 *   - Runs cornerSubPix refinement
 *   - Flattens (N,1,2) img_loc to (N,2)
 */
static bool
find_corners_gray(
    const cv::Mat                   &gray,
    const cv::aruco::Dictionary     &dictionary,
    const cv::aruco::CharucoBoard   &board,
    std::vector<int>                &ids_out,
    std::vector<cv::Point2f>        &corners_out
)
{
    ids_out.clear();
    corners_out.clear();

    /* Detect ArUco markers */
    std::vector<std::vector<cv::Point2f>> aruco_corners;
    std::vector<int>                      aruco_ids;

    cv::aruco::DetectorParameters det_params;
    cv::aruco::ArucoDetector aruco_detector(dictionary, det_params);
    aruco_detector.detectMarkers(gray, aruco_corners, aruco_ids);

    if (aruco_ids.empty() || aruco_corners.size() < 2) {
        return false;
    }

    /* Interpolate ChArUco corners */
    cv::Mat charuco_corners_mat;
    cv::Mat charuco_ids_mat;

    cv::aruco::CharucoDetector charuco_detector(board);
    charuco_detector.detectBoard(gray, charuco_corners_mat, charuco_ids_mat);

    if (charuco_ids_mat.empty() || charuco_corners_mat.empty()) {
        return false;
    }

    /* Flatten ids */
    int n = charuco_ids_mat.rows;
    if (n == 0) return false;

    ids_out.resize((size_t)n);
    for (int i = 0; i < n; ++i) {
        ids_out[(size_t)i] = charuco_ids_mat.at<int>(i, 0);
    }

    /* Flatten corners from (N,1,2) or (N,2) to flat vector */
    corners_out.resize((size_t)n);
    if (charuco_corners_mat.type() == CV_32FC2 && charuco_corners_mat.rows == n) {
        for (int i = 0; i < n; ++i) {
            corners_out[(size_t)i] = charuco_corners_mat.at<cv::Point2f>(i, 0);
        }
    } else {
        /* reshape to Nx1 of Point2f */
        cv::Mat flat = charuco_corners_mat.reshape(2, n);
        for (int i = 0; i < n; ++i) {
            corners_out[(size_t)i] = flat.at<cv::Point2f>(i, 0);
        }
    }

    /* Sub-pixel refinement */
    if (!corners_out.empty()) {
        cv::TermCriteria criteria(
            cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.0001
        );
        try {
            cv::cornerSubPix(gray, corners_out, cv::Size(11, 11), cv::Size(-1, -1), criteria);
        } catch (...) {
            /* Sub-pixel refinement failed; use raw corners */
        }
    }

    return !corners_out.empty();
}

/* ============================================================================
 * cm_detect_charuco
 * ============================================================================ */

extern "C"
CM_PointPacket *
cm_detect_charuco(
    const uint8_t          *bgr,
    int                     width,
    int                     height,
    const CM_CharucoConfig *cfg
)
{
    if (!bgr || width <= 0 || height <= 0 || !cfg) {
        /* Return an empty (count=0) packet rather than NULL so callers don't
         * have to distinguish "error" from "no detection". */
        return alloc_point_packet(0);
    }

    /* Wrap pixel buffer in cv::Mat (no copy) */
    cv::Mat frame_bgr(height, width, CV_8UC3, (void *)bgr);

    /* Convert to grayscale */
    cv::Mat gray;
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);

    /* Optionally invert */
    if (cfg->inverted) {
        cv::bitwise_not(gray, gray);
    }

    auto dict_type = dict_name_to_type(cfg->dictionary);
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dict_type);
    cv::aruco::CharucoBoard board    = make_charuco_board(cfg);

    std::vector<int>          ids;
    std::vector<cv::Point2f>  corners;

    bool found = find_corners_gray(gray, dictionary, board, ids, corners);

    /* If not found, try horizontally mirrored frame */
    if (!found) {
        cv::Mat gray_mirror;
        cv::flip(gray, gray_mirror, 1);

        std::vector<int>         ids_mirror;
        std::vector<cv::Point2f> corners_mirror;

        if (find_corners_gray(gray_mirror, dictionary, board, ids_mirror, corners_mirror)) {
            /* Flip x coordinates back to original frame space */
            for (auto &pt : corners_mirror) {
                pt.x = (float)width - pt.x;
            }
            ids     = std::move(ids_mirror);
            corners = std::move(corners_mirror);
            found   = true;
        }
    }

    int n = found ? (int)ids.size() : 0;
    CM_PointPacket *p = alloc_point_packet(n);
    if (!p) return NULL;

    if (n == 0) return p;

    /* Get 3D object-space positions for detected corner IDs */
    std::vector<cv::Point3f> board_corners = board.getChessboardCorners();

    for (int i = 0; i < n; ++i) {
        /* img_loc: (x, y) */
        p->img_loc[i * 2 + 0] = corners[(size_t)i].x;
        p->img_loc[i * 2 + 1] = corners[(size_t)i].y;

        /* obj_loc: (x, y, z) from board corner table */
        int corner_id = ids[(size_t)i];
        if (corner_id >= 0 && corner_id < (int)board_corners.size()) {
            p->obj_loc[i * 3 + 0] = board_corners[(size_t)corner_id].x;
            p->obj_loc[i * 3 + 1] = board_corners[(size_t)corner_id].y;
            p->obj_loc[i * 3 + 2] = board_corners[(size_t)corner_id].z;
        } else {
            p->obj_loc[i * 3 + 0] = 0.0f;
            p->obj_loc[i * 3 + 1] = 0.0f;
            p->obj_loc[i * 3 + 2] = 0.0f;
        }

        /* confidence: ChArUco corners don't have individual scores; use 1.0 */
        p->confidence[i] = 1.0f;

        /* point_id: ChArUco corner index */
        p->point_id[i] = corner_id;
    }

    return p;
}

/* ============================================================================
 * cm_free_point_packet
 * ============================================================================ */

extern "C"
void
cm_free_point_packet(CM_PointPacket *p)
{
    if (!p) return;
    /* The data arrays are packed into a single buffer starting at img_loc */
    if (p->img_loc) {
        free(p->img_loc);
    }
    free(p);
}

/* ============================================================================
 * cm_generate_board_image
 * ============================================================================ */

extern "C"
uint8_t *
cm_generate_board_image(
    const CM_CharucoConfig *cfg,
    int                     width,
    int                     height,
    int                    *out_stride
)
{
    if (!cfg || width <= 0 || height <= 0) return NULL;

    cv::aruco::CharucoBoard board = make_charuco_board(cfg);

    /* Generate board image (typically grayscale).
     * OpenCV 4.7+: generateImage(outSize, outImg, marginSize, borderBits) — output by ref. */
    cv::Mat board_img;
    board.generateImage(cv::Size(width, height), board_img);

    /* Invert if requested */
    if (cfg->inverted) {
        cv::bitwise_not(board_img, board_img);
    }

    /* Convert to BGR if grayscale */
    cv::Mat bgr_img;
    if (board_img.channels() == 1) {
        cv::cvtColor(board_img, bgr_img, cv::COLOR_GRAY2BGR);
    } else {
        bgr_img = board_img;
    }

    /* Ensure continuous memory */
    if (!bgr_img.isContinuous()) {
        bgr_img = bgr_img.clone();
    }

    int stride = bgr_img.cols * 3;  /* BGR: 3 bytes per pixel, no padding */
    size_t total_bytes = (size_t)bgr_img.rows * (size_t)stride;

    uint8_t *buf = (uint8_t *)malloc(total_bytes);
    if (!buf) return NULL;

    memcpy(buf, bgr_img.data, total_bytes);

    if (out_stride) *out_stride = stride;
    return buf;
}

/* ============================================================================
 * cm_free_image
 * ============================================================================ */

extern "C"
void
cm_free_image(uint8_t *img)
{
    free(img);
}
