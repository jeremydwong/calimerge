/*
 * intrinsic.cpp
 *
 * C++ implementation of camera intrinsic calibration.
 * Port of src/calimerge/calibration/intrinsic.py
 *
 * Included as a translation unit by calibration_unity.cpp — do not compile
 * this file directly.
 */

#include "cm_calibration.h"

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>

/* ============================================================================
 * cm_calibrate_intrinsics
 * ============================================================================ */

/*
 * Minimum corners required per frame to include it in calibration.
 * Matches Python's min_corners default of 4.
 */
#define CM_MIN_CORNERS_PER_FRAME 4

extern "C"
int
cm_calibrate_intrinsics(
    CM_PointPacket    **packets,
    int                 n_packets,
    int                 width,
    int                 height,
    const char         *serial,
    CM_CameraIntrinsics *out
)
{
    if (!packets || n_packets <= 0 || width <= 0 || height <= 0 || !out) {
        return CM_CAL_ERR_INVALID_PARAM;
    }

    /*
     * Build valid_obj / valid_img lists.
     *
     * Mirrors Python calibrate_intrinsics():
     *   - Skip packets with count < min_corners
     *   - Skip collinear views: all corners share a single X value or single Y value
     *     (unique X count < 2 OR unique Y count < 2)
     */
    std::vector<std::vector<cv::Point3f>> valid_obj;
    std::vector<std::vector<cv::Point2f>> valid_img;

    for (int i = 0; i < n_packets; ++i) {
        CM_PointPacket *p = packets[i];
        if (!p || p->count < CM_MIN_CORNERS_PER_FRAME) continue;
        if (!p->img_loc || !p->obj_loc) continue;

        int n = p->count;

        /* Check for collinear views: collect unique X and Y values from obj_loc */
        std::set<float> unique_x;
        std::set<float> unique_y;
        for (int j = 0; j < n; ++j) {
            unique_x.insert(p->obj_loc[j * 3 + 0]);
            unique_y.insert(p->obj_loc[j * 3 + 1]);
        }
        if (unique_x.size() < 2 || unique_y.size() < 2) {
            continue;  /* collinear — skip */
        }

        /* Collect 3D object points */
        std::vector<cv::Point3f> obj_pts((size_t)n);
        for (int j = 0; j < n; ++j) {
            obj_pts[(size_t)j] = cv::Point3f(
                p->obj_loc[j * 3 + 0],
                p->obj_loc[j * 3 + 1],
                p->obj_loc[j * 3 + 2]
            );
        }

        /* Collect 2D image points */
        std::vector<cv::Point2f> img_pts((size_t)n);
        for (int j = 0; j < n; ++j) {
            img_pts[(size_t)j] = cv::Point2f(
                p->img_loc[j * 2 + 0],
                p->img_loc[j * 2 + 1]
            );
        }

        valid_obj.push_back(std::move(obj_pts));
        valid_img.push_back(std::move(img_pts));
    }

    /* Need at least 3 valid frames */
    if ((int)valid_obj.size() < 3) {
        return CM_CAL_ERR_INSUFFICIENT;
    }

    /*
     * Run OpenCV calibration.
     * Mirrors Python: cv2.calibrateCamera(valid_obj, valid_img, (width, height), None, None)
     * No fixed flags — let OpenCV estimate all parameters.
     */
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    double rms = 0.0;
    try {
        rms = cv::calibrateCamera(
            valid_obj,
            valid_img,
            cv::Size(width, height),
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs
        );
    } catch (const cv::Exception &) {
        return CM_CAL_ERR_OPENCV;
    }

    /*
     * Fill output struct.
     * matrix: row-major 3x3 double
     * distortion: first 5 coefficients [k1, k2, p1, p2, k3]
     * error: rounded to 4 decimal places (matching Python's round(error, 4))
     */
    memset(out, 0, sizeof(CM_CameraIntrinsics));

    if (serial) {
        strncpy(out->serial_number, serial, sizeof(out->serial_number) - 1);
        out->serial_number[sizeof(out->serial_number) - 1] = '\0';
    }

    out->width      = width;
    out->height     = height;
    out->grid_count = (int)valid_obj.size();

    /* Copy 3x3 camera matrix (row-major) */
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out->matrix[r * 3 + c] = camera_matrix.at<double>(r, c);
        }
    }

    /* Copy distortion coefficients — take first 5 */
    int n_dist = dist_coeffs.rows * dist_coeffs.cols;
    for (int j = 0; j < 5 && j < n_dist; ++j) {
        out->distortion[j] = dist_coeffs.at<double>(j);
    }

    /* Round RMSE to 4 decimal places */
    out->error = round(rms * 10000.0) / 10000.0;

    return CM_CAL_OK;
}
