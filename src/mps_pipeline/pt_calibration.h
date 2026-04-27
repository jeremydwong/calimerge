/*
 * pt_calibration.h - Calibration TOML loader (shared between pipelines).
 *
 * Loads camera intrinsics/extrinsics from the TOML format produced by
 * calimerge's extrinsic calibration.
 */

#ifndef PT_CALIBRATION_H
#define PT_CALIBRATION_H

#include "../pt_shared/pt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Load calibration from a TOML file into PT_CameraConstants.
 *
 * Supports both calimerge format ([camera_N]) and caliscope format ([cam_N]).
 * Handles Rodrigues rotation vectors (auto-converted to 3x3 matrices).
 *
 * Returns PT_OK on success, PT_ERR_FILE_NOT_FOUND if file doesn't exist.
 */
int pt_load_calibration(PT_CameraConstants *constants, const char *toml_path);

/*
 * After loading calibration, compute derived matrices (projection, fundamental).
 * Must be called before using constants for matching/triangulation.
 */
void pt_compute_derived_matrices(PT_CameraConstants *constants);

#ifdef __cplusplus
}
#endif

#endif /* PT_CALIBRATION_H */
