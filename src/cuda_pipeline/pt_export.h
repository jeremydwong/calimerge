/*
 * pt_export.h - CSV export API for tracked 3D pose data.
 *
 * Writes per-person CSV files compatible with the Python output format:
 *   {base}_person{id}.csv
 *
 * Column format:
 *   sync_index,Nose_X,Nose_Y,Nose_Z,L_Eye_X,L_Eye_Y,L_Eye_Z,...,T6_X,T6_Y,T6_Z
 *
 * One file per tracked person.  Empty/NaN for invalid (non-triangulated) keypoints.
 *
 * Style: Plain C structs + free functions. No classes, no templates, no STL.
 */

#ifndef PT_EXPORT_H
#define PT_EXPORT_H

#include "pt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * pt_export_csv - Export all tracked person data to CSV files.
 *
 * For each person track in the track state, writes a CSV file:
 *   {output_base_path}_person{person_id}.csv
 *
 * Only exports tracks that have at least one frame of history.
 *
 * Parameters:
 *   tracks           - pointer to the track state (contains all person histories)
 *   output_base_path - base path for output files (e.g. "output/poses.csv")
 *                      Actual files will be "output/poses.csv_person0.csv", etc.
 *
 * Returns PT_OK on success.
 * Returns PT_ERR_INVALID_PARAM if tracks or output_base_path is NULL.
 * Returns PT_ERR_FILE_NOT_FOUND if the output directory cannot be written to.
 */
int pt_export_csv(const PT_TrackState *tracks, const char *output_base_path);

/*
 * pt_export_get_person_count - Count how many persons have exportable data.
 *
 * Returns the number of tracks with history_count > 0.
 */
int pt_export_get_person_count(const PT_TrackState *tracks);

#ifdef __cplusplus
}
#endif

#endif /* PT_EXPORT_H */
