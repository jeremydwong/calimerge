"""
Multi-person tracking: cross-view matching, candidate generation, track assignment.

Adapted from posetrack/process_synced_poses.py.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

from .triangulation import (
    triangulate_keypoints,
    calculate_fundamental_matrix,
    point_to_epipolar_line_distance,
    project_3d_to_2d,
)


class PersonTrack:
    """Tracks one person across frames using 3D center-of-mass."""

    def __init__(
        self,
        person_id: int,
        track_id: int,
        keypoints_3d: list,
        sync_index: int,
        hip_indices: tuple[int, int],
        views_used=None,
        track_frames_til_lost_patience: int = 10,
        min_keypoints_for_com: int = 2,
    ):
        self.track_id = track_id
        self.person_id = person_id
        self.keypoints_3d_history = [keypoints_3d]
        self.views_used_history = [views_used] if views_used is not None else []
        self.last_seen_sync = sync_index
        self.frames_since_seen = 0
        self.is_active = True
        self.hip_indices = hip_indices
        self.track_frames_til_lost_patience = track_frames_til_lost_patience
        self.min_keypoints_for_com = min_keypoints_for_com

    def update(self, keypoints_3d, sync_index, views_used=None):
        self.keypoints_3d_history.append(keypoints_3d)
        if views_used is not None:
            self.views_used_history.append(views_used)
        self.last_seen_sync = sync_index
        self.frames_since_seen = 0

    def get_last_views_used(self):
        if not self.views_used_history:
            return None
        return self.views_used_history[-1]

    def increment_lost_counter(self):
        self.frames_since_seen += 1
        if self.frames_since_seen > self.track_frames_til_lost_patience:
            self.is_active = False

    def get_com_3d(self) -> np.ndarray | None:
        """3D center of mass from last known hip keypoints."""
        if not self.keypoints_3d_history:
            return None
        last_kps = self.keypoints_3d_history[-1]
        if last_kps is None:
            return None

        valid_hips = []
        for hip_idx in self.hip_indices:
            if hip_idx < len(last_kps):
                hip = last_kps[hip_idx]
                if hip is not None and not np.isnan(hip).any():
                    valid_hips.append(hip)

        if len(valid_hips) >= self.min_keypoints_for_com:
            return np.mean(valid_hips, axis=0)
        return None


def calculate_2d_com(
    keypoints_2d: list, hip_indices: tuple[int, int]
) -> np.ndarray | None:
    """Calculate 2D center of mass from hip keypoints."""
    valid_hips = []
    for hip_idx in hip_indices:
        if hip_idx < len(keypoints_2d):
            hip = keypoints_2d[hip_idx][:2]
            if not np.isnan(hip).any():
                valid_hips.append(hip)

    if len(valid_hips) >= 1:
        return np.mean(valid_hips, axis=0)
    return None


def calculate_3d_com_from_keypoints(
    keypoints_3d: list, hip_indices: tuple[int, int]
) -> np.ndarray | None:
    """Calculate 3D center of mass from keypoints."""
    if keypoints_3d is None:
        return None

    valid_hips = []
    for hip_idx in hip_indices:
        if hip_idx < len(keypoints_3d) and keypoints_3d[hip_idx] is not None:
            hip = keypoints_3d[hip_idx]
            if not np.isnan(hip).any():
                valid_hips.append(hip)

    if len(valid_hips) >= 1:
        return np.mean(valid_hips, axis=0)
    return None


def calculate_triangulation_quality(
    kps_to_triangulate: dict,
    keypoints_3d: list,
    projection_matrices: list,
    port_to_cam_index: dict,
    hip_indices: tuple[int, int],
) -> float:
    """
    Quality score (0-1) based on reprojection error of hip keypoints.
    """
    hip_errors = []

    for hip_idx in hip_indices:
        if hip_idx < len(keypoints_3d) and keypoints_3d[hip_idx] is not None:
            com_3d = keypoints_3d[hip_idx]

            for port, kps_2d in kps_to_triangulate.items():
                if hip_idx < len(kps_2d):
                    observed_2d = kps_2d[hip_idx][:2]
                    if not np.isnan(observed_2d).any():
                        cam_idx = port_to_cam_index[port]
                        P = projection_matrices[cam_idx]
                        projected_2d = project_3d_to_2d(com_3d, P)

                        if projected_2d is not None:
                            error = np.linalg.norm(observed_2d - projected_2d)
                            hip_errors.append(error)

    if not hip_errors:
        return 0.0

    avg_error = np.mean(hip_errors)
    max_acceptable_error = 50  # pixels
    return max(0.0, 1.0 - (avg_error / max_acceptable_error))


def group_detections_across_views_bipartite(
    detected_persons_2d: dict[int, list[dict]],
    projection_matrices: list[np.ndarray],
    port_to_cam_index: dict[int, int],
    camera_params: list[dict],
    epipolar_threshold: float = 30.0,
) -> list[dict[int, dict]]:
    """
    Group person detections across camera views using epipolar geometry
    and bipartite matching with union-find.

    Returns list of groups, where each group is a dict of port -> detection.
    """
    ports = list(detected_persons_2d.keys())
    if len(ports) < 2:
        return []

    pairwise_matches = {}

    # Match all pairs of views
    for i, port1 in enumerate(ports):
        detections1 = detected_persons_2d[port1]
        if not detections1:
            continue

        for port2 in ports[i + 1 :]:
            detections2 = detected_persons_2d[port2]
            if not detections2:
                continue

            cam_idx1 = port_to_cam_index[port1]
            cam_idx2 = port_to_cam_index[port2]
            F = calculate_fundamental_matrix(
                projection_matrices[cam_idx1], projection_matrices[cam_idx2]
            )

            n1, n2 = len(detections1), len(detections2)
            cost_matrix = np.full((n1, n2), 1000.0)

            for idx1, det1 in enumerate(detections1):
                for idx2, det2 in enumerate(detections2):
                    dist = point_to_epipolar_line_distance(
                        det1["com_2d"], det2["com_2d"], F
                    )
                    if dist < epipolar_threshold:
                        cost_matrix[idx1, idx2] = dist

            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            for idx1, idx2 in zip(row_indices, col_indices):
                if cost_matrix[idx1, idx2] < epipolar_threshold:
                    pairwise_matches[(port1, idx1, port2, idx2)] = cost_matrix[idx1, idx2]
                    pairwise_matches[(port2, idx2, port1, idx1)] = cost_matrix[idx1, idx2]

    # Union-Find to merge matches into groups
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for (port1, idx1, port2, idx2), _ in pairwise_matches.items():
        if port1 < port2:
            union((port1, idx1), (port2, idx2))

    groups_dict = {}
    for port in ports:
        for idx, detection in enumerate(detected_persons_2d[port]):
            root = find((port, idx))
            if root not in groups_dict:
                groups_dict[root] = {}
            groups_dict[root][port] = detection

    return [group for group in groups_dict.values() if len(group) >= 2]


def generate_3d_candidates_from_groups(
    groups: list[dict],
    port_to_cam_index: dict[int, int],
    camera_params: list[dict],
    projection_matrices: list[np.ndarray],
    hip_indices: tuple[int, int],
) -> list[list[dict]]:
    """
    Generate 3D person candidates by triangulating detections across groups.

    Returns list (one per group) of lists of candidate dicts, each with:
    keypoints_3d, com_3d, triang_quality, views, num_views.
    """
    results = []

    for group in groups:
        active_ports = list(group.keys())

        if len(active_ports) < 2:
            results.append([])
            continue

        group_candidates = []

        for num_views in range(2, len(active_ports) + 1):
            for view_combo in itertools.combinations(active_ports, num_views):
                kps_to_triangulate = {}
                total_confidence = 0.0
                valid_views = 0

                for port in view_combo:
                    detection = group[port]
                    if detection["com_2d"] is not None:
                        kps_to_triangulate[port] = detection["keypoints"]
                        total_confidence += detection["confidence"]
                        valid_views += 1

                if valid_views >= 2:
                    keypoints_3d = triangulate_keypoints(
                        kps_to_triangulate,
                        port_to_cam_index,
                        camera_params,
                        projection_matrices,
                    )

                    if keypoints_3d and any(kp is not None for kp in keypoints_3d):
                        com_3d = calculate_3d_com_from_keypoints(keypoints_3d, hip_indices)

                        if com_3d is not None:
                            triang_quality = calculate_triangulation_quality(
                                kps_to_triangulate,
                                keypoints_3d,
                                projection_matrices,
                                port_to_cam_index,
                                hip_indices,
                            )

                            group_candidates.append({
                                "keypoints_3d": keypoints_3d,
                                "com_3d": com_3d,
                                "triang_quality": triang_quality,
                                "views": view_combo,
                                "num_views": len(view_combo),
                            })

        results.append(group_candidates)

    return results


def assign_3d_candidates_to_tracks(
    active_tracks: list[PersonTrack],
    candidate_groups: list[list[dict]],
    max_distance: float = 0.2,
    default_views=None,
    max_tracks: int | None = None,
    min_new_track_distance: float = 0.3,
    max_new_track_distance: float = 5.0,
) -> tuple[dict, list[int], dict]:
    """
    Assign 3D candidates to existing tracks using Hungarian algorithm.

    Returns:
        assignments: dict of track_idx -> (group_idx, candidate_idx)
        unassigned_groups: list of unassigned group indices
        new_track_assignments: dict of new track indices -> (group_idx, candidate_idx)
    """
    n_tracks = len(active_tracks)
    n_groups = len(candidate_groups)

    assignments = {}
    used_groups = set()

    def _set_default_views(ingroups):
        if ingroups and ingroups[0]:
            return [ingroups[0][0]["views"]]
        return None

    if n_tracks == 0:
        if default_views is None:
            default_views = _set_default_views(candidate_groups)

        if default_views is not None:
            for i, curgroup in enumerate(candidate_groups):
                for j, candidate in enumerate(curgroup):
                    if set(candidate["views"]) == set(default_views[0]):
                        assignments[i] = (i, j)

        if max_tracks is not None and max_tracks > 0:
            new_track_assignments = {}
            for idx, (group_idx, cand_idx) in assignments.items():
                if idx < max_tracks:
                    new_track_assignments[idx] = (group_idx, cand_idx)
                    used_groups.add(group_idx)
            unassigned = [g for g in range(n_groups) if g not in used_groups]
            return assignments, unassigned, new_track_assignments

        return assignments, None, {}

    # Build cost matrix: tracks x groups
    HI = 1000.0
    cost_matrix = np.full((n_tracks, n_groups), HI)
    best_candidate_per_cell = {}

    for i_track, track in enumerate(active_tracks):
        track_com = track.get_com_3d()
        track_views = track.get_last_views_used()

        if track_com is None or track_views is None:
            continue

        track_views_set = set(track_views)

        for group_idx, candidate_list in enumerate(candidate_groups):
            best_distance = np.inf
            best_cand_idx = None

            for cand_idx, candidate in enumerate(candidate_list):
                cand_views = set(candidate.get("views", []))

                if cand_views != track_views_set:
                    continue

                cand_com = candidate.get("hip_3d") or candidate.get("com_3d")
                if cand_com is None:
                    continue

                distance = np.linalg.norm(track_com - cand_com)
                if distance < best_distance and distance < max_distance:
                    best_distance = distance
                    best_cand_idx = cand_idx

            if best_cand_idx is not None:
                cost_matrix[i_track, group_idx] = best_distance
                best_candidate_per_cell[(i_track, group_idx)] = best_cand_idx

    if np.all(cost_matrix == HI):
        unassigned_groups = list(range(n_groups))
    else:
        track_opt, group_opt = linear_sum_assignment(cost_matrix)

        for i_track, group_idx in zip(track_opt, group_opt):
            if cost_matrix[i_track, group_idx] < max_distance:
                cand_idx = best_candidate_per_cell[(i_track, group_idx)]
                assignments[i_track] = (group_idx, cand_idx)
                used_groups.add(group_idx)

        unassigned_groups = [g for g in range(n_groups) if g not in used_groups]

    # Add new tracks from unassigned groups
    new_track_assignments = {}
    if max_tracks is not None and n_tracks < max_tracks and unassigned_groups:
        n_new = min(max_tracks - n_tracks, len(unassigned_groups))

        if n_new > 0:
            # Reference position from first track
            reference_position = None
            if 0 in assignments:
                g_idx, c_idx = assignments[0]
                ref_cand = candidate_groups[g_idx][c_idx]
                reference_position = ref_cand.get("hip_3d") or ref_cand.get("com_3d")
            elif active_tracks:
                reference_position = active_tracks[0].get_com_3d()

            if reference_position is None:
                return assignments, unassigned_groups, new_track_assignments

            group_scores = []

            for group_idx in unassigned_groups:
                candidate_list = candidate_groups[group_idx]
                best_score = -np.inf
                best_cand_idx = None

                for cand_idx, candidate in enumerate(candidate_list):
                    cand_com = candidate.get("hip_3d") or candidate.get("com_3d")
                    if cand_com is None:
                        continue

                    dist_from_ref = np.linalg.norm(cand_com - reference_position)
                    if dist_from_ref > max_new_track_distance:
                        continue

                    too_close = False
                    min_dist = np.inf

                    for track in active_tracks:
                        track_com = track.get_com_3d()
                        if track_com is not None:
                            d = np.linalg.norm(cand_com - track_com)
                            min_dist = min(min_dist, d)
                            if d < min_new_track_distance:
                                too_close = True
                                break

                    if not too_close:
                        for t_idx, (g, c) in assignments.items():
                            assigned_com = candidate_groups[g][c].get("hip_3d") or candidate_groups[g][c].get("com_3d")
                            if assigned_com is not None:
                                d = np.linalg.norm(cand_com - assigned_com)
                                min_dist = min(min_dist, d)
                                if d < min_new_track_distance:
                                    too_close = True
                                    break

                    if too_close:
                        continue

                    score = 0.0
                    if min_dist != np.inf:
                        optimal = min_new_track_distance * 2
                        if min_new_track_distance <= min_dist <= optimal:
                            score += 10.0
                        elif min_dist > optimal:
                            score += 5.0 - (min_dist - optimal) * 0.5

                    score += (max_new_track_distance - dist_from_ref) / max_new_track_distance * 5.0

                    if default_views is not None:
                        cand_views = set(candidate.get("views", []))
                        if cand_views == set(default_views[0]):
                            score += 15.0

                    if score > best_score:
                        best_score = score
                        best_cand_idx = cand_idx

                if best_cand_idx is not None:
                    group_scores.append((best_score, group_idx, best_cand_idx))

            if group_scores:
                group_scores.sort(reverse=True)
                for i in range(min(n_new, len(group_scores))):
                    _, g_idx, c_idx = group_scores[i]
                    new_idx = n_tracks + i
                    assignments[new_idx] = (g_idx, c_idx)
                    new_track_assignments[new_idx] = (g_idx, c_idx)
                    used_groups.add(g_idx)

                unassigned_groups = [g for g in range(n_groups) if g not in used_groups]

    return assignments, unassigned_groups, new_track_assignments
