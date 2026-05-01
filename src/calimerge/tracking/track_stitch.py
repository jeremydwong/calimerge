"""
Python-side track stitcher for the offline pose pipeline.

The C tracker in ``pt_main.cpp`` / ``pt_stream*.dll`` spawns a fresh track id
whenever the camera subset feeding triangulation changes for a single frame
(e.g. one camera drops a detection for a frame, then comes back). For a
single-subject trial this fragments one person into a dozen short tracks.

This module provides the post-processing step that re-merges tracks whose
hip-COM trajectories are spatially close and temporally adjacent. It lives
here, free of any Qt or worker dependency, so both the deprecated
``OfflineProcessingWorker`` and the new ``UnifiedOfflineWorker`` can use the
exact same merging logic — the schema of on-disk outputs has to stay
identical between the two.

A "track" in the input dict is::

    track_id -> {sync_index: list_of_(np.ndarray | None) keypoints}

The output is the same shape, but with track_ids merged where appropriate.
The merge rules are:

1. Two tracks may merge only if their sync-index ranges are *disjoint*
   (no overlap). Two people standing side by side are emitted on the
   same syncs and must not be stitched.
2. The temporal gap between the older track's last frame and the newer
   track's first frame must be at most ``max_gap_frames``.
3. The 3D hip COM at the seam (last frame of older + first frame of
   newer) must be within ``max_distance_m`` meters.

The hip COM helper is exposed separately so detection workers (live and
offline) can compute the same quantity for their own bookkeeping
(per-frame tracking, primary-person selection, etc.).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def hip_com(
    kps_3d: Sequence,
    hip_indices: tuple[int, int] = (11, 12),
) -> np.ndarray | None:
    """Return the 3D hip midpoint for one person, or None if both hips are NaN.

    Parameters
    ----------
    kps_3d : sequence of (np.ndarray | None)
        Per-keypoint 3D positions for a single person (length ``K``).
    hip_indices : tuple of int
        Indices of (L_Hip, R_Hip) in the keypoint list. Defaults to the
        SynthPose / COCO convention (11, 12).
    """
    valid = []
    for hip_idx in hip_indices:
        if 0 <= hip_idx < len(kps_3d):
            kp = kps_3d[hip_idx]
            if kp is None:
                continue
            arr = np.asarray(kp, dtype=float)
            if arr.size >= 3 and not np.isnan(arr[:3]).any():
                valid.append(arr[:3])
    if not valid:
        return None
    return np.mean(valid, axis=0)


def stitch_tracks(
    tracks: dict[int, dict[int, list]],
    *,
    max_gap_frames: int = 90,
    max_distance_m: float = 0.6,
    hip_indices: tuple[int, int] = (11, 12),
) -> dict[int, dict[int, list]]:
    """Re-merge fragmented per-frame keypoint tracks.

    Parameters
    ----------
    tracks : dict
        ``{track_id: {sync_index: [kp | None, ...]}}``. The values are
        keypoint lists exactly as returned by triangulation, NaNs and all.
    max_gap_frames : int
        Maximum number of sync-index frames between an older track's last
        frame and a newer track's first frame for the two to be eligible
        to merge.
    max_distance_m : float
        Maximum 3D hip-COM distance (meters) between the seam frames of
        two candidate tracks.
    hip_indices : tuple of int
        Indices of L_Hip and R_Hip in the keypoint list.

    Returns
    -------
    dict[int, dict[int, list]]
        ``{merged_track_id: {sync_index: [kp | None, ...]}}``. The merged
        track id is always one of the input ids (the earliest survivor in
        each cluster).
    """
    if not tracks:
        return {}

    # Per-track summary: first/last sync, COM at first/last sync.
    summaries: dict[int, dict] = {}
    for tid, frames in tracks.items():
        if not frames:
            continue
        sorted_syncs = sorted(frames.keys())
        first_sync = sorted_syncs[0]
        last_sync = sorted_syncs[-1]
        first_com = hip_com(frames[first_sync], hip_indices)
        last_com = hip_com(frames[last_sync], hip_indices)
        summaries[tid] = {
            "first_sync": first_sync,
            "last_sync": last_sync,
            "first_com": first_com,
            "last_com": last_com,
        }

    if not summaries:
        return {}

    # Union-find with iterative merging: repeatedly look for the closest
    # eligible pair (older.last + newer.first) and merge until no pairs
    # remain. Greedy on minimum gap then minimum distance — this matches
    # the deprecated worker's behaviour and the test expectations.
    parent = {tid: tid for tid in summaries}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(older: int, newer: int) -> None:
        ro, rn = find(older), find(newer)
        if ro == rn:
            return
        # Merge frames into the older root.
        for sync, kps in tracks[rn].items():
            tracks[ro][sync] = kps
        del tracks[rn]
        # Refresh summary for the merged root: spans whichever extremes are wider.
        merged_first_sync = min(summaries[ro]["first_sync"], summaries[rn]["first_sync"])
        merged_last_sync = max(summaries[ro]["last_sync"], summaries[rn]["last_sync"])
        merged_first_com = (
            summaries[ro]["first_com"]
            if summaries[ro]["first_sync"] <= summaries[rn]["first_sync"]
            else summaries[rn]["first_com"]
        )
        merged_last_com = (
            summaries[ro]["last_com"]
            if summaries[ro]["last_sync"] >= summaries[rn]["last_sync"]
            else summaries[rn]["last_com"]
        )
        summaries[ro] = {
            "first_sync": merged_first_sync,
            "last_sync": merged_last_sync,
            "first_com": merged_first_com,
            "last_com": merged_last_com,
        }
        del summaries[rn]
        parent[rn] = ro

    while True:
        # Find the best (older, newer) candidate pair where:
        #   older.last_sync < newer.first_sync (disjoint ranges)
        #   gap = newer.first_sync - older.last_sync <= max_gap_frames
        #   ||older.last_com - newer.first_com|| <= max_distance_m
        roots = sorted(summaries.keys())
        best = None
        best_score = None
        for i, a in enumerate(roots):
            sa = summaries[a]
            for b in roots[i + 1:]:
                sb = summaries[b]
                # Decide which is older — the one whose last_sync precedes
                # the other's first_sync.
                if sa["last_sync"] < sb["first_sync"]:
                    older, newer = a, b
                    older_s, newer_s = sa, sb
                elif sb["last_sync"] < sa["first_sync"]:
                    older, newer = b, a
                    older_s, newer_s = sb, sa
                else:
                    # Time-overlapping → cannot be the same person.
                    continue
                gap = newer_s["first_sync"] - older_s["last_sync"]
                if gap > max_gap_frames:
                    continue
                last_com = older_s["last_com"]
                first_com = newer_s["first_com"]
                if last_com is None or first_com is None:
                    continue
                dist = float(np.linalg.norm(last_com - first_com))
                if dist > max_distance_m:
                    continue
                # Score: prefer smaller gap, then smaller distance.
                score = (gap, dist)
                if best_score is None or score < best_score:
                    best = (older, newer)
                    best_score = score

        if best is None:
            break
        union(best[0], best[1])

    return tracks
