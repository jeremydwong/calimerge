"""Tests for the cross-frame _LiveTracker in the live PyTorch pose worker.

The tracker associates triangulated persons across frames by 3D hip COM. The
goal of these tests is to lock the contract so the live worker, the CSV
exporter and the foot-placement viz all agree on what "person id N" means.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_kps(n_keypoints: int, hip_l: tuple, hip_r: tuple, l_idx: int, r_idx: int):
    """Build a fake `list[np.ndarray | None]` keypoints array with given hips."""
    kps: list = [None] * n_keypoints
    kps[l_idx] = np.asarray(hip_l, dtype=float)
    kps[r_idx] = np.asarray(hip_r, dtype=float)
    return kps


@pytest.fixture
def tracker():
    from calimerge.gui.workers import _LiveTracker
    from calimerge.tracking.markers import HIP_INDICES
    return _LiveTracker(hip_indices=HIP_INDICES, max_match_distance=0.5,
                         patience=5, max_persons=8)


@pytest.fixture
def n_kps():
    from calimerge.tracking.markers import SYNTHPOSE_MARKERS
    return len(SYNTHPOSE_MARKERS)


@pytest.fixture
def hip_idx():
    from calimerge.tracking.markers import HIP_INDICES
    return HIP_INDICES


def test_empty_input_no_tracks(tracker):
    assert tracker.step([]) == []


def test_first_frame_assigns_fresh_ids(tracker, n_kps, hip_idx):
    p1 = _make_kps(n_kps, (0.0, 0.0, 1.0), (0.1, 0.0, 1.0), *hip_idx)
    p2 = _make_kps(n_kps, (2.0, 0.0, 1.0), (2.1, 0.0, 1.0), *hip_idx)
    ids = tracker.step([p1, p2])
    assert len(ids) == 2
    assert all(i > 0 for i in ids)
    assert ids[0] != ids[1]


def test_ids_stable_when_persons_dont_move(tracker, n_kps, hip_idx):
    p1 = _make_kps(n_kps, (0.0, 0.0, 1.0), (0.1, 0.0, 1.0), *hip_idx)
    p2 = _make_kps(n_kps, (2.0, 0.0, 1.0), (2.1, 0.0, 1.0), *hip_idx)
    ids_t0 = tracker.step([p1, p2])
    ids_t1 = tracker.step([p1, p2])
    ids_t2 = tracker.step([p1, p2])
    assert ids_t0 == ids_t1 == ids_t2


def test_ids_stable_when_input_order_swaps(tracker, n_kps, hip_idx):
    """Detector might emit persons in any order frame-to-frame. The tracker
    must follow each *spatial* person, not their list position."""
    a = _make_kps(n_kps, (0.0, 0.0, 1.0), (0.1, 0.0, 1.0), *hip_idx)
    b = _make_kps(n_kps, (2.0, 0.0, 1.0), (2.1, 0.0, 1.0), *hip_idx)
    ids_t0 = tracker.step([a, b])
    # Swap input order; the tracker should still report the same id for
    # each spatial person.
    ids_t1 = tracker.step([b, a])
    assert ids_t0[0] == ids_t1[1]
    assert ids_t0[1] == ids_t1[0]


def test_track_ages_out_after_patience(tracker, n_kps, hip_idx):
    p = _make_kps(n_kps, (0.0, 0.0, 1.0), (0.1, 0.0, 1.0), *hip_idx)
    [pid] = tracker.step([p])
    # Disappear for patience+1 frames
    for _ in range(tracker.patience + 1):
        tracker.step([])
    # Coming back gets a fresh id (track was dropped)
    [pid2] = tracker.step([p])
    assert pid2 != pid


def test_track_survives_brief_dropout(tracker, n_kps, hip_idx):
    p = _make_kps(n_kps, (0.0, 0.0, 1.0), (0.1, 0.0, 1.0), *hip_idx)
    [pid] = tracker.step([p])
    # Disappear for 1 frame (well within patience=5)
    tracker.step([])
    # Reappear at same location → same id
    [pid2] = tracker.step([p])
    assert pid2 == pid


def test_distant_jump_creates_new_track(tracker, n_kps, hip_idx):
    """If a person teleports far enough that the cost exceeds
    max_match_distance, the matcher should reject the assignment and the new
    detection should get a fresh track id."""
    a = _make_kps(n_kps, (0.0, 0.0, 1.0), (0.1, 0.0, 1.0), *hip_idx)
    b = _make_kps(n_kps, (10.0, 0.0, 1.0), (10.1, 0.0, 1.0), *hip_idx)  # 10m away
    [id_a] = tracker.step([a])
    [id_b] = tracker.step([b])
    assert id_b != id_a


def test_kps_with_no_hips_returns_zero_id(tracker, n_kps):
    """A detection lacking valid hip COM cannot be tracked → id 0."""
    kps_no_hips: list = [None] * n_kps
    [pid] = tracker.step([kps_no_hips])
    assert pid == 0


def test_reset_clears_state(tracker, n_kps, hip_idx):
    p = _make_kps(n_kps, (0.0, 0.0, 1.0), (0.1, 0.0, 1.0), *hip_idx)
    [pid_before] = tracker.step([p])
    tracker.reset()
    [pid_after] = tracker.step([p])
    # Reset should restart id counter at 1
    assert pid_after == 1
    assert pid_before >= 1


def test_max_persons_caps_creation(n_kps, hip_idx):
    from calimerge.gui.workers import _LiveTracker
    t = _LiveTracker(hip_indices=hip_idx, max_persons=2, patience=5)
    persons = [
        _make_kps(n_kps, (0.0, 0.0, 1.0), (0.1, 0.0, 1.0), *hip_idx),
        _make_kps(n_kps, (2.0, 0.0, 1.0), (2.1, 0.0, 1.0), *hip_idx),
        _make_kps(n_kps, (4.0, 0.0, 1.0), (4.1, 0.0, 1.0), *hip_idx),
    ]
    ids = t.step(persons)
    nonzero = [i for i in ids if i != 0]
    assert len(nonzero) == 2  # third person dropped — only max_persons tracks


def test_smooth_walk_keeps_id(tracker, n_kps, hip_idx):
    """Walking 0.1 m per frame across 20 frames stays under
    max_match_distance every step → id should never change."""
    pid = None
    for step in range(20):
        x = 0.1 * step
        p = _make_kps(n_kps, (x, 0.0, 1.0), (x + 0.1, 0.0, 1.0), *hip_idx)
        [new_id] = tracker.step([p])
        if pid is None:
            pid = new_id
        else:
            assert new_id == pid, f"track id changed at step {step}"
