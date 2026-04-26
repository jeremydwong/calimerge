"""Tests for calimerge.skeleton_3d."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from calimerge.skeleton_3d import (
    SYNTHPOSE_SCHEMA,
    build_skeleton_3d,
    load_skeleton_3d_npz,
    load_sync_index_times,
    save_skeleton_3d_npz,
)
from calimerge.types import KeypointSchema, Skeleton3D, XYZPoints


class TestKeypointSchema:
    def test_index_by_name(self):
        assert SYNTHPOSE_SCHEMA.index("L_Hip") == 11
        assert SYNTHPOSE_SCHEMA.index("R_Hip") == 12
        assert SYNTHPOSE_SCHEMA.K == 52

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="not in schema"):
            SYNTHPOSE_SCHEMA.index("Tail")


class TestBuildSkeleton3D:
    def test_empty_input(self):
        skel = build_skeleton_3d([], SYNTHPOSE_SCHEMA)
        assert skel.xyz.shape == (0, 1, 52, 3)
        assert skel.timestamps.shape == (0,)
        assert skel.track_ids == (0,)

    def test_single_frame_densifies(self):
        pts = XYZPoints(
            sync_index=5,
            point_ids=np.array([11, 12], dtype=np.int64),
            xyz=np.array([[0.0, 0.1, 0.8], [0.0, -0.1, 0.8]], dtype=np.float32),
        )
        skel = build_skeleton_3d([pts], SYNTHPOSE_SCHEMA, fps=30.0)

        assert skel.xyz.shape == (1, 1, 52, 3)
        np.testing.assert_array_almost_equal(skel.xyz[0, 0, 11], [0.0, 0.1, 0.8])
        np.testing.assert_array_almost_equal(skel.xyz[0, 0, 12], [0.0, -0.1, 0.8])
        # everything else is NaN
        assert np.isnan(skel.xyz[0, 0, 0]).all()
        assert np.isnan(skel.xyz[0, 0, 20]).all()

    def test_stacks_over_time(self):
        pts = [
            XYZPoints(
                sync_index=i,
                point_ids=np.array([11], dtype=np.int64),
                xyz=np.array([[0.0, 0.0, 0.5 + 0.1 * i]], dtype=np.float32),
            )
            for i in range(4)
        ]
        skel = build_skeleton_3d(pts, SYNTHPOSE_SCHEMA, fps=30.0)

        assert skel.xyz.shape == (4, 1, 52, 3)
        z_series = skel.xyz[:, 0, 11, 2]
        np.testing.assert_array_almost_equal(z_series, [0.5, 0.6, 0.7, 0.8])
        # timestamps start at 0, step at 1/fps
        np.testing.assert_array_almost_equal(
            skel.timestamps, [0.0, 1 / 30, 2 / 30, 3 / 30]
        )
        np.testing.assert_array_equal(skel.sync_indices, [0, 1, 2, 3])

    def test_timestamps_from_csv(self, tmp_path: Path):
        csv_path = tmp_path / "frame_time_history.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sync_index", "port", "frame_index", "frame_time"])
            # sync 100: three ports, near-identical times
            w.writerow([100, 0, 50, 1000.000])
            w.writerow([100, 1, 50, 1000.001])
            w.writerow([100, 2, 50, 1000.002])
            # sync 101
            w.writerow([101, 0, 51, 1000.033])
            w.writerow([101, 1, 51, 1000.034])
            w.writerow([101, 2, 51, 1000.035])

        pts = [
            XYZPoints(sync_index=100, point_ids=np.array([0]), xyz=np.array([[0, 0, 0]])),
            XYZPoints(sync_index=101, point_ids=np.array([0]), xyz=np.array([[0, 0, 0]])),
        ]
        skel = build_skeleton_3d(pts, SYNTHPOSE_SCHEMA, frame_time_csv=csv_path)

        np.testing.assert_array_almost_equal(
            skel.timestamps, [0.0, 1000.034 - 1000.001], decimal=6
        )

    def test_out_of_range_keypoint_ids_ignored(self):
        # point_id = 99 is outside a 52-keypoint schema; should silently skip
        pts = XYZPoints(
            sync_index=0,
            point_ids=np.array([11, 99], dtype=np.int64),
            xyz=np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        )
        skel = build_skeleton_3d([pts], SYNTHPOSE_SCHEMA)
        np.testing.assert_array_equal(skel.xyz[0, 0, 11], [1.0, 2.0, 3.0])


class TestLoadSyncIndexTimes:
    def test_median_across_ports(self, tmp_path: Path):
        csv_path = tmp_path / "times.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sync_index", "port", "frame_index", "frame_time"])
            w.writerow([1, 0, 0, 10.0])
            w.writerow([1, 1, 0, 10.1])  # outlier
            w.writerow([1, 2, 0, 10.01])
        times = load_sync_index_times(csv_path)
        assert times[1] == 10.01  # median is robust to the outlier


class TestNpzRoundtrip:
    def test_roundtrip(self, tmp_path: Path):
        pts = [
            XYZPoints(
                sync_index=i,
                point_ids=np.array([11, 12], dtype=np.int64),
                xyz=np.array(
                    [[0.0, 0.1, 0.5 + 0.05 * i], [0.0, -0.1, 0.5 + 0.05 * i]],
                    dtype=np.float32,
                ),
            )
            for i in range(3)
        ]
        skel = build_skeleton_3d(pts, SYNTHPOSE_SCHEMA, fps=30.0)

        path = tmp_path / "skeleton.npz"
        save_skeleton_3d_npz(path, skel)
        loaded = load_skeleton_3d_npz(path)

        np.testing.assert_array_equal(loaded.xyz, skel.xyz)
        np.testing.assert_array_equal(loaded.timestamps, skel.timestamps)
        np.testing.assert_array_equal(loaded.sync_indices, skel.sync_indices)
        assert loaded.track_ids == skel.track_ids
        assert loaded.schema.names == skel.schema.names
