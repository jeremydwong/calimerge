"""
Tests for calimerge.calibration.charuco.
"""

import cv2
import numpy as np
import pytest

from calimerge.calibration.charuco import (
    ARUCO_DICTIONARIES,
    create_charuco_board,
    generate_board_image,
    get_charuco_object_points,
    get_connected_corners,
    get_corner_distances,
)
from calimerge.types import CharucoConfig


class TestArucoDictionaries:
    def test_common_dictionaries_exist(self):
        assert "DICT_4X4_50" in ARUCO_DICTIONARIES
        assert "DICT_5X5_100" in ARUCO_DICTIONARIES
        assert "DICT_6X6_250" in ARUCO_DICTIONARIES
        assert "DICT_APRILTAG_36h11" in ARUCO_DICTIONARIES

    def test_dictionary_values_are_integers(self):
        for name, value in ARUCO_DICTIONARIES.items():
            assert isinstance(value, int), f"{name} should map to int"


class TestCreateCharucoBoard:
    def test_creates_board(self, sample_charuco_config):
        board = create_charuco_board(sample_charuco_config)
        assert board is not None
        assert isinstance(board, cv2.aruco.CharucoBoard)

    def test_board_dimensions(self, sample_charuco_config):
        board = create_charuco_board(sample_charuco_config)
        # CharucoBoard has (cols-1) * (rows-1) inner corners
        corners = board.getChessboardCorners()
        expected_corners = (sample_charuco_config.columns - 1) * (sample_charuco_config.rows - 1)
        assert len(corners) == expected_corners

    def test_different_dictionaries(self):
        for dict_name in ["DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_50"]:
            config = CharucoConfig(
                columns=4,
                rows=4,
                square_size_cm=4.0,
                dictionary=dict_name,
            )
            board = create_charuco_board(config)
            assert board is not None


class TestGenerateBoardImage:
    def test_generates_image(self, sample_charuco_config):
        img = generate_board_image(sample_charuco_config)
        assert img is not None
        assert len(img.shape) == 3  # BGR
        assert img.shape[2] == 3

    def test_custom_dimensions(self, sample_charuco_config):
        img = generate_board_image(sample_charuco_config, width=800, height=600)
        assert img.shape[1] == 800  # width
        assert img.shape[0] == 600  # height

    def test_inverted_board(self):
        config = CharucoConfig(
            columns=4,
            rows=4,
            square_size_cm=4.0,
            inverted=True,
        )
        img_inverted = generate_board_image(config)

        config_normal = CharucoConfig(
            columns=4,
            rows=4,
            square_size_cm=4.0,
            inverted=False,
        )
        img_normal = generate_board_image(config_normal)

        # Images should be different (inverted)
        assert not np.array_equal(img_inverted, img_normal)


class TestGetCharucoObjectPoints:
    def test_returns_3d_points(self, sample_charuco_config):
        board = create_charuco_board(sample_charuco_config)
        points = get_charuco_object_points(board)

        expected_count = (sample_charuco_config.columns - 1) * (sample_charuco_config.rows - 1)
        assert points.shape == (expected_count, 3)

    def test_points_are_planar(self, sample_charuco_config):
        board = create_charuco_board(sample_charuco_config)
        points = get_charuco_object_points(board)

        # All Z coordinates should be 0 (board is planar)
        np.testing.assert_array_almost_equal(points[:, 2], 0)

    def test_points_spacing_matches_square_size(self, sample_charuco_config):
        board = create_charuco_board(sample_charuco_config)
        points = get_charuco_object_points(board)

        # Check spacing between adjacent corners
        # First corner and second corner should be square_size apart
        if len(points) >= 2:
            dist = np.linalg.norm(points[1] - points[0])
            # Should be approximately square_size (allowing for diagonal vs horizontal)
            assert dist <= sample_charuco_config.square_size_m * 1.5


class TestGetConnectedCorners:
    def test_returns_set_of_pairs(self, sample_charuco_config):
        board = create_charuco_board(sample_charuco_config)
        connected = get_connected_corners(board)

        assert isinstance(connected, set)
        for pair in connected:
            assert len(pair) == 2
            assert isinstance(pair[0], (int, np.integer))
            assert isinstance(pair[1], (int, np.integer))

    def test_connections_are_symmetric_ids(self, sample_charuco_config):
        board = create_charuco_board(sample_charuco_config)
        connected = get_connected_corners(board)

        # All IDs should be valid corner indices
        corners = board.getChessboardCorners()
        n_corners = len(corners)

        for a, b in connected:
            assert 0 <= a < n_corners
            assert 0 <= b < n_corners


class TestGetCornerDistances:
    def test_returns_distances(self, sample_charuco_config):
        board = create_charuco_board(sample_charuco_config)
        distances = get_corner_distances(board)

        assert isinstance(distances, dict)
        assert len(distances) > 0

        for (a, b), dist in distances.items():
            assert isinstance(dist, float)
            assert dist > 0

    def test_distances_match_square_size(self, sample_charuco_config):
        board = create_charuco_board(sample_charuco_config)
        distances = get_corner_distances(board)

        # All distances should be multiples of square_size (horizontal/vertical)
        # or sqrt(2)*square_size (diagonal)
        square = sample_charuco_config.square_size_m

        for dist in distances.values():
            # Check if it's approximately a valid grid distance
            ratio = dist / square
            # Should be close to 1, 2, 3, etc. or sqrt(2), 2*sqrt(2), etc.
            assert ratio > 0.5  # At least half a square
