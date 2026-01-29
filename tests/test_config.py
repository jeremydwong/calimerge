"""
Tests for calimerge.config (TOML + SQLite).
"""

import numpy as np
import pytest

from calimerge.config import (
    load_project_config,
    save_project_config,
    create_default_project_config,
    init_intrinsics_db,
    save_intrinsics,
    load_intrinsics,
    list_intrinsics,
    delete_intrinsics,
    save_calibration_to_toml,
    load_calibration_from_toml,
)
from calimerge.types import (
    CameraConfig,
    CameraIntrinsics,
    CameraExtrinsics,
    CalibratedCamera,
    CharucoConfig,
    ProjectConfig,
)


class TestProjectConfig:
    def test_save_and_load_roundtrip(self, temp_dir, sample_charuco_config):
        """Config should survive save/load cycle."""
        cameras = {
            "CAM001": CameraConfig(
                serial_number="CAM001",
                port=0,
                enabled=True,
                resolution=(1920, 1080),
                rotation_count=1,
                exposure=-5,
            ),
            "CAM002": CameraConfig(
                serial_number="CAM002",
                port=1,
                enabled=False,
                resolution=(1280, 720),
            ),
        }

        original = ProjectConfig(
            fps=24,
            cameras=cameras,
            charuco_intrinsic=sample_charuco_config,
            charuco_extrinsic=sample_charuco_config,
            pose_backend="mediapipe",
            pose_device="mps",
            max_persons=2,
        )

        config_path = temp_dir / "config.toml"
        save_project_config(original, config_path)

        assert config_path.exists()

        loaded = load_project_config(config_path)

        assert loaded.fps == 24
        assert loaded.pose_backend == "mediapipe"
        assert loaded.pose_device == "mps"
        assert loaded.max_persons == 2
        assert len(loaded.cameras) == 2
        assert "CAM001" in loaded.cameras
        assert loaded.cameras["CAM001"].resolution == (1920, 1080)
        assert loaded.cameras["CAM002"].enabled is False

    def test_create_default_config(self):
        config = create_default_project_config()
        assert config.fps == 30
        # Intrinsic: 7x5 board with 3cm squares
        assert config.charuco_intrinsic.columns == 7
        assert config.charuco_intrinsic.rows == 5
        assert config.charuco_intrinsic.square_size_cm == 3.0
        # Extrinsic: 4x3 board with 5cm squares
        assert config.charuco_extrinsic.columns == 4
        assert config.charuco_extrinsic.rows == 3
        assert config.charuco_extrinsic.square_size_cm == 5.0
        assert len(config.cameras) == 0


class TestIntrinsicsDatabase:
    def test_init_creates_db(self, temp_dir):
        db_path = temp_dir / "test_intrinsics.db"
        assert not db_path.exists()

        init_intrinsics_db(db_path)

        assert db_path.exists()

    def test_save_and_load_intrinsics(self, temp_dir, sample_camera_intrinsics):
        db_path = temp_dir / "intrinsics.db"

        save_intrinsics(sample_camera_intrinsics, db_path)

        loaded = load_intrinsics(
            sample_camera_intrinsics.serial_number,
            sample_camera_intrinsics.resolution,
            db_path,
        )

        assert loaded is not None
        assert loaded.serial_number == sample_camera_intrinsics.serial_number
        assert loaded.resolution == sample_camera_intrinsics.resolution
        assert loaded.error == sample_camera_intrinsics.error
        assert loaded.grid_count == sample_camera_intrinsics.grid_count
        np.testing.assert_array_almost_equal(
            loaded.matrix, sample_camera_intrinsics.matrix
        )
        np.testing.assert_array_almost_equal(
            loaded.distortion, sample_camera_intrinsics.distortion
        )

    def test_load_nonexistent_returns_none(self, temp_dir):
        db_path = temp_dir / "intrinsics.db"
        init_intrinsics_db(db_path)

        result = load_intrinsics("NONEXISTENT", (1280, 720), db_path)
        assert result is None

    def test_load_from_nonexistent_db_returns_none(self, temp_dir):
        db_path = temp_dir / "does_not_exist.db"
        result = load_intrinsics("ANY", (1280, 720), db_path)
        assert result is None

    def test_update_existing_intrinsics(self, temp_dir, sample_intrinsics_matrix, sample_distortion):
        db_path = temp_dir / "intrinsics.db"

        # Save first version
        v1 = CameraIntrinsics(
            serial_number="CAM001",
            resolution=(1280, 720),
            matrix=sample_intrinsics_matrix,
            distortion=sample_distortion,
            error=0.5,
            grid_count=20,
        )
        save_intrinsics(v1, db_path)

        # Save updated version (same key)
        v2 = CameraIntrinsics(
            serial_number="CAM001",
            resolution=(1280, 720),
            matrix=sample_intrinsics_matrix,
            distortion=sample_distortion,
            error=0.25,  # Better calibration
            grid_count=50,
        )
        save_intrinsics(v2, db_path)

        # Should get the updated version
        loaded = load_intrinsics("CAM001", (1280, 720), db_path)
        assert loaded.error == 0.25
        assert loaded.grid_count == 50

    def test_list_intrinsics(self, temp_dir, sample_intrinsics_matrix, sample_distortion):
        db_path = temp_dir / "intrinsics.db"

        # Add several entries
        for serial, res, err in [
            ("CAM001", (1280, 720), 0.3),
            ("CAM001", (1920, 1080), 0.25),
            ("CAM002", (1280, 720), 0.4),
        ]:
            intr = CameraIntrinsics(
                serial_number=serial,
                resolution=res,
                matrix=sample_intrinsics_matrix,
                distortion=sample_distortion,
                error=err,
                grid_count=30,
            )
            save_intrinsics(intr, db_path)

        entries = list_intrinsics(db_path)
        assert len(entries) == 3

        # Check they're sorted
        serials = [e[0] for e in entries]
        assert serials == sorted(serials)

    def test_delete_specific_resolution(self, temp_dir, sample_intrinsics_matrix, sample_distortion):
        db_path = temp_dir / "intrinsics.db"

        # Add two resolutions for same camera
        for res in [(1280, 720), (1920, 1080)]:
            intr = CameraIntrinsics(
                serial_number="CAM001",
                resolution=res,
                matrix=sample_intrinsics_matrix,
                distortion=sample_distortion,
                error=0.3,
                grid_count=30,
            )
            save_intrinsics(intr, db_path)

        # Delete one
        deleted = delete_intrinsics("CAM001", (1280, 720), db_path)
        assert deleted == 1

        # Check other still exists
        assert load_intrinsics("CAM001", (1920, 1080), db_path) is not None
        assert load_intrinsics("CAM001", (1280, 720), db_path) is None

    def test_delete_all_for_serial(self, temp_dir, sample_intrinsics_matrix, sample_distortion):
        db_path = temp_dir / "intrinsics.db"

        # Add multiple
        for serial, res in [("CAM001", (1280, 720)), ("CAM001", (1920, 1080)), ("CAM002", (1280, 720))]:
            intr = CameraIntrinsics(
                serial_number=serial,
                resolution=res,
                matrix=sample_intrinsics_matrix,
                distortion=sample_distortion,
                error=0.3,
                grid_count=30,
            )
            save_intrinsics(intr, db_path)

        # Delete all for CAM001
        deleted = delete_intrinsics("CAM001", db_path=db_path)
        assert deleted == 2

        # CAM002 should still exist
        assert load_intrinsics("CAM002", (1280, 720), db_path) is not None


class TestCalibrationToml:
    def test_save_and_load_calibration(self, temp_dir, sample_calibrated_camera):
        cameras = {0: sample_calibrated_camera}
        toml_path = temp_dir / "calibration.toml"
        db_path = temp_dir / "intrinsics.db"

        # Save intrinsics first (required for loading)
        save_intrinsics(sample_calibrated_camera.intrinsics, db_path)

        # Save calibration
        save_calibration_to_toml(cameras, toml_path)
        assert toml_path.exists()

        # Load it back
        loaded = load_calibration_from_toml(toml_path, db_path)

        assert loaded is not None
        assert 0 in loaded
        assert loaded[0].serial_number == "TEST123"
        np.testing.assert_array_almost_equal(
            loaded[0].extrinsics.rotation,
            sample_calibrated_camera.extrinsics.rotation,
        )

    def test_load_nonexistent_returns_none(self, temp_dir):
        result = load_calibration_from_toml(temp_dir / "nope.toml")
        assert result is None
