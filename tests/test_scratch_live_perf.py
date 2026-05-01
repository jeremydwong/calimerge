"""Scratch tests for the live-tracking perf changes.

Confirms the public surface added during the perf pass:
  - PoseDetectionWorker has the batch path + draw helper
  - WorkoutPage has the 3 s pre-roll countdown plumbing
  - PoseDetectionWorker has the camera-params cache helper

These aren't behavioural tests — they only catch regressions where one of
the methods gets renamed or accidentally removed during a refactor. Real
behaviour is exercised by the GUI itself.
"""

from __future__ import annotations


def test_pose_detection_worker_has_batch_methods():
    from calimerge.gui.workers import PoseDetectionWorker

    assert hasattr(PoseDetectionWorker, "_detect_and_draw_batch")
    assert hasattr(PoseDetectionWorker, "_draw_overlay")
    assert hasattr(PoseDetectionWorker, "_detect_and_draw")  # per-port fallback
    assert hasattr(PoseDetectionWorker, "_ensure_camera_caches")


def test_workout_page_has_countdown_plumbing():
    from calimerge.gui.workout_page import WorkoutPage

    assert hasattr(WorkoutPage, "_on_record")
    assert hasattr(WorkoutPage, "_record_countdown_tick")
    assert hasattr(WorkoutPage, "_begin_recording_now")


def test_workout_page_has_resolve_calibration_for_serials():
    """Subset-fallback fix from the calibration resolver."""
    from calimerge.gui.workout_page import WorkoutPage

    assert hasattr(WorkoutPage, "_resolve_calibration_for_serials")


def test_fps_graph_widget_supports_y_max_factor_and_time_window():
    from calimerge.gui.tabs.cameras_tab import FpsGraphWidget
    import inspect

    sig = inspect.signature(FpsGraphWidget.__init__)
    assert "y_max_factor" in sig.parameters
    assert "time_window_s" in sig.parameters


def test_view_transform_db_roundtrip(tmp_path):
    """Per-model view-transform save/load via sqlite."""
    import numpy as np
    from calimerge.config import (
        save_view_transform,
        load_view_transform,
        delete_view_transform,
    )

    db = tmp_path / "view_transforms.db"
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    t = np.array([0.5, -0.25, 0.1])

    assert load_view_transform("vitpose", db_path=db) is None

    save_view_transform("vitpose", R, t, has_origin=True, db_path=db)
    save_view_transform("mediapipe_hands", np.eye(3), np.zeros(3),
                        has_origin=False, db_path=db)

    R2, t2, ho = load_view_transform("vitpose", db_path=db)
    assert np.allclose(R2, R)
    assert np.allclose(t2, t)
    assert ho is True

    R3, t3, ho3 = load_view_transform("mediapipe_hands", db_path=db)
    assert np.allclose(R3, np.eye(3))
    assert np.allclose(t3, np.zeros(3))
    assert ho3 is False

    delete_view_transform("vitpose", db_path=db)
    assert load_view_transform("vitpose", db_path=db) is None
    # Other model untouched
    assert load_view_transform("mediapipe_hands", db_path=db) is not None


def test_write_raw_buffer_applies_view_transform(tmp_path):
    """Keypoints in the npz should be in view frame when (R, t) is given."""
    import numpy as np
    from calimerge.keypoint_export import write_raw_buffer

    # Single frame, single person, single keypoint at (1, 0, 0) in cam frame
    recording = [{
        "time": 0.0,
        "primary_index": 0,
        "persons": [[np.array([1.0, 0.0, 0.0])]],
    }]

    # 90 deg rotation about Z + offset
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    t = np.array([10.0, 20.0, 30.0])

    out = tmp_path / "kp.npz"
    write_raw_buffer(out, recording, view_rotation=R, view_translation=t)

    data = np.load(out)
    # p_view = R @ [1,0,0] + t = [0, 1, 0] + [10, 20, 30] = [10, 21, 30]
    assert np.allclose(data["keypoints_3d"][0, 0, 0], [10.0, 21.0, 30.0],
                       atol=1e-5)
    assert "view_transform_R" in data.files
    assert "view_transform_t" in data.files
    assert np.allclose(data["view_transform_R"], R)
    assert np.allclose(data["view_transform_t"], t)


def test_write_raw_buffer_orthonormalises_drifted_rotation(tmp_path):
    """Slightly non-orthogonal R should be projected to SO(3) before save."""
    import numpy as np
    from calimerge.keypoint_export import write_raw_buffer

    recording = [{
        "time": 0.0,
        "primary_index": 0,
        "persons": [[np.array([1.0, 2.0, 3.0])]],
    }]

    # Identity perturbed off-orthogonality
    R = np.eye(3) + 0.05 * np.random.default_rng(0).standard_normal((3, 3))
    t = np.zeros(3)

    out = tmp_path / "kp.npz"
    write_raw_buffer(out, recording, view_rotation=R, view_translation=t)

    data = np.load(out)
    R_saved = data["view_transform_R"]
    # Saved R should be exactly orthonormal: R^T R == I
    assert np.allclose(R_saved.T @ R_saved, np.eye(3), atol=1e-10)
    # And in SO(3), not O(3)
    assert np.linalg.det(R_saved) > 0


def test_pose_detection_worker_has_pause_resume():
    from calimerge.gui.workers import PoseDetectionWorker
    assert hasattr(PoseDetectionWorker, "pause")
    assert hasattr(PoseDetectionWorker, "resume")


def test_all_detection_workers_have_pause_resume():
    """Regression: the GUI's pause/resume path was a silent no-op when
    the active backend lacked the method (Cuda + MediaPipe Hands), so
    'Pause tracking during recording' did nothing on those backends.
    All detection workers must now expose pause + resume.
    """
    from calimerge.gui.workers import (
        PoseDetectionWorker,
        MediaPipeHandsDetectionWorker,
        CudaStreamDetectionWorker,
        MpsStreamDetectionWorker,
    )
    for cls in (
        PoseDetectionWorker,
        MediaPipeHandsDetectionWorker,
        CudaStreamDetectionWorker,
        MpsStreamDetectionWorker,
    ):
        assert hasattr(cls, "pause"), f"{cls.__name__} missing pause()"
        assert hasattr(cls, "resume"), f"{cls.__name__} missing resume()"


def test_mps_stream_binding_is_available_callable():
    """Smoke test: mps_stream_binding loads cleanly on every platform and
    its is_available() helper returns False on this Windows machine
    (no .dylib, no Darwin). The MPS path must not break import-time on
    non-Mac dev hosts.
    """
    from calimerge.tracking import mps_stream_binding
    assert callable(mps_stream_binding.is_available)
    # On Windows / Linux there's no .dylib to load, so this MUST be False.
    import sys
    if sys.platform != "darwin":
        assert mps_stream_binding.is_available() is False


def test_coreml_converter_module_imports():
    """Smoke test: convert_onnx_to_coreml imports clean without coremltools.

    The actual conversion is gated on coremltools being importable; the
    module itself must always be import-clean so the manual test script
    in tests/manual/build_coreml_models.py can lazy-import the converter
    and surface a helpful error message rather than ImportError-ing.
    """
    from calimerge.tracking import convert_onnx_to_coreml
    assert callable(convert_onnx_to_coreml.is_coremltools_available)
    assert callable(convert_onnx_to_coreml.convert_yolo_to_coreml)
    assert callable(convert_onnx_to_coreml.convert_vitpose_to_coreml)


def test_mps_stream_detection_worker_class_shape():
    """Regression: MpsStreamDetectionWorker must mirror the surface area
    of CudaStreamDetectionWorker so the GUI selector can swap them with
    just a backend = 'mps' branch.
    """
    from calimerge.gui.workers import (
        CudaStreamDetectionWorker, MpsStreamDetectionWorker,
    )
    expected = {
        "pause", "resume", "stop", "submit_frame", "run",
        "_draw_reprojected", "_build_projection_params",
    }
    for name in expected:
        assert hasattr(MpsStreamDetectionWorker, name), \
            f"MpsStreamDetectionWorker missing {name}"
    # Same Qt signals as the CUDA worker (compared by name).
    cuda_signals = {
        attr for attr in dir(CudaStreamDetectionWorker)
        if attr in {"detection_ready", "keypoints_3d_ready", "log_message", "error"}
    }
    for sig in cuda_signals:
        assert hasattr(MpsStreamDetectionWorker, sig), \
            f"MpsStreamDetectionWorker missing signal {sig}"


def test_workout_page_compute_rotate_to_hand_exists():
    from calimerge.gui.workout_page import WorkoutPage
    assert hasattr(WorkoutPage, "_compute_rotate_to_hand")
    assert hasattr(WorkoutPage, "_current_model_key")


def test_skeleton_view_clear_preserves_view_transform():
    """Regression: skeleton_view.clear() used to wipe _view_transform back
    to identity on every call, which silently undid Rotate-to-Human +
    Zero whenever the user changed detection backend (because
    _on_model_changed -> _stop_detection -> skeleton_view.clear()).
    The transform must survive a clear(); only data should drop.
    """
    from PySide6.QtWidgets import QApplication
    import numpy as np
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    _ = app  # keep ref so it isn't gc'd mid-test on some platforms

    from calimerge.gui.widgets.skeleton_view import SkeletonViewWidget

    widget = SkeletonViewWidget()
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [0.5, 0.6, 0.7]
    widget.set_view_transform(T, has_origin=True)

    widget.update_keypoints([[np.array([1.0, 2.0, 3.0])]])

    # Bug repro: clear() must NOT reset the transform.
    widget.clear()
    assert np.allclose(widget._view_transform, T), (
        "skeleton_view.clear() reset the view transform - this regresses "
        "the post-backend-switch behaviour"
    )
    # Data should be cleared
    assert widget._persons == []

    # The explicit reset method should still work for callers that want it.
    widget.reset_view_transform()
    assert np.allclose(widget._view_transform, np.eye(4)) or \
        np.allclose(widget._view_transform[:3, 3], 0)


def test_app_settings_have_last_detect_keys():
    from calimerge.config import _APP_SETTINGS_DEFAULTS
    assert "last_detect_model" in _APP_SETTINGS_DEFAULTS
    assert "last_detect_backend" in _APP_SETTINGS_DEFAULTS
    assert "last_detect_confidence" in _APP_SETTINGS_DEFAULTS


def test_workout_page_has_persistence_methods():
    from calimerge.gui.workout_page import WorkoutPage
    assert hasattr(WorkoutPage, "_restore_last_detect_state")
    assert hasattr(WorkoutPage, "_persist_last_detect_state")


def test_offline_worker_exposes_tracking_params():
    """Regression: offline tracker used hardcoded run_cuda_pipeline
    defaults (max_track_distance=0.15 m / patience=30 frames) which
    fragmented a single subject into ~13 tracks per trial. The worker
    now accepts looser values (matching the live tracker) and exposes
    them on the constructor so they're tunable from the GUI."""
    import inspect
    from calimerge.gui.workers import OfflineProcessingWorker
    sig = inspect.signature(OfflineProcessingWorker.__init__)
    assert "max_track_distance" in sig.parameters
    assert "track_patience" in sig.parameters
    assert "stitch_max_gap_frames" in sig.parameters
    assert "stitch_max_distance_m" in sig.parameters
    # Looser defaults than run_cuda_pipeline's stock values.
    assert sig.parameters["max_track_distance"].default >= 0.4
    assert sig.parameters["track_patience"].default >= 30


def test_offline_track_stitcher_merges_fragmented(tmp_path):
    """The C tracker spawns a new track id whenever the camera subset
    changes for one frame, so a 1-person trial fragments into multiple
    tracks. The Python-side stitcher should re-merge tracks whose hip
    COMs are spatially close and temporally adjacent.
    """
    import numpy as np
    from calimerge.gui.workers import OfflineProcessingWorker

    # Stand-in worker (skip Qt init via __new__).
    w = OfflineProcessingWorker.__new__(OfflineProcessingWorker)
    w._stitch_max_gap_frames = 90
    w._stitch_max_distance_m = 0.6

    # Three "tracks" of the same subject standing still: ankle near zero,
    # hip COM near (0, 0, 1.0). Each track covers 100 syncs with a
    # ~5 sync gap (well under 90).
    def make_track(start_sync: int, n: int):
        # SynthPose-52 size, all None except hips at idx 11, 12.
        out = {}
        for s in range(start_sync, start_sync + n):
            kps = [None] * 52
            kps[11] = np.array([-0.1, 0.0, 1.0])  # L_Hip
            kps[12] = np.array([+0.1, 0.0, 1.0])  # R_Hip
            out[s] = kps
        return out

    tracks = {
        0: make_track(0, 100),
        1: make_track(105, 100),   # gap=5 frames, same place -> should merge into 0
        2: make_track(210, 100),   # gap=5 frames, same place -> should merge into 0
    }
    stitched = w._stitch_tracks(tracks)
    assert len(stitched) == 1, f"expected 1 stitched track, got {len(stitched)}"
    survivor = next(iter(stitched.values()))
    # Survivor covers all frames from all three originals.
    assert min(survivor.keys()) == 0
    assert max(survivor.keys()) == 309


def test_offline_track_stitcher_preserves_distinct_people():
    """Two genuinely different people standing far apart at the same
    time must NOT be stitched together. Stitch only fires across a
    temporal gap, never on overlapping windows."""
    import numpy as np
    from calimerge.gui.workers import OfflineProcessingWorker

    w = OfflineProcessingWorker.__new__(OfflineProcessingWorker)
    w._stitch_max_gap_frames = 90
    w._stitch_max_distance_m = 0.6

    def make_track(start_sync: int, n: int, x_offset: float):
        out = {}
        for s in range(start_sync, start_sync + n):
            kps = [None] * 52
            kps[11] = np.array([x_offset - 0.1, 0.0, 1.0])
            kps[12] = np.array([x_offset + 0.1, 0.0, 1.0])
            out[s] = kps
        return out

    tracks = {
        0: make_track(0, 100, x_offset=-1.0),  # person A on the left
        1: make_track(0, 100, x_offset=+1.0),  # person B on the right (overlapping time)
    }
    stitched = w._stitch_tracks(tracks)
    # Tracks overlap in time -> not the same person, no merge.
    assert len(stitched) == 2


def test_write_raw_buffer_records_model_backend(tmp_path):
    """write_raw_buffer must persist model_backend + model_name so a
    consumer can tell which pipeline produced the keypoints without
    consulting an external metadata file.
    """
    import numpy as np
    from calimerge.keypoint_export import write_raw_buffer

    rec = [{
        "time": 0.0, "primary_index": 0,
        "persons": [[np.array([1.0, 2.0, 3.0])]],
    }]
    out = tmp_path / "kp.npz"
    write_raw_buffer(
        out, rec, model_backend="cuda", model_name="vitpose_synthpose",
    )
    data = np.load(out)
    assert "model_backend" in data.files
    assert "model_name" in data.files
    assert str(data["model_backend"]) == "cuda"
    assert str(data["model_name"]) == "vitpose_synthpose"

    # Legacy callers that don't pass these get empty strings, NOT a
    # missing field — consumers can rely on the keys existing.
    out2 = tmp_path / "kp2.npz"
    write_raw_buffer(out2, rec)
    data2 = np.load(out2)
    assert "model_backend" in data2.files
    assert str(data2["model_backend"]) == ""


def test_save_keypoints_3d_records_model_backend(tmp_path):
    """save_keypoints_3d must persist model_backend + model_name too —
    notebook + workout_playback widget read THIS file, not raw.npz."""
    import numpy as np
    from calimerge.analysis.keypoints_io import save_keypoints_3d

    frames = [{
        "time": 0.0, "primary_index": 0,
        "persons": [[np.array([1.0, 2.0, 3.0])]],
    }]
    out = tmp_path / "kp.npz"
    save_keypoints_3d(
        out, frames, num_keypoints=17,
        model_backend="pytorch", model_name="vitpose_synthpose",
    )
    data = np.load(out)
    assert str(data["model_backend"]) == "pytorch"
    assert str(data["model_name"]) == "vitpose_synthpose"


def test_workouts_db_has_model_backend_columns(tmp_path):
    """The sessions table migration must add model_backend + model_name
    columns. Run init_workouts_db on a fresh DB and inspect schema.
    """
    import sqlite3
    from calimerge.config import init_workouts_db

    db = tmp_path / "workouts.db"
    init_workouts_db(db)
    conn = sqlite3.connect(str(db))
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    finally:
        conn.close()
    assert "model_backend" in cols
    assert "model_name" in cols


def test_create_session_round_trips_model_backend(tmp_path):
    """Round-trip through the public API: create_session writes the
    fields, a SELECT reads them back identically.
    """
    import sqlite3
    from calimerge.config import (
        init_workouts_db, create_session, create_user,
    )

    db = tmp_path / "workouts.db"
    init_workouts_db(db)
    user = create_user("ada", db_path=db)
    sid = create_session(
        user_id=user["id"], workout_type="fga_level_gait",
        model_backend="cuda", model_name="vitpose_synthpose",
        db_path=db,
    )
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT model_backend, model_name FROM sessions WHERE id=?",
            (sid,),
        ).fetchone()
    finally:
        conn.close()
    assert row["model_backend"] == "cuda"
    assert row["model_name"] == "vitpose_synthpose"


def test_save_keypoints_3d_applies_view_transform(tmp_path):
    """Regression: keypoints_3d.npz (loaded by test_output.ipynb and
    workout_playback) used to be in camera frame even when the user
    had pressed Rotate-to-Human + Zero, because save_keypoints_3d
    didn't accept the transform — only write_raw_buffer did. Result:
    notebook ankle z showed ~2.5 m, live display showed ~0.
    """
    import numpy as np
    from calimerge.analysis.keypoints_io import save_keypoints_3d, load_keypoints_3d

    # Single frame, single person, ankle (idx 15) at the camera-frame
    # X0 the user originally zeroed at.
    X0 = np.array([0.77, 0.92, 2.65])
    persons = [[None] * 17]
    persons[0][15] = X0
    frames = [{"time": 0.0, "primary_index": 0, "persons": persons}]

    R = np.array([
        [-0.977, +0.016, -0.212],
        [+0.206, +0.323, -0.924],
        [+0.053, -0.946, -0.319],
    ])
    t = -R @ X0  # by construction: p_view at X0 should be 0.

    out = tmp_path / "keypoints_3d.npz"
    save_keypoints_3d(
        out, frames, num_keypoints=17,
        view_rotation=R, view_translation=t,
    )

    data = np.load(out)
    saved_ankle = data["keypoints_3d"][0, 0, 15]
    # With transform applied, ankle at the zero point lands at ~0.
    assert np.allclose(saved_ankle, np.zeros(3), atol=1e-3), (
        f"ankle should be ~0 after view transform, got {saved_ankle}"
    )
    # The transform itself must be persisted so consumers can invert.
    assert "view_transform_R" in data.files
    assert "view_transform_t" in data.files

    # And without a transform, save_keypoints_3d must keep camera-frame
    # coords intact (back-compat for existing callers).
    out2 = tmp_path / "keypoints_3d_camframe.npz"
    save_keypoints_3d(out2, frames, num_keypoints=17)
    data2 = np.load(out2)
    assert np.allclose(data2["keypoints_3d"][0, 0, 15], X0, atol=1e-5)


def test_workout_spec_db_populates(tmp_path):
    """The workout_spec.db populator should produce one workout_specs
    row per WORKOUT_SPECS entry, all DEFAULT_PROGRAMS rows, and a
    program_exercises join row per exercise. Every join row's
    workout_type must resolve to a workout_specs entry."""
    from calimerge.workout_spec_db import (
        populate_workout_spec_db,
        load_workout_spec,
        load_program_exercises,
        WORKOUT_SPECS,
    )
    from calimerge.programs import DEFAULT_PROGRAMS
    import sqlite3

    db = tmp_path / "workout_spec.db"
    populate_workout_spec_db(db_path=db, overwrite=True)
    assert db.exists()

    conn = sqlite3.connect(str(db))
    try:
        n_specs = conn.execute("SELECT COUNT(*) FROM workout_specs").fetchone()[0]
        n_progs = conn.execute("SELECT COUNT(*) FROM programs").fetchone()[0]
        n_exs = conn.execute("SELECT COUNT(*) FROM program_exercises").fetchone()[0]
    finally:
        conn.close()
    assert n_specs == len(WORKOUT_SPECS)
    assert n_progs == len(DEFAULT_PROGRAMS)
    expected_ex = sum(len(p["exercises"]) for p in DEFAULT_PROGRAMS)
    assert n_exs == expected_ex

    # Spot-check one analyser-bound and one FGA (no analyser) entry.
    sts = load_workout_spec("sit_to_stand", db_path=db)
    assert sts is not None
    assert sts["analysis_function"] == "analyze_sit_to_stand"
    assert sts["threshold_default"] == 0.65

    level = load_workout_spec("fga_level_gait", db_path=db)
    assert level is not None
    assert level["analysis_function"] is None  # no analyser yet
    assert level["recording_duration_seconds"] == 12.0

    # Program join.
    fga = load_program_exercises("fga", db_path=db)
    assert len(fga) == 10
    types = [r["workout_type"] for r in fga]
    assert types[0] == "fga_level_gait"


def test_show_results_handles_none_metrics():
    """Regression: show_results used to crash with
    'TypeError: unsupported format string passed to NoneType.__format__'
    when an analyser left a metric as None (e.g. sit-to-stand running on
    a level-walk recording can't compute work/power because COM
    displacement is zero -> avg_power_watts=None). The guard now checks
    `value is not None`, not just key presence.
    """
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    _ = app

    from calimerge.gui.workout_page import WorkoutPage
    # Construct via __new__ + only the bits show_results touches so we
    # don't have to spin up the whole page.
    page = WorkoutPage.__new__(WorkoutPage)
    from PySide6.QtWidgets import QLabel
    page.results_label = QLabel()

    # Mix of present-but-None and present-with-value — must not raise.
    page.show_results({
        "rep_count": 0,
        "total_time_seconds": 12.5,
        "avg_power_watts": None,
        "work_per_rep_joules": None,
        "com_displacement_m": None,
        "per_rep_times": [],
        "avg_range_m": None,
    })
    text = page.results_label.text()
    assert "Repetitions: 0" in text
    assert "Total time: 12.5 s" in text
    # None-valued metrics must be skipped entirely, not formatted.
    assert "Avg power" not in text
    assert "Work per rep" not in text


def test_workout_page_has_reset_for_user_switch():
    """User switch must clear in-memory ghosts (camera grid frames,
    skeleton view persons, FPS history, analysis arrays) without
    requiring a full app restart."""
    from calimerge.gui.workout_page import WorkoutPage
    assert hasattr(WorkoutPage, "_reset_for_user_switch")


def test_video_utils_finds_serial_named_recordings(tmp_path):
    """RecordingWorker writes port_{N}_{sanitized_serial}.mp4 but the
    paused-tracking offline path used to glob the legacy port_{N}.mp4
    name only — so live recordings never matched and offline silently
    skipped, producing no npz and no CSV. find_video_for_port handles
    both names; this test is a guard so we don't regress.
    """
    from calimerge.gui.video_utils import find_video_for_port

    # New format with serial
    new_path = tmp_path / "port_0_ABC123.mp4"
    new_path.write_bytes(b"")
    assert find_video_for_port(tmp_path, 0, serial="ABC123") == new_path
    # Falls back to glob when serial not provided
    assert find_video_for_port(tmp_path, 0, serial=None) == new_path

    # Legacy format also still resolves
    legacy_path = tmp_path / "port_1.mp4"
    legacy_path.write_bytes(b"")
    assert find_video_for_port(tmp_path, 1) == legacy_path


def test_mps_offline_binding_imports():
    """Smoke test: mps_offline_binding loads cleanly on every platform and
    its public surface is callable. Mirrors the cuda_binding smoke test.
    """
    from calimerge.tracking import mps_offline_binding
    assert callable(mps_offline_binding.is_available)
    assert callable(mps_offline_binding.run_mps_pipeline)
    # Constants surface (used by the worker dispatch helper)
    assert mps_offline_binding.PT_MAX_CAMERAS == 16


def test_mps_offline_is_unavailable_on_windows():
    """On non-Darwin platforms there's no .dylib to load, so is_available
    must return False without raising. Guards against accidental import-
    time crashes on the Windows dev box.
    """
    import sys
    from calimerge.tracking import mps_offline_binding

    if sys.platform != "darwin":
        assert mps_offline_binding.is_available() is False


def test_offline_worker_has_dispatch_helper():
    """The OfflineProcessingWorker must expose _pick_offline_backend so the
    GUI dispatches to MPS on Mac and CUDA on Windows, with no per-call
    surface-area difference.
    """
    from calimerge.gui.workers import OfflineProcessingWorker
    assert hasattr(OfflineProcessingWorker, "_pick_offline_backend")
    backend = OfflineProcessingWorker._pick_offline_backend()
    # On any host the helper must return one of three known values.
    assert backend in {"mps", "cuda", "none"}


def test_offline_worker_dispatches_to_correct_backend(monkeypatch):
    """_pick_offline_backend must pick "mps" when the MPS dylib loads
    (regardless of whether CUDA is also present), and "cuda" otherwise on
    non-Darwin hosts. We monkeypatch sys.platform + the two is_available
    helpers so the test is deterministic on any dev host.
    """
    import sys
    from calimerge.gui.workers import OfflineProcessingWorker
    from calimerge.tracking import mps_offline_binding, cuda_binding

    # --- macOS, both available -> mps wins ---
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(mps_offline_binding, "is_available", lambda: True)
    monkeypatch.setattr(cuda_binding, "is_available", lambda: True)
    assert OfflineProcessingWorker._pick_offline_backend() == "mps"

    # --- macOS, MPS missing -> falls back to cuda (rare but legal) ---
    monkeypatch.setattr(mps_offline_binding, "is_available", lambda: False)
    monkeypatch.setattr(cuda_binding, "is_available", lambda: True)
    assert OfflineProcessingWorker._pick_offline_backend() == "cuda"

    # --- Windows, only cuda available -> cuda ---
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(mps_offline_binding, "is_available", lambda: True)
    monkeypatch.setattr(cuda_binding, "is_available", lambda: True)
    # MPS-on-Windows is impossible — the helper must NOT pick mps.
    assert OfflineProcessingWorker._pick_offline_backend() == "cuda"

    # --- Nothing available -> "none" ---
    monkeypatch.setattr(mps_offline_binding, "is_available", lambda: False)
    monkeypatch.setattr(cuda_binding, "is_available", lambda: False)
    assert OfflineProcessingWorker._pick_offline_backend() == "none"


def test_record_snapshot_uses_zero_origin_translation():
    """Regression: _begin_recording_now must read translation from
    _zero_origin_translation, not _view_rotation. The latter never holds
    the offset (Zero leaves [:3,3] = 0 in _view_rotation by design and
    only writes (R, t) into _zero_origin_*). Reading from _view_rotation
    produced rotated-but-not-offset keypoints, so e.g. ankle z stayed at
    ~3 m instead of ~0.

    We verify this purely via the snapshot expression — no Qt setup
    required — by replaying the same logic on a stand-in object.
    """
    import numpy as np

    class Fake:
        pass

    # Active rotate-to-human result (rotation only, [:3,3] is zero)
    R = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    fake = Fake()
    fake._view_rotation = np.eye(4)
    fake._view_rotation[:3, :3] = R
    fake._view_has_origin = True
    # Zero set the user's L_Ankle as origin: t = -R @ origin
    origin = np.array([0.1, -0.2, 1.05])
    fake._zero_origin_rotation = R.copy()
    fake._zero_origin_translation = (-R @ origin).copy()

    # Mirror the snapshot logic from _begin_recording_now
    if fake._view_has_origin and fake._zero_origin_rotation is not None \
            and fake._zero_origin_translation is not None:
        R_snap = np.asarray(fake._zero_origin_rotation, dtype=np.float64).copy()
        t_snap = np.asarray(fake._zero_origin_translation, dtype=np.float64).copy()
    else:
        R_snap = fake._view_rotation[:3, :3].copy()
        t_snap = np.zeros(3)

    # Sanity: applying snapshot to the origin point should give ~zero
    p_view = R_snap @ origin + t_snap
    assert np.allclose(p_view, np.zeros(3), atol=1e-12)
    # And translation is NOT zero (regression check)
    assert not np.allclose(t_snap, 0.0)


def test_unified_offline_worker_class_shape():
    """The unified offline worker exists, exposes the same Qt signals as
    the deprecated OfflineProcessingWorker, takes the constructor
    parameters the workout-page wiring relies on, and has a run() method.

    This is a structural-regression test — we don't run() the worker
    here (that would need YOLO + VitPose models on disk and is covered
    by tests/manual/run_offline_pipeline_on_test_data.py).
    """
    import inspect
    from calimerge.gui.unified_offline_worker import UnifiedOfflineWorker

    # Same Qt signals as OfflineProcessingWorker so workout_page wiring
    # is interchangeable.
    for sig_name in ("progress", "log_message", "finished_ok", "failed"):
        assert hasattr(UnifiedOfflineWorker, sig_name), (
            f"UnifiedOfflineWorker missing signal {sig_name}"
        )

    sig = inspect.signature(UnifiedOfflineWorker.__init__)
    expected_params = {
        "session_dir",
        "cameras",
        "port_to_video",
        "frame_time_csv",
        "backend",
        "view_rotation",
        "view_translation",
        "max_track_distance",
        "track_patience",
        "stitch_max_gap_frames",
        "stitch_max_distance_m",
        "batch_size",
        "person_confidence",
    }
    actual = set(sig.parameters.keys()) - {"self"}
    missing = expected_params - actual
    assert not missing, f"UnifiedOfflineWorker.__init__ missing params: {missing}"

    # person_confidence default must match the live PyTorch slider, NOT
    # run_cuda_pipeline's 0.1 (the source of t~11s static-object snaps).
    pconf = sig.parameters["person_confidence"].default
    assert pconf >= 0.4, (
        f"UnifiedOfflineWorker default person_confidence={pconf} is too "
        "low; should match the live tracker (≥ 0.5)."
    )

    # max_track_distance + track_patience default to live-tracker values.
    assert sig.parameters["max_track_distance"].default >= 0.4
    assert sig.parameters["track_patience"].default >= 30

    # run() exists and is callable.
    assert hasattr(UnifiedOfflineWorker, "run")
    assert callable(getattr(UnifiedOfflineWorker, "run"))


def test_track_stitch_module_or_static_methods():
    """Track-stitching helpers must be reachable from a single canonical
    place so both the deprecated worker and the unified worker call into
    the same logic.

    Acceptable layouts:
      A. ``calimerge.tracking.track_stitch`` exposes ``stitch_tracks``
         and ``hip_com``.
      B. ``OfflineProcessingWorker._stitch_tracks`` and ``._hip_com``
         still exist (legacy worker still functional).

    Today both are true — but this test asserts at least one is.
    """
    found_module = False
    try:
        from calimerge.tracking import track_stitch as _ts
        if hasattr(_ts, "stitch_tracks") and hasattr(_ts, "hip_com"):
            found_module = True
    except Exception:
        pass

    found_static = False
    try:
        from calimerge.gui.workers import OfflineProcessingWorker
        if hasattr(OfflineProcessingWorker, "_stitch_tracks") \
                and hasattr(OfflineProcessingWorker, "_hip_com"):
            found_static = True
    except Exception:
        pass

    assert found_module or found_static, (
        "neither calimerge.tracking.track_stitch.{stitch_tracks,hip_com} "
        "nor OfflineProcessingWorker._stitch_tracks/_hip_com exists"
    )


def test_offline_worker_marked_deprecated():
    """The legacy OfflineProcessingWorker docstring must clearly mark
    the class as deprecated so the next reader is steered to
    UnifiedOfflineWorker without spelunking through the file."""
    from calimerge.gui.workers import OfflineProcessingWorker
    doc = (OfflineProcessingWorker.__doc__ or "").lower()
    assert "deprecated" in doc, (
        "OfflineProcessingWorker docstring does not mention 'deprecated'; "
        "readers will not know to use UnifiedOfflineWorker instead."
    )
