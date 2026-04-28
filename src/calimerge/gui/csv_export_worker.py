"""
QThread worker that turns a recorded session into ``keypoints_3d.csv``
plus ``keypoints_3d.meta.json``.

The heavy lifting lives in :mod:`calimerge.keypoint_export` -- this file
just shovels parameters in and emits Qt signals so the GUI can stay
responsive.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..keypoint_export import (
    FRAME_TIME_HISTORY_FILENAME,
    export_session_csv,
    read_raw_buffer,
)


class CsvExportWorker(QThread):
    """Run :func:`export_session_csv` off the GUI thread."""

    finished_ok = Signal(dict)        # {csv_path, meta_path, rows, session_dir}
    failed = Signal(str, str)          # session_dir, error message
    progress = Signal(str)             # human-readable status

    def __init__(
        self,
        session_dir: Path,
        recording_keypoints: list[dict] | None,
        *,
        calibrated_cameras=None,
        extrinsic_session_id: int | None = None,
        extrinsic_calibrated_at: str | None = None,
        model_backend: str | None = None,
        model_name: str | None = None,
        num_keypoints: int = 52,
        session_id: int | None = None,
        intrinsics_db_path: Path | None = None,
        raw_buffer_path: Path | None = None,
        extra_meta: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.session_dir = Path(session_dir)
        self._buffer = recording_keypoints
        self._raw_buffer_path = raw_buffer_path
        # Pre-compute the conventional frame_time_history path so the
        # worker can both surface it on the CSV/npz outputs *and* survive
        # session_dir relocation.
        history_candidate = self.session_dir / FRAME_TIME_HISTORY_FILENAME
        self._frame_time_history_path: Path | None = (
            history_candidate if history_candidate.exists() else None
        )
        self._kwargs = dict(
            calibrated_cameras=calibrated_cameras,
            extrinsic_session_id=extrinsic_session_id,
            extrinsic_calibrated_at=extrinsic_calibrated_at,
            model_backend=model_backend,
            model_name=model_name,
            num_keypoints=num_keypoints,
            session_id=session_id,
            intrinsics_db_path=intrinsics_db_path,
            extra_meta=extra_meta,
            frame_time_history_path=self._frame_time_history_path,
        )

    def run(self):  # noqa: D401 -- Qt convention
        try:
            buf = self._buffer
            if (buf is None or len(buf) == 0) and self._raw_buffer_path is not None:
                self.progress.emit(f"Loading raw buffer from {self._raw_buffer_path.name}")
                buf = read_raw_buffer(self._raw_buffer_path)

            if not buf:
                self.failed.emit(
                    str(self.session_dir),
                    "No keypoints to export (empty buffer).",
                )
                return

            self.progress.emit(
                f"Writing keypoints_3d.csv for {self.session_dir.name}..."
            )
            csv_path, meta_path, rows = export_session_csv(
                self.session_dir, buf, **self._kwargs
            )
            self.finished_ok.emit(
                {
                    "session_dir": str(self.session_dir),
                    "csv_path": str(csv_path),
                    "meta_path": str(meta_path),
                    "rows": int(rows),
                }
            )
        except Exception as exc:  # noqa: BLE001 -- surface to GUI
            import traceback

            self.failed.emit(str(self.session_dir), f"{exc}\n{traceback.format_exc()}")
