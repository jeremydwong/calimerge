"""Smoke test for notebooks/test_output.ipynb.

Catches the kinds of accidents that would silently break the student-facing
notebook: invalid JSON, missing the inlined SynthPose schema (which is the
whole point — making it standalone), or removing the cells that read the
npz layout Calimerge actually writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "test_output.ipynb"


def test_notebook_exists_and_is_valid_json():
    assert NOTEBOOK.is_file(), f"missing {NOTEBOOK}"
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert isinstance(data["cells"], list) and data["cells"]


def test_notebook_inlines_synthpose_schema():
    """The notebook must NOT import from `calimerge` — it has to keep working
    when the package is uninstalled / refactored. The SynthPose-52 dict and
    the side-classification helper must be present in source form."""
    src = NOTEBOOK.read_text(encoding="utf-8")
    assert "SYNTHPOSE_MARKERS" in src
    assert "L_Ankle" in src and "R_Ankle" in src
    assert "from calimerge" not in src and "import calimerge" not in src


def test_notebook_has_expected_sections():
    """Lock the four sections the user asked for."""
    src = NOTEBOOK.read_text(encoding="utf-8").lower()
    assert "file browser" in src or "filedialog" in src
    assert "skeleton" in src
    assert "foot position" in src or "ankle position" in src
    assert "frame-skip" in src or "inter-frame dt" in src


def test_notebook_reads_calimerge_npz_keys():
    """Make sure the loader cell still reads the keys Calimerge actually
    writes in keypoint_export.write_raw_buffer. If we ever rename one in
    the writer, this test plus the notebook need updating in lockstep."""
    src = NOTEBOOK.read_text(encoding="utf-8")
    for key in ("timestamps", "keypoints_3d", "person_count",
                "primary_person_index"):
        assert key in src, f"notebook doesn't reference npz key {key!r}"


def test_notebook_imports_only_stdlib_and_well_known_libs():
    """Survival check: nothing exotic that students would have to chase."""
    src = NOTEBOOK.read_text(encoding="utf-8")
    forbidden = ["calimerge", "torch", "PySide6", "cv2", "scipy"]
    for name in forbidden:
        assert f"import {name}" not in src, f"unexpected import: {name}"
        assert f"from {name}" not in src, f"unexpected from-import: {name}"


def test_notebook_has_inlined_skeleton_bones():
    """Bones list must be in the notebook (otherwise the skeleton plot is
    just a dot cloud). Keep this in lockstep with the inlined schema."""
    src = NOTEBOOK.read_text(encoding="utf-8")
    assert "SKELETON_BONES" in src


@pytest.mark.parametrize("expected_marker", [
    "Nose", "L_Eye", "R_Eye", "L_Hip", "R_Hip",
    "L_Ankle", "R_Ankle", "sternum", "C7",
])
def test_inlined_marker_names_present(expected_marker):
    """Sanity-check the inlined marker dict against the canonical SynthPose-52
    schema names. If somebody edits the notebook and accidentally drops a
    keypoint name, this catches it."""
    src = NOTEBOOK.read_text(encoding="utf-8")
    assert expected_marker in src
