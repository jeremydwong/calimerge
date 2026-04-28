"""
Smoke tests for the CSV-input notebook.

Covers Task 3:

* ``notebooks/test_output_csv.ipynb`` exists and is valid JSON.
* It has the four expected sections (file picker / load / skeleton plot
  / dt diagnostic / camera_frame_time_s comparison).
* It does not import the ``calimerge`` package (must work standalone).
* It references the new ``camera_frame_time_s`` column from Task 1 so a
  student reading it gets the full context.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "test_output_csv.ipynb"


def _load_notebook() -> dict:
    assert NOTEBOOK.exists(), f"missing notebook: {NOTEBOOK}"
    with open(NOTEBOOK, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Existence / validity
# ---------------------------------------------------------------------------


def test_notebook_exists_and_is_valid_json():
    nb = _load_notebook()
    assert "cells" in nb and isinstance(nb["cells"], list) and nb["cells"]
    assert nb.get("nbformat") == 4


def _all_source(nb: dict) -> str:
    """Concatenate every cell's source as a single string for substring search."""
    out: list[str] = []
    for cell in nb["cells"]:
        src = cell.get("source", "")
        if isinstance(src, list):
            out.extend(src)
        else:
            out.append(src)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Required sections
# ---------------------------------------------------------------------------


def test_notebook_has_four_expected_sections():
    """File picker, load, skeleton, foot trace, dt diagnostic, camera-time panel."""
    text = _all_source(_load_notebook())
    # File picker (tkinter)
    assert "filedialog" in text
    assert "askopenfilename" in text
    # Skeleton plot at start/middle/end frames
    assert "start" in text.lower() and "middle" in text.lower() and "end" in text.lower()
    assert "SKELETON_BONES" in text
    # Foot / ankle position over time
    assert "L_Ankle" in text and "R_Ankle" in text
    # Inter-frame dt diagnostic
    assert "median" in text and "skip" in text
    assert "dt" in text  # axis label or variable


def test_notebook_has_camera_frame_time_panel():
    text = _all_source(_load_notebook())
    # Must reference the new column from Task 1 so the student knows the
    # expected schema.
    assert "camera_frame_time_s" in text
    # The comparison panel description should also mention "time_s" so the
    # contrast is obvious.
    assert "time_s" in text


def test_notebook_documents_csv_header_at_top():
    """A student reading from cold should see the expected header up front."""
    nb = _load_notebook()
    # First handful of markdown cells should describe the CSV format.
    md_cells = [c for c in nb["cells"] if c.get("cell_type") == "markdown"]
    assert md_cells, "notebook has no markdown cells"
    first_md = "".join(
        s if isinstance(s, str) else "".join(s) for s in [md_cells[0].get("source", "")]
    )
    # Header columns should be enumerated in the opening markdown.
    for col in (
        "time_s",
        "sync_index",
        "person_index",
        "person_id",
        "kp_index",
        "valid",
        "camera_frame_time_s",
    ):
        assert col in first_md, f"column not documented in opening cell: {col}"


# ---------------------------------------------------------------------------
# Standalone-ness
# ---------------------------------------------------------------------------


def test_notebook_does_not_import_calimerge():
    text = _all_source(_load_notebook())
    # Both `import calimerge` and `from calimerge` would create a hard
    # dependency on the package.
    assert "import calimerge" not in text
    assert "from calimerge" not in text


def test_notebook_imports_only_safe_third_party():
    """Spot-check that the notebook sticks to numpy / pandas / matplotlib / stdlib."""
    text = _all_source(_load_notebook())
    # Required deps
    assert "import numpy" in text
    assert "import pandas" in text
    assert "matplotlib" in text
    # tkinter file picker
    assert "import tkinter" in text or "from tkinter" in text


# ---------------------------------------------------------------------------
# Don't accidentally edit the npz notebook
# ---------------------------------------------------------------------------


def test_csv_notebook_is_distinct_from_npz_notebook():
    """The new notebook must not be a literal copy that still says ``.npz``."""
    nb = _load_notebook()
    text = _all_source(nb)
    # The CSV notebook should reference CSV picking explicitly.
    assert "keypoints_3d.csv" in text or 'filetypes=[("CSV"' in text
