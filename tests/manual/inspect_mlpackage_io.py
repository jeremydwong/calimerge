"""
Print input + output specs of a CoreML .mlpackage.

Useful for verifying the converter produced what the C-side
pt_coreml.m expects (tensor names, shapes, dtypes).

Usage:
    uv run python3 tests/manual/inspect_mlpackage_io.py
       (defaults to both yolo + vitpose under models/coreml/)

    uv run python3 tests/manual/inspect_mlpackage_io.py path/to/X.mlpackage
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = [
    REPO_ROOT / "models/coreml/yolo_v10s.mlpackage",
    REPO_ROOT / "models/coreml/vitpose_synthpose.mlpackage",
]


def _describe(path: Path) -> None:
    import coremltools as ct
    print(f"\n=== {path.name} ===")
    if not path.exists():
        print(f"  not found: {path}")
        return
    spec = ct.models.MLModel(str(path)).get_spec()
    desc = spec.description
    print(f"  inputs:")
    for i in desc.input:
        print(f"    {i.name}: {i.type}")
    print(f"  outputs:")
    for o in desc.output:
        print(f"    {o.name}: {o.type}")


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] or DEFAULTS
    try:
        import coremltools  # noqa: F401
    except ImportError:
        print("coremltools not installed in this env. Run via:")
        print("  uv run --with coremltools python3 tests/manual/inspect_mlpackage_io.py")
        return 1
    for p in paths:
        _describe(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
