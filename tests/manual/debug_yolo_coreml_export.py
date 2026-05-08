"""
Direct Ultralytics YOLO → CoreML export, no wrapper.

Diagnostic: a previous wrapped run had the MIL pipeline complete but no
mlpackage materialised on disk. This script does the bare minimum so we
can see exactly where it lands and what error (if any) the export raises.

Run:
    uv run --with coremltools python3 tests/manual/debug_yolo_coreml_export.py
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PT_PATH = REPO_ROOT / "models" / "yolo" / "yolov10s.pt"


def main() -> int:
    print(f"[debug] PT source: {PT_PATH}  exists={PT_PATH.exists()}", flush=True)

    from ultralytics import YOLO
    model = YOLO(str(PT_PATH))
    print("[debug] YOLO loaded; calling export...", flush=True)

    out = model.export(
        format="coreml",
        imgsz=640,
        nms=False,
        dynamic=True,
        batch=16,
        half=True,
    )
    print(f"[debug] export() returned: {out!r}", flush=True)

    p = Path(out)
    print(f"[debug] Path.exists: {p.exists()}", flush=True)
    if p.exists():
        if p.is_dir():
            entries = sorted(x.name for x in p.iterdir())
            print(f"[debug] mlpackage children: {entries}", flush=True)
        print(f"[debug] absolute path: {p.resolve()}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
