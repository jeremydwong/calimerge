"""
MPS pipeline smoke test — no test data required.

Confirms:
  1. build/mps/libcalimerge_mps.dylib is loadable
  2. The offline + streaming Python bindings find it
  3. is_available() returns True for both
  4. The exported symbols (function pointer table) are wired

If anything here fails, the dylib build is broken — full integration tests
(models + recordings) won't help diagnose. Run this first.

Run:  uv run python3 tests/manual/check_mps_smoke.py
"""

from __future__ import annotations

import sys

from calimerge.tracking import mps_offline_binding, mps_stream_binding


def main() -> int:
    failed = 0

    # 1. Locate dylib
    offline_path = mps_offline_binding._find_dylib()
    stream_path = mps_stream_binding._find_mps_lib()
    print("[locate]")
    print(f"  offline binding sees: {offline_path}")
    print(f"  stream  binding sees: {stream_path}")
    if offline_path is None:
        print("  FAIL: offline binding cannot find libcalimerge_mps.dylib")
        failed += 1
    if stream_path is None:
        print("  FAIL: stream binding cannot find libcalimerge_mps.dylib")
        failed += 1

    # 2. Library loads
    print("\n[load]")
    try:
        mps_offline_binding._load_library()
        print("  offline _load_library() OK")
    except Exception as e:
        print(f"  FAIL offline load: {type(e).__name__}: {e}")
        failed += 1
    try:
        mps_stream_binding._load_lib()
        print("  stream _load_lib() OK")
    except Exception as e:
        print(f"  FAIL stream load: {type(e).__name__}: {e}")
        failed += 1

    # 3. is_available
    print("\n[is_available]")
    off_avail = mps_offline_binding.is_available()
    str_avail = mps_stream_binding.is_available()
    print(f"  offline.is_available() = {off_avail}")
    print(f"  stream.is_available()  = {str_avail}")
    if not off_avail:
        failed += 1
    if not str_avail:
        failed += 1

    # 4. Exported function table — call a stats fields lookup
    print("\n[stats fields]")
    try:
        fields = mps_offline_binding.get_pipeline_stats_fields()
        print(f"  {len(fields)} stats fields: {fields}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        failed += 1

    print("\n[summary]")
    if failed == 0:
        print("  PASS — MPS dylib is loaded and bindings are functional")
        return 0
    print(f"  FAIL — {failed} check(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
