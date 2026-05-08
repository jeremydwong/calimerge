"""
Print every entry in the pose-model registry, with a column flagging
whether each fits today's C-side shape contract (CUDA/MPS-runnable).

Useful for spot-checking what TOML files in <app_data>/models/registry
or <repo>/models/registry are picking up.

Run:
    uv run python3 tests/manual/list_pose_models.py
"""

from __future__ import annotations

import sys

from calimerge.tracking.registry import all_specs, is_c_runnable


def main() -> int:
    specs = all_specs()
    if not specs:
        print("(empty registry)")
        return 1

    print(f"{'id':<22} {'display':<35} {'shape':>10} {'kp':>4} {'preproc':<14} {'C':>3}  hf_repo")
    print("-" * 110)
    for s in specs:
        c_ok = "OK" if is_c_runnable(s) else "—"
        shape = f"{s.input_shape[0]}x{s.input_shape[1]}"
        repo = s.hf_repo or "(local)"
        print(
            f"{s.id:<22} {s.display_name:<35} {shape:>10} {s.schema.K:>4} "
            f"{s.preprocess:<14} {c_ok:>3}  {repo}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
