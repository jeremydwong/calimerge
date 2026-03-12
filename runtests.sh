#!/usr/bin/env bash
# Usage: bash runtests.sh [pytest args...]
# Example: bash runtests.sh tests/test_extrinsic_real.py -v -s
VIRTUAL_ENV= ~/.local/bin/uv run pytest "$@"
