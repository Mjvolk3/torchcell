#!/usr/bin/env bash
# scripts/run-in-env.sh
# [[scripts.run-in-env]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/run-in-env.sh
#
# pre-commit wrapper: run `<cmd> [args...]` with the torchcell conda env's bin/ first on
# PATH, whether or not the env is activated at commit time. pre-commit's `entry:` is not
# shell-interpreted, so `$HOME`/`~` cannot expand there; this resolves the env itself
# (same convention as scripts/run-mypy.sh: ~/miniconda3/envs/torchcell, CLAUDE.local.md).
#
#   entry: bash scripts/run-in-env.sh python scripts/test_quality_check.py
set -euo pipefail
export PATH="$HOME/miniconda3/envs/torchcell/bin:$PATH"
exec "$@"
