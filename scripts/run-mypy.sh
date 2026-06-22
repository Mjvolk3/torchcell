#!/usr/bin/env bash
# pre-commit mypy wrapper.
#
# Runs strict mypy in the torchcell conda env regardless of whether the env is
# PATH-activated at commit time. pre-commit's `entry:` is not shell-interpreted,
# so `$HOME`/`~` cannot expand there; this wrapper resolves the env path itself.
# The env path follows the ~/miniconda3/envs/torchcell convention (CLAUDE.local.md),
# so $HOME makes it portable across machines (/home vs /Users) without a hardcode.
#
# --follow-imports=silent scopes reported errors to the staged files pre-commit
# passes (you fix what you touch); the full strict run lives in the /mypy skill + CI.
set -euo pipefail
exec "$HOME/miniconda3/envs/torchcell/bin/mypy" \
  --config-file=pyproject.toml --follow-imports=silent "$@"
