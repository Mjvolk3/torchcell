#!/bin/bash
# experiments/042-hpc-smoke-tests/scripts/sync_code.sh
# [[experiments.042-hpc-smoke-tests.scripts.sync_code]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/042-hpc-smoke-tests/scripts/sync_code
#
# Mirror THIS checkout (the worktree the script is run from) to a machine's TC_TREE.
#
#   bash experiments/042-hpc-smoke-tests/scripts/sync_code.sh polaris
#
# Code, configs and env files cross; notes, paper, images, data, results and job
# output stay home. The remote tree is job-owned: nothing on the remote machine
# edits it, and a later sync never reaches a running job that snapshotted it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/machines.sh" "${1:-}"
SRC="$(cd "$HERE/../../.." && pwd)"

echo "== sync $SRC -> $TC_MACHINE:$TC_TREE =="
# shellcheck disable=SC2086
rsync -a --delete --human-readable --stats \
  -e "ssh $TC_SSH_OPTS" \
  --rsync-path="mkdir -p '$TC_TREE' && rsync" \
  --exclude '.git/' \
  --exclude '.claude/worktrees/' \
  --exclude 'notes/' \
  --exclude 'notes-tex/' \
  --exclude 'paper/' \
  --exclude 'data/' \
  --exclude 'database/data/' \
  --exclude 'wandb/' \
  --exclude '__pycache__/' \
  --exclude '.mypy_cache/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '*.egg-info/' \
  --exclude 'node_modules/' \
  --exclude 'experiments/*/results/' \
  --exclude 'experiments/*/slurm/output/' \
  --exclude 'experiments/*/profiling_results/' \
  --exclude '.env' \
  --exclude '.env.vscode' \
  "$SRC/" "$TC_SSH_HOST:$TC_TREE/" | tail -4
