#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/alcf/transfer_043.sh
# [[experiments.043-expression-proteome-morphology.scripts.alcf.transfer_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/transfer_043.sh
#
# Send the 043 data (and the code) to one ALCF machine over the owner's ControlMaster
# socket, then verify it there. Run from the repository root (the worktree):
#
#   bash experiments/043-expression-proteome-morphology/scripts/alcf/transfer_043.sh polaris code
#   bash experiments/043-expression-proteome-morphology/scripts/alcf/transfer_043.sh polaris train
#   bash experiments/043-expression-proteome-morphology/scripts/alcf/transfer_043.sh polaris prepare
#   bash experiments/043-expression-proteome-morphology/scripts/alcf/transfer_043.sh polaris verify-train
#
# Data tiers come from transfer_manifest_043.py (files_<tier>_043.txt, sha256_<tier>_043.txt,
# sizes_<tier>_043.txt beside this script) and land under $TC_PREFIX/torchcell-data/043/,
# mirroring the $DATA_ROOT-relative layout, so a job sets DATA_ROOT to that directory.
#
# rsync never deletes on the remote side: no --delete, no --delete-excluded, for the data
# AND the code (the 042 sync_code.sh mirrors with --delete; this script keeps its excludes
# but not that flag, so a stale remote file can remain and nothing remote is removed).
# The code copy also skips torchcell/models/pretrained_LLM/ (22 GB of Hugging Face encoder
# weights, ProtT5 alone 2 x 11 GB): the 043 trainer reads the frozen allele tables instead.
# The socket is the owner's (MobilePASS+); when `ssh -O check` fails, this script stops.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
source "$REPO/experiments/042-hpc-smoke-tests/scripts/machines.sh" "${1:-}"
action="${2:-}"
LOCAL_DATA_ROOT=/scratch/projects/torchcell-scratch
DEST="$TC_PREFIX/torchcell-data/043"

ssh -O check "$TC_SSH_HOST" 2>/dev/null || {
  echo "socket for $TC_SSH_HOST is down: the owner runs 'ssh $TC_SSH_HOST' once" >&2
  exit 3
}

case "$action" in
  code)
    rsync -a --partial --human-readable --stats \
      --rsync-path="mkdir -p '$TC_TREE' && rsync" \
      --exclude '/.git/' --exclude '/.claude/worktrees/' --exclude '/notes/' \
      --exclude '/notes-tex/' --exclude '/paper/' --exclude '/data/' \
      --exclude '/database/data/' --exclude '/wandb/' --exclude '__pycache__/' \
      --exclude '.mypy_cache/' --exclude '.pytest_cache/' --exclude '.ruff_cache/' \
      --exclude '*.egg-info/' --exclude 'node_modules/' \
      --exclude '/torchcell/models/pretrained_LLM/' \
      --exclude 'experiments/*/results/' --exclude 'experiments/*/slurm/output/' \
      --exclude 'experiments/*/profiling_results/' --exclude '.env' --exclude '.env.vscode' \
      "$REPO/" "$TC_SSH_HOST:$TC_TREE/" | tail -4
    # the commit the remote tree was synced from (git is excluded from the copy)
    git -C "$REPO" rev-parse HEAD | ssh "$TC_SSH_HOST" "cat > '$TC_TREE/.synced_commit'"
    # PBS -o directory of the 043 smoke jobs (job output is excluded from the copy)
    ssh "$TC_SSH_HOST" "mkdir -p '$TC_TREE/experiments/043-expression-proteome-morphology/slurm/output'"
    ;;
  train|prepare)
    rsync -a --partial --info=progress2 --human-readable \
      --files-from="$HERE/files_${action}_043.txt" \
      --rsync-path="mkdir -p '$DEST' && rsync" \
      "$LOCAL_DATA_ROOT/" "$TC_SSH_HOST:$DEST/"
    ;;
  verify-train|verify-prepare)
    tier="${action#verify-}"
    echo "== sizes ($tier) on $TC_MACHINE"
    ssh "$TC_SSH_HOST" "cd '$DEST' && while read -r n p; do s=\$(stat -c %s \"\$p\" 2>/dev/null || echo missing); [ \"\$s\" = \"\$n\" ] || echo \"SIZE MISMATCH \$p local=\$n remote=\$s\"; done; echo sizes-checked" \
      < "$HERE/sizes_${tier}_043.txt"
    echo "== sha256 ($tier) on $TC_MACHINE"
    ssh "$TC_SSH_HOST" "cd '$DEST' && sha256sum -c --quiet - && echo sha256-ok" \
      < "$HERE/sha256_${tier}_043.txt"
    ;;
  *)
    echo "usage: $0 <polaris|aurora> <code|train|prepare|verify-train|verify-prepare>" >&2
    exit 2
    ;;
esac
