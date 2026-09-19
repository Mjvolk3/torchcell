#!/bin/bash
# experiments/029-solid-growth-ko/scripts/sync_igb_029_build.sh
# Mirror the 029 deletion-only build from GilaHyper -> IGB BioCluster scratch, so the
# OFFLINE IGB compute nodes can train on the 9,297,912-record dataset with no Neo4j.
# Same shape as experiments/025-solid-growth/scripts/sync_igb_025_build.sh, which
# explains the choices; only processed/ (547 GB) is sent, since that is the only stage
# a loader reads, and the empty raw/lmdb stub satisfies PyG's files_exist check.
#
# RUN FROM GilaHyper (or under gh_sync_igb_029.slurm). Resumable: rsync -aP, no --delete.
#
#   bash experiments/029-solid-growth-ko/scripts/sync_igb_029_build.sh
#   DRY_RUN=1 bash .../sync_igb_029_build.sh               # size the transfer, send nothing
set -euo pipefail

GH_DATA_ROOT="${DATA_ROOT:-/scratch/projects/torchcell-scratch}"
REL="data/torchcell/experiments/029-solid-growth-ko/001-ko-build"
SRC="$GH_DATA_ROOT/$REL"

IGB_USER="${IGB_USER:-mjvolk3}"
IGB_HOST="${IGB_HOST:-biologin.igb.illinois.edu}"
IGB_DATA_ROOT="${IGB_DATA_ROOT:-/home/a-m/mjvolk3/scratch/torchcell}"
DEST_DIR="$IGB_DATA_ROOT/$REL"

if [[ ! -d "$SRC/processed" ]]; then
  echo "ERROR: source not found: $SRC/processed" >&2
  exit 1
fi

echo "== 029 build sync GilaHyper -> IGB =="
echo "  src : $SRC -> $(readlink -f "$SRC")"
echo "        processed  $(du -sh "$SRC/processed" | cut -f1)"
echo "  dest: $IGB_USER@$IGB_HOST:$DEST_DIR"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "== DRY_RUN: nothing sent, no SSH opened =="
  exit 0
fi

CTL="${TMPDIR:-/tmp}/igb-029-sync-%r@%h:%p"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$CTL" -o ControlPersist=4h)
trap 'ssh "${SSH_OPTS[@]}" -O exit "$IGB_USER@$IGB_HOST" 2>/dev/null || true' EXIT

ssh "${SSH_OPTS[@]}" "$IGB_USER@$IGB_HOST" "mkdir -p '$DEST_DIR/raw/lmdb'"

echo "== rsync processed =="
rsync -aP --human-readable --info=progress2 \
  -e "ssh ${SSH_OPTS[*]}" \
  "$SRC/processed/" "$IGB_USER@$IGB_HOST:$DEST_DIR/processed/"

echo "== verifying: a second pass must report no transfers =="
rsync -a --itemize-changes --dry-run -e "ssh ${SSH_OPTS[*]}" \
  "$SRC/processed/" "$IGB_USER@$IGB_HOST:$DEST_DIR/processed/" | head -20

echo "== done =="
echo "On IGB, DATA_ROOT=$IGB_DATA_ROOT and the loader root resolves to:"
echo "  $DEST_DIR"
