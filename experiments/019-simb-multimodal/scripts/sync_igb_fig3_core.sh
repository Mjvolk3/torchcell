#!/bin/bash
# experiments/019-simb-multimodal/scripts/sync_igb_fig3_core.sh
# Mirror the built `fig3_core` dataset tree (LMDB + splits, ~13 GB — serves BOTH expression
# and morphology) from GilaHyper -> IGB BioCluster scratch, so the OFFLINE IGB compute nodes
# can train without any Neo4j access (Neo4jCellDataset skips the DB when processed/ exists).
#
# RUN FROM GilaHyper (the source machine). Creates the mirror dir tree on IGB first, then
# rsync's (resumable, no --delete). The IGB target path MUST match how run_training resolves
# dataset_root: $DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig3_core  (on IGB,
# DATA_ROOT=/home/a-m/mjvolk3/scratch/torchcell).
#
#   bash experiments/019-simb-multimodal/scripts/sync_igb_fig3_core.sh
#   IGB_HOST=biologin-2 bash .../sync_igb_fig3_core.sh    # override login node
set -euo pipefail

# --- Source (GilaHyper) ---
GH_DATA_ROOT="${DATA_ROOT:-/scratch/projects/torchcell-scratch}"
REL="data/torchcell/experiments/019-simb-multimodal/fig3_core"
SRC="$GH_DATA_ROOT/$REL"

# --- Destination (IGB) ---
# From GilaHyper only the FQDN resolves (the `biologin` alias / `biologin-3` short name live
# in the LAPTOP's ~/.ssh/config and inside the IGB network respectively). One-time setup on
# GilaHyper: `ssh mjvolk3@biologin.igb.illinois.edu` to accept the host key + confirm key auth.
IGB_USER="${IGB_USER:-mjvolk3}"
IGB_HOST="${IGB_HOST:-biologin.igb.illinois.edu}"
IGB_DATA_ROOT="${IGB_DATA_ROOT:-/home/a-m/mjvolk3/scratch/torchcell}"
DEST_DIR="$IGB_DATA_ROOT/$REL"

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: source not found: $SRC" >&2
  exit 1
fi

echo "== fig3_core sync GilaHyper -> IGB =="
echo "  src : $SRC  ($(du -sh "$SRC" | cut -f1))"
echo "  dest: $IGB_USER@$IGB_HOST:$DEST_DIR"

# Create the mirror tree on IGB (parent of fig3_core so rsync lands the dir in place).
ssh "$IGB_USER@$IGB_HOST" "mkdir -p '$(dirname "$DEST_DIR")'"

# Trailing slash on SRC copies its CONTENTS into DEST_DIR. -a preserves times/perms so the
# processed/ LMDB and data_module_cache/ splits arrive identical; -P resumes partial files.
rsync -aP --human-readable \
  "$SRC/" "$IGB_USER@$IGB_HOST:$DEST_DIR/"

echo "== done. Verify on IGB: ls '$DEST_DIR' =="
