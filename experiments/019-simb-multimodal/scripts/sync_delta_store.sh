#!/usr/bin/env bash
# experiments/019-simb-multimodal/scripts/sync_delta_store.sh
# [[experiments.019-simb-multimodal.scripts.sync_delta_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/sync_delta_store
#
# Push one built 019 store from GilaHyper to Delta. Generalizes sync_delta_fig3_core.sh,
# whose destination root (/work/hdd) disagreed with the root every Delta launcher reads
# (/scratch/bbub/mjvolk3/torchcell, delta_grid_common.sh): a store synced to the wrong root
# makes a compute node try to REBUILD it with no Neo4j. The default here is the launchers'.
#
#   bash experiments/019-simb-multimodal/scripts/sync_delta_store.sh fig3_proteome
#
# Delta is Duo-gated: the user runs this and approves ONE push prompt. 14 GB for
# fig3_proteome (processed LMDB plus the data_module_cache that holds the split indices for
# seeds 0 to 11, which must travel with it so Delta draws the same partitions).
set -euo pipefail

STORE="${1:?usage: sync_delta_store.sh <store>  (fig3_core | fig3_proteome | ...)}"
GH_DATA_ROOT="${DATA_ROOT:-/scratch/projects/torchcell-scratch}"
REL="data/torchcell/experiments/019-simb-multimodal/$STORE"
SRC="$GH_DATA_ROOT/$REL"
DELTA_USER="${DELTA_USER:-mjvolk3}"
DELTA_HOST="${DELTA_HOST:-login.delta.ncsa.illinois.edu}"
DELTA_DATA_ROOT="${DELTA_DATA_ROOT:-/scratch/bbub/mjvolk3/torchcell}"
DEST_DIR="$DELTA_DATA_ROOT/$REL"

if [[ ! -d "$SRC/processed" ]]; then
  echo "ERROR: $SRC has no processed/ directory; nothing built to sync" >&2
  exit 1
fi
echo "== $STORE sync GilaHyper -> Delta =="
echo "  src : $SRC  ($(du -sh "$SRC" | cut -f1))"
echo "  dest: $DELTA_USER@$DELTA_HOST:$DEST_DIR"
echo "  NOTE: Delta uses Duo 2FA; approve the ONE push prompt when it appears."
rsync -aP --human-readable \
  --rsync-path="mkdir -p '$DEST_DIR' && rsync" \
  "$SRC/" "$DELTA_USER@$DELTA_HOST:$DEST_DIR/"
echo "== done. Verify on Delta: ls '$DEST_DIR/processed' '$DEST_DIR/data_module_cache' =="
