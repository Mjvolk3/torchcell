#!/bin/bash
# experiments/019-simb-multimodal/scripts/sync_delta_fig3_core.sh
# Mirror the built `fig3_core` dataset tree (~13 GB, serves expression + morphology) from
# GilaHyper -> Delta (NCSA) scratch, for the controlled expr<->morph joint sweep.
#
# RUN FROM GilaHyper. Target path MUST match run_training's dataset_root on Delta:
# $DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig3_core, where on Delta
# DATA_ROOT=/scratch/bbub/mjvolk3/torchcell.
#
#   bash experiments/019-simb-multimodal/scripts/sync_delta_fig3_core.sh
#   DELTA_HOST=dt-login02.delta.ncsa.illinois.edu bash .../sync_delta_fig3_core.sh
set -euo pipefail

# --- Source (GilaHyper) ---
GH_DATA_ROOT="${DATA_ROOT:-/scratch/projects/torchcell-scratch}"
REL="data/torchcell/experiments/019-simb-multimodal/fig3_core"
SRC="$GH_DATA_ROOT/$REL"

# --- Destination (Delta) ---
DELTA_USER="${DELTA_USER:-mjvolk3}"
DELTA_HOST="${DELTA_HOST:-login.delta.ncsa.illinois.edu}"
# Must equal DATA_ROOT in the cloned repo's .env on Delta (default = co-located with the
# torchcell clone at /projects/bbhh/mjvolk3). Change to bbtp / a scratch path if you relocate.
DELTA_DATA_ROOT="${DELTA_DATA_ROOT:-/projects/bbhh/mjvolk3/torchcell}"
DEST_DIR="$DELTA_DATA_ROOT/$REL"

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: source not found: $SRC" >&2
  exit 1
fi

echo "== fig3_core sync GilaHyper -> Delta =="
echo "  src : $SRC  ($(du -sh "$SRC" | cut -f1))"
echo "  dest: $DELTA_USER@$DELTA_HOST:$DEST_DIR"
echo "  NOTE: Delta uses Duo 2FA — approve the ONE push prompt when it appears."

# Single SSH connection = a SINGLE Duo approval: create the dest dir inside the rsync's own
# remote session via --rsync-path (mkdir -p ... && rsync), instead of a separate ssh mkdir.
rsync -aP --human-readable \
  --rsync-path="mkdir -p '$DEST_DIR' && rsync" \
  "$SRC/" "$DELTA_USER@$DELTA_HOST:$DEST_DIR/"

echo "== done. Verify on Delta: ls '$DEST_DIR' =="
