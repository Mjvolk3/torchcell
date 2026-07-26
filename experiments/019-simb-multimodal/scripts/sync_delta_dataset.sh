#!/bin/bash
# experiments/019-simb-multimodal/scripts/sync_delta_dataset.sh
# [[plan.cgt-metabolism.2026.07.25]]
# Mirror ONE built 019 dataset tree from GilaHyper -> Delta (NCSA).
#
# Generalizes `sync_delta_fig3_core.sh` (which hardcodes fig3_core) so every new build --
# fig6_pigment_transfer, fig6_build, fig3_core -- uses one code path.
#
#   RUN FROM GilaHyper.
#   bash experiments/019-simb-multimodal/scripts/sync_delta_dataset.sh fig6_pigment_transfer
#   DELTA_HOST=dt-login02.delta.ncsa.illinois.edu bash ... fig6_pigment_transfer
#
# The destination MUST match the dataset_root the training script derives on Delta:
# $DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/<tag>, where Delta's .env has
# DATA_ROOT=/work/hdd/bbub/mjvolk3/torchcell -- the large /work/hdd space, NOT /projects
# (tight quota).
set -euo pipefail

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  echo "usage: $0 <dataset_tag>   e.g. fig6_pigment_transfer" >&2
  exit 2
fi

GH_DATA_ROOT="${DATA_ROOT:-/scratch/projects/torchcell-scratch}"
REL="data/torchcell/experiments/019-simb-multimodal/$TAG"
SRC="$GH_DATA_ROOT/$REL"

DELTA_USER="${DELTA_USER:-mjvolk3}"
DELTA_HOST="${DELTA_HOST:-login.delta.ncsa.illinois.edu}"
DELTA_DATA_ROOT="${DELTA_DATA_ROOT:-/work/hdd/bbub/mjvolk3/torchcell}"
DEST_DIR="$DELTA_DATA_ROOT/$REL"

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: source not found: $SRC" >&2
  exit 1
fi

echo "== $TAG sync GilaHyper -> Delta =="
echo "  src : $SRC  ($(du -sh "$SRC" | cut -f1))"
echo "  dest: $DELTA_USER@$DELTA_HOST:$DEST_DIR"
echo "  NOTE: Delta uses Duo 2FA -- approve the ONE push prompt when it appears."

# Single SSH connection = a SINGLE Duo approval: create the dest dir inside the rsync's own
# remote session via --rsync-path, instead of a separate ssh mkdir.
rsync -aP --human-readable \
  --rsync-path="mkdir -p '$DEST_DIR' && rsync" \
  "$SRC/" "$DELTA_USER@$DELTA_HOST:$DEST_DIR/"

echo "== done. Verify on Delta: ls '$DEST_DIR' =="
