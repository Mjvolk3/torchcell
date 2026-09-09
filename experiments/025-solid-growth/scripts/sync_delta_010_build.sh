#!/bin/bash
# experiments/025-solid-growth/scripts/sync_delta_010_build.sh
# [[experiments.025-solid-growth.scripts.delta_cgt]]
#
# Mirror the 010 build (1.5 GB) from GilaHyper -> Delta, so the graph-regularization sweep
# can run there. The 025 full build is 3.2 TB and stays here; the 010 build holds the same
# 376,732 records with bit-identical labels (results/label_parity_010_vs_025.json).
#
#   RUN FROM GilaHyper, from the repo root. Delta uses Duo 2FA: approve ONE push prompt.
#   bash experiments/025-solid-growth/scripts/sync_delta_010_build.sh
#
# Destination MUST be the dataset_root the training script derives on Delta:
#   $DATA_ROOT/data/torchcell/experiments/010-kuzmin-tmi/001-small-build-schema-v2
# with Delta's DATA_ROOT=/scratch/bbub/mjvolk3/torchcell (the large space, not /projects).
#
# The graph roots the script also needs (data/sgd/genome, data/go, data/string, data/tflink)
# were shipped for the 019 campaign; delta_preflight_025.sh checks they are still there.
set -euo pipefail

GH_DATA_ROOT="${DATA_ROOT:-/scratch/projects/torchcell-scratch}"
REL="data/torchcell/experiments/010-kuzmin-tmi/001-small-build-schema-v2"
SRC="$GH_DATA_ROOT/$REL"

DELTA_USER="${DELTA_USER:-mjvolk3}"
DELTA_HOST="${DELTA_HOST:-login.delta.ncsa.illinois.edu}"
DELTA_DATA_ROOT="${DELTA_DATA_ROOT:-/scratch/bbub/mjvolk3/torchcell}"
DEST_DIR="$DELTA_DATA_ROOT/$REL"

[[ -d "$SRC/processed/lmdb" ]] || { echo "ERROR: no LMDB at $SRC/processed/lmdb" >&2; exit 1; }

echo "== 010 build sync GilaHyper -> Delta =="
echo "  src : $SRC  ($(du -sh "$SRC" | cut -f1))"
echo "  dest: $DELTA_USER@$DELTA_HOST:$DEST_DIR"

# One SSH connection = one Duo approval: create the dest dir inside the rsync's own remote
# session. The stale per-run caches are left behind on purpose; the 025 script writes its
# own tagged index files into data_module_cache.
rsync -aP --human-readable \
  --exclude='data_module_cache/perturbation_subset_*' \
  --exclude='*.lock' \
  --rsync-path="mkdir -p '$DEST_DIR' && rsync" \
  "$SRC/" "$DELTA_USER@$DELTA_HOST:$DEST_DIR/"

echo "== done. On Delta: bash experiments/025-solid-growth/scripts/delta_preflight_025.sh =="
