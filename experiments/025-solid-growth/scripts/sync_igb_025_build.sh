#!/bin/bash
# experiments/025-solid-growth/scripts/sync_igb_025_build.sh
# Mirror the 025 all-solid-growth build from GilaHyper -> IGB BioCluster scratch, so the
# OFFLINE IGB compute nodes can train on the 13,525,071-record dataset with no Neo4j.
#
# RUN FROM GilaHyper. Resumable: rsync -aP, no --delete, so an interrupted run continues
# where it stopped and a second pass is a cheap verification.
#
#   bash experiments/025-solid-growth/scripts/sync_igb_025_build.sh
#   IGB_HOST=biologin-2 bash .../sync_igb_025_build.sh     # override login node
#   DRY_RUN=1 bash .../sync_igb_025_build.sh               # size the transfer, send nothing
#
# WHAT IS SENT, AND WHY IT IS NOT THE WHOLE TREE. The build is 3.2 TB across five stages,
# of which only `processed/` (517 GB) is read at training time:
#
#     raw            839 G   Neo4j query output
#     conversion     837 G   CompositeFitnessConverter output
#     deduplication  517 G   MeanExperimentDeduplicator output
#     aggregation    517 G   GenotypeAggregator output
#     processed      517 G   <- the only stage a loader reads
#
# The four intermediates exist to resume an interrupted BUILD, and a build cannot be run
# on IGB anyway since there is no Neo4j there. Sending them would cost 2.7 TB to carry
# stages nothing can use. (The 019 precedent, sync_igb_fig3_core.sh, sent every stage,
# but fig3_core was 13 GB so the choice never came up.)
#
# THE ONE CATCH, and it is PyG's rather than torchcell's. `Dataset._download` returns early
# only when `files_exist(self.raw_paths)`, and `raw_paths` is `<root>/raw/lmdb`.
# Neo4jCellDataset does not override `download()`, so a missing raw/ reaches the base class
# and tries to build. An EMPTY `raw/lmdb` DIRECTORY satisfies the check, which is why this
# script creates one on the destination.
#
# VERIFIED on GilaHyper before writing this, against a root holding only a `processed/`
# symlink plus an empty `raw/lmdb`, with NEO4J_URI pointed at a dead port: the dataset
# opened at len=13,525,071 with perturbation counts 1:5,694 / 2:13,142,648 / 3:376,732 and
# returned a triple with its label. Nothing reached the database.
#
# `data_module_cache/` (54 MB) also goes: those are the computed split indices, keyed by
# the pinned-split and subset hashes. Carrying them means IGB does not recompute a split
# over 13.5 M records on first run.
#
# The SUBSET AND SPLIT ARTIFACTS are NOT sent here. subset_S*_indices.json.gz,
# query_pair_disjoint_splits_025.json.gz and pinned_splits_from_010_seed_42.json.gz live in
# experiments/025-solid-growth/results/ and are committed, so they arrive on IGB by
# `git pull`. Sending them by rsync would create a second copy that can silently disagree
# with the repo.
set -euo pipefail

# --- Source (GilaHyper) ---
GH_DATA_ROOT="${DATA_ROOT:-/scratch/projects/torchcell-scratch}"
REL="data/torchcell/experiments/025-solid-growth/001-full-build"
SRC="$GH_DATA_ROOT/$REL"

# --- Destination (IGB) ---
# From GilaHyper only the FQDN resolves; the `biologin` alias lives in the laptop's
# ~/.ssh/config and `biologin-3` inside the IGB network. One-time setup on GilaHyper:
# `ssh mjvolk3@biologin.igb.illinois.edu` to accept the host key and confirm key auth.
IGB_USER="${IGB_USER:-mjvolk3}"
IGB_HOST="${IGB_HOST:-biologin.igb.illinois.edu}"
IGB_DATA_ROOT="${IGB_DATA_ROOT:-/home/a-m/mjvolk3/scratch/torchcell}"
DEST_DIR="$IGB_DATA_ROOT/$REL"

for d in processed data_module_cache; do
  if [[ ! -d "$SRC/$d" ]]; then
    echo "ERROR: source not found: $SRC/$d" >&2
    exit 1
  fi
done

echo "== 025 build sync GilaHyper -> IGB =="
echo "  src : $SRC"
echo "        processed          $(du -sh "$SRC/processed" | cut -f1)"
echo "        data_module_cache  $(du -sh "$SRC/data_module_cache" | cut -f1)"
echo "  dest: $IGB_USER@$IGB_HOST:$DEST_DIR"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "== DRY_RUN: nothing sent, no SSH opened =="
  exit 0
fi

# One multiplexed SSH connection for every rsync below, so the transfer authenticates once.
CTL="${TMPDIR:-/tmp}/igb-025-sync-%r@%h:%p"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$CTL" -o ControlPersist=4h)
trap 'ssh "${SSH_OPTS[@]}" -O exit "$IGB_USER@$IGB_HOST" 2>/dev/null || true' EXIT

# Destination tree plus the empty raw/lmdb stub PyG's _download looks for.
ssh "${SSH_OPTS[@]}" "$IGB_USER@$IGB_HOST" "mkdir -p '$DEST_DIR/raw/lmdb'"

# -a preserves times and permissions so the LMDB arrives byte-identical; -P resumes a
# partial file and shows progress; --info=progress2 gives one aggregate percentage over
# the whole transfer rather than per-file noise across a 517 GB LMDB.
for d in processed data_module_cache; do
  echo "== rsync $d =="
  rsync -aP --human-readable --info=progress2 \
    -e "ssh ${SSH_OPTS[*]}" \
    "$SRC/$d/" "$IGB_USER@$IGB_HOST:$DEST_DIR/$d/"
done

echo "== verifying: a second pass must report no transfers =="
rsync -a --itemize-changes --dry-run -e "ssh ${SSH_OPTS[*]}" \
  "$SRC/processed/" "$IGB_USER@$IGB_HOST:$DEST_DIR/processed/" | head -20

echo "== done =="
echo "On IGB, DATA_ROOT=$IGB_DATA_ROOT and the loader root resolves to:"
echo "  $DEST_DIR"
