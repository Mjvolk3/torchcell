#!/bin/bash
# experiments/030-solid-growth-multi/scripts/stage_pool_store_shm.sh
# [[experiments.030-solid-growth-multi.scripts.stage_pool_store_shm]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/stage_pool_store_shm.sh
#
# Sourced by the IGB slurm scripts. Copies a pool store (make_pool_store_030.py) from GPFS
# onto the compute node's RAM disk and removes it when the job ends, so the loader reads
# the training pool from memory instead of faulting pages of the 955 GB build through
# GPFS's mmap lock. /dev/shm on compute-5-7 is a 252 GB tmpfs; the S3Q store is a few GB.
# The copy is charged to the job's memory cgroup, so size --mem for RSS plus the store.
#
#   source experiments/030-solid-growth-multi/scripts/stage_pool_store_shm.sh
#   stage_pool_store "$IGB_DATA_ROOT/$POOL_STORE_REL" "$SHM_ROOT"   # sets up the EXIT trap
#
# Nothing else cleans /dev/shm on these nodes (job_container/none; files from July 2025
# are still there), so the trap is the only thing standing between a staged copy and a
# permanent, invisible 5 GB of node RAM. Do not disable it.
stage_pool_store() {
  local src="$1" dst="$2"
  [[ -f "$src/processed/lmdb/STORE.json" ]] || { echo "ERROR: $src is not a pool store (no processed/lmdb/STORE.json)" >&2; return 1; }
  [[ -d "$src/data_module_cache" ]] || { echo "ERROR: $src has no data_module_cache" >&2; return 1; }
  echo "staging pool store: $src -> $dst"
  df -h "$(dirname "$(dirname "$dst")")" | tail -1
  mkdir -p "$dst"
  # shellcheck disable=SC2064
  trap "echo 'removing staged pool store $dst'; rm -rf '$dst'" EXIT
  # Slurm ends a timed-out or cancelled job with SIGTERM; bash runs no EXIT trap when a
  # signal kills it, so TERM is turned into an exit (which runs the trap above).
  trap 'exit 143' TERM INT
  local t0; t0=$(date +%s)
  rsync -a "$src/" "$dst/"
  echo "staged in $(( $(date +%s) - t0 )) s: $(du -sh "$dst" | cut -f1)"
  cat "$dst/processed/lmdb/STORE.json"
}
