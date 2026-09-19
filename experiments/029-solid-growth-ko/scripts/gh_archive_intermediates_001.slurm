#!/bin/bash
#SBATCH --job-name=029-archive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/025-fitness-joint-head/experiments/029-solid-growth-ko/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/025-fitness-joint-head/experiments/029-solid-growth-ko/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Archive the three regenerable stages of build 001 (raw, conversion, aggregation;
# 615 G + 614 G + 547 G on /db) to /bulk as tar.zst, then test each archive. The
# 025 precedent (/bulk/experiments/025-solid-growth-001-full-build-intermediates/)
# compressed the JSON stages about 170x in about 12 minutes each. Deleting the
# originals from /db is a separate, manual step after archive.log says ALL_DONE.
set -euo pipefail

SRC=/db/experiments/029-solid-growth-ko-001-ko-build
DST=/bulk/experiments/029-solid-growth-ko-001-ko-build-intermediates
mkdir -p "$DST"
echo "Running on $(hostname) at $(date)"
df -h /db /bulk | tail -2

for stage in raw conversion aggregation; do
  t0=$SECONDS
  tar -C "$SRC" -cf - "$stage" | zstd -T16 -q -o "$DST/$stage.tar.zst"
  zstd -t -q "$DST/$stage.tar.zst"
  echo "$stage done $((SECONDS - t0)) s $(du -h "$DST/$stage.tar.zst" | cut -f1)" | tee -a "$DST/archive.log"
done
echo ALL_DONE | tee -a "$DST/archive.log"
echo "finished at $(date)"
