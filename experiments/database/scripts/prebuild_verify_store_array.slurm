#!/bin/bash
#SBATCH --job-name=prebuild-verify-store
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/projects/torchcell-scratch/dev/prebuild-verify/slurm-output/%A_%a_%x.out
#SBATCH --error=/scratch/projects/torchcell-scratch/dev/prebuild-verify/slurm-output/%A_%a_%x.out
#SBATCH --no-requeue
# experiments/database/scripts/prebuild_verify_store_array.slurm
# [[experiments.database.scripts.prebuild_verify_store_array]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/prebuild_verify_store_array.slurm
#
# Lane P of the ontology unification (Path U, 2026-10-15): build a REPRESENTATIVE
# sample of dev stores from the integrated `ontology-unify` branch under a PRIVATE
# DATA_ROOT, one array task per dataset, so the records can be proved right before the
# owner commits to the 170-store sweep. Same shape as
# database/slurm/scripts/gilahyper_build_dataset_lmdbs_array.slurm (`--retire-existing
# --verify`, 16 CPU / 64 G / 8 h per task, overestimated on purpose) but it never
# touches the shared dev tree: `VERIFY_DATA_ROOT` is a tree whose `data/<aux>` and
# `torchcell-*` entries are symlinks to the shared inputs and whose
# `data/torchcell/<slug>/raw` is a symlink to the shared raw mirror, so only
# `processed/` and `preprocess/` are written, and only there.
#
#   CLASSES=$VERIFY_DATA_ROOT/classes.txt      # one registered dataset class per line
#   sbatch --array=1-$(wc -l < "$CLASSES")%2 \
#       --export=ALL,CLASSES_FILE="$CLASSES",TORCHCELL_SRC=<worktree>,VERIFY_DATA_ROOT=<tree> \
#       experiments/database/scripts/prebuild_verify_store_array.slurm
#
# Wall time and peak RSS come from `/usr/bin/time -v` on the python process (the slurm
# cgroup also accounts them: `sacct -j <id> --format=JobID,Elapsed,MaxRSS -P`).
set -euo pipefail
echo "Running on $(hostname) at $(date); array task ${SLURM_ARRAY_TASK_ID:?not an array job}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
: "${CLASSES_FILE:?set CLASSES_FILE=<one dataset class name per line>}"
: "${TORCHCELL_SRC:?set TORCHCELL_SRC=<the checkout whose code builds the store>}"
: "${VERIFY_DATA_ROOT:?set VERIFY_DATA_ROOT=<the private data root>}"
cd "$TORCHCELL_SRC"
DATASET_CLASS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CLASSES_FILE")
[ -n "$DATASET_CLASS" ] || { echo "no class on line $SLURM_ARRAY_TASK_ID of $CLASSES_FILE"; exit 1; }
echo "source: $TORCHCELL_SRC ($(git rev-parse --short HEAD 2>/dev/null || echo not-a-git-repo))"
echo "data root: $VERIFY_DATA_ROOT"
echo "dataset: $DATASET_CLASS (line $SLURM_ARRAY_TASK_ID of $CLASSES_FILE)"
export DATA_ROOT="$VERIFY_DATA_ROOT"
/usr/bin/time -v env PYTHONPATH="$TORCHCELL_SRC" python -m torchcell.database.build_dataset_lmdb \
    --dataset "$DATASET_CLASS" --data-root "$VERIFY_DATA_ROOT" \
    --io-workers "$SLURM_CPUS_PER_TASK" --retire-existing --verify
echo "DONE at $(date)"
