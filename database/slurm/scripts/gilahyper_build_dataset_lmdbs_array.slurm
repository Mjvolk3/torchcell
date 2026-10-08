#!/bin/bash
#SBATCH --job-name=build_dataset_lmdbs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%A_%a_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%A_%a_%x.out
#SBATCH --no-requeue
# Rebuild MANY datasets' LMDBs in the DEV tree ($DATA_ROOT/data/torchcell/<slug>), one
# array task per dataset, so a full knowledge-graph rebuild finds every mapped store
# fresh. One sbatch call covers the whole list:
#
#   CLASSES=/scratch/projects/torchcell/database/slurm/stale_classes.txt
#   PYTHONPATH=$TORCHCELL_SRC python -m torchcell.database.build_dataset_lmdb \
#       --list-stale --include-private > "$CLASSES"
#   sbatch --array=1-$(wc -l < "$CLASSES")%4 --export=ALL,CLASSES_FILE="$CLASSES" \
#       database/slurm/scripts/gilahyper_build_dataset_lmdbs_array.slurm
#
# Task N builds line N of CLASSES_FILE (one registered dataset class name per line).
# The `%4` caps how many tasks run at once, which is what keeps the node's other jobs
# fed; raise it when the node is idle. Each task retires the dataset's existing
# processed/ and preprocess/ as *.superseded.<timestamp> siblings (a rename, never a
# delete) and then builds, so a stale store is never silently reused and nothing is
# lost. Sizing: the same 16 CPUs / 64 G the single-dataset script used for the two
# Costanzo 2016 interaction stores (4 h 41 min and 4 h 20 min on 2026-10-05); every
# other store took minutes.
#
# TORCHCELL_SRC (optional) points at the checkout whose code should build the datasets
# (the tagged release commit the live rebuild will pair with); DATA_ROOT comes from that
# checkout's .env (the dev tree, torchcell-scratch), never the build tree.
set -euo pipefail
echo "Running on $(hostname) at $(date); array task ${SLURM_ARRAY_TASK_ID:?not an array job}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
: "${CLASSES_FILE:?set CLASSES_FILE=<one dataset class name per line>}"
TORCHCELL_SRC="${TORCHCELL_SRC:-/home/michaelvolk/Documents/projects/torchcell}"
cd "$TORCHCELL_SRC"
DATASET_CLASS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CLASSES_FILE")
[ -n "$DATASET_CLASS" ] || { echo "no class on line $SLURM_ARRAY_TASK_ID of $CLASSES_FILE"; exit 1; }
echo "source: $TORCHCELL_SRC ($(git rev-parse --short HEAD 2>/dev/null || echo not-a-git-repo))"
echo "dataset: $DATASET_CLASS (line $SLURM_ARRAY_TASK_ID of $CLASSES_FILE)"
PYTHONPATH="$TORCHCELL_SRC" python -m torchcell.database.build_dataset_lmdb \
    --dataset "$DATASET_CLASS" --io-workers "$SLURM_CPUS_PER_TASK" --retire-existing
echo "DONE at $(date)"
