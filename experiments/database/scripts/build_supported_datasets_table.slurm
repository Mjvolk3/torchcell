#!/bin/bash
#SBATCH --job-name=yeast-signal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --error=/scratch/projects/torchcell/database/slurm/output/%j_%x.out
#SBATCH --no-requeue

# experiments/database/scripts/build_supported_datasets_table.slurm
# [[experiments.database.scripts.build_supported_datasets_table]]
#
# The supported-datasets signal scan under slurm: every dev-tree LMDB the registry
# names is read READ-ONLY (lmdb readonly, lock=False) and gzip-streamed in a process
# pool, largest store first, with each result persisted to dataset_signals_cache.json as
# it lands, so a killed run resumes. Nothing under $DATA_ROOT is written.
#
#   sbatch --export=ALL,TORCHCELL_SRC=<worktree> \
#       experiments/database/scripts/build_supported_datasets_table.slurm
#
# TORCHCELL_SRC is the checkout to run (its PYTHONPATH and its registry); WORKERS
# defaults to the allocated CPUs. The JSON lands under that checkout's
# experiments/database/results/pre-build/<date>/supported_datasets.json.
set -euo pipefail
SRC="${TORCHCELL_SRC:?set TORCHCELL_SRC to the checkout to run}"
cd "$SRC"
export PYTHONPATH="$SRC"
PY="$HOME/miniconda3/envs/torchcell/bin/python"
echo "== $(date) host $(hostname) src $SRC commit $(git rev-parse --short HEAD) workers ${WORKERS:-$SLURM_CPUS_PER_TASK}"
/usr/bin/time -v "$PY" experiments/database/scripts/build_supported_datasets_table.py \
    --workers "${WORKERS:-$SLURM_CPUS_PER_TASK}"
echo "== $(date) done"
