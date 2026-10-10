#!/bin/bash
#SBATCH --job-name=035-ladder
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/038-env-chemgen-vanacloig-cgt-corrected/experiments/038-env-chemgen-vanacloig-cgt-corrected/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/038-env-chemgen-vanacloig-cgt-corrected/experiments/038-env-chemgen-vanacloig-cgt-corrected/slurm/output/%j_%x.out
#SBATCH --no-requeue

# The compound-similarity ladder of experiment 035 on CPU.
#
#   sbatch --export=ALL,ARGS="--fold-seeds 0 1 2" \
#     experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/gh_baseline_ladder.slurm
#
# ARGS  further arguments to baseline_ladder.py (no commas: sbatch splits on them)
set -euo pipefail
ARGS=${ARGS:-}
PROJECT_ROOT=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/038-env-chemgen-vanacloig-cgt-corrected
PYTHON=/home/michaelvolk/miniconda3/envs/torchcell/bin/python
cd "$PROJECT_ROOT"
echo "Running on $(hostname) at $(date -Is), $(git rev-parse --short HEAD), ARGS=$ARGS"
export PYTHONPATH="$PROJECT_ROOT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# shellcheck disable=SC2086
$PYTHON experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/baseline_ladder.py \
  --cell-table /scratch/projects/torchcell-scratch/experiments/033-env-chemgen-pooled/cell_table_002/cell_table.parquet \
  --workers 8 $ARGS
echo "finished at $(date -Is)"
