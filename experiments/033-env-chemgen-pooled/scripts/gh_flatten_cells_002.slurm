#!/bin/bash
#SBATCH --job-name=033-flatten-002
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Flatten build 002 (the 2026.10.06 store) into its own cell table,
# $DATA_ROOT/experiments/033-env-chemgen-pooled/cell_table_002/cell_table.parquet, beside
# the build 001 table that every 031 and 035 result reads. Only the flatten step: the
# plan check and the figures of gh_flatten_cells.slurm write the build 001 record under
# results/ and are not rerun here.
#
#   sbatch experiments/033-env-chemgen-pooled/scripts/gh_flatten_cells_002.slurm
set -euo pipefail
echo "Running on $(hostname) at $(date -Is)"
PROJECT_ROOT=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled
PYTHON=/home/michaelvolk/miniconda3/envs/torchcell/bin/python
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git rev-parse --short HEAD))"
export PYTHONPATH="$PROJECT_ROOT"
export BUILD_NAME=002-pooled-build
$PYTHON experiments/033-env-chemgen-pooled/scripts/flatten_cells.py --workers 16
echo "finished at $(date -Is)"
