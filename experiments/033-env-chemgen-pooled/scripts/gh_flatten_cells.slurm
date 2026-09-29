#!/bin/bash
#SBATCH --job-name=033-flatten
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Flatten the processed store of build 001 (6,042,771 entries, 106 G of JSON) into one
# parquet row per (genotype, environment) cell. Read-only on /db; writes the table to
# $DATA_ROOT/experiments/033-env-chemgen-pooled/cell_table/.
#
# Sizing: the smoke run (every 3,000th entry, 2,015 entries) took 7.6 CPU seconds
# including two interpreter starts, so the full pass is about 3 CPU hours and about
# 15 minutes on 16 workers. The one-hour limit is deliberately honest so the job can
# backfill beside the GPU packs.
#
#   sbatch experiments/033-env-chemgen-pooled/scripts/gh_flatten_cells.slurm
set -euo pipefail

echo "Running on $(hostname) at $(date -Is)"
PROJECT_ROOT=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git rev-parse --short HEAD))"
export PYTHONPATH="$PROJECT_ROOT"

/home/michaelvolk/miniconda3/envs/torchcell/bin/python \
  experiments/033-env-chemgen-pooled/scripts/flatten_cells.py --workers 16

echo "finished at $(date -Is)"
