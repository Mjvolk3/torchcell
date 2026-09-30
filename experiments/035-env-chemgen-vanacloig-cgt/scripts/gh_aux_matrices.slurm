#!/bin/bash
#SBATCH --job-name=035-aux
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/035-env-chemgen-vanacloig-cgt/experiments/035-env-chemgen-vanacloig-cgt/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/035-env-chemgen-vanacloig-cgt/experiments/035-env-chemgen-vanacloig-cgt/slurm/output/%j_%x.out
#SBATCH --no-requeue
# The other chemogenomic stores as oriented gene x compound matrices (aux_matrices.py).
set -euo pipefail
PROJECT_ROOT=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/035-env-chemgen-vanacloig-cgt
cd "$PROJECT_ROOT"
echo "Running on $(hostname) at $(date -Is), $(git rev-parse --short HEAD)"
export PYTHONPATH="$PROJECT_ROOT"
/home/michaelvolk/miniconda3/envs/torchcell/bin/python experiments/035-env-chemgen-vanacloig-cgt/scripts/aux_matrices.py \
  --cell-table /scratch/projects/torchcell-scratch/experiments/033-env-chemgen-pooled/cell_table/cell_table.parquet
echo "finished at $(date -Is)"
