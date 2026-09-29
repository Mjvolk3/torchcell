#!/bin/bash
#SBATCH --job-name=035-diagnose
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/035-env-chemgen-vanacloig-cgt/experiments/035-env-chemgen-vanacloig-cgt/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/035-env-chemgen-vanacloig-cgt/experiments/035-env-chemgen-vanacloig-cgt/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Token and strain-summary spread for every saved round-1 checkpoint, on CPU.
#   sbatch experiments/035-env-chemgen-vanacloig-cgt/scripts/gh_diagnose_token_collapse.slurm
set -euo pipefail
PROJECT_ROOT=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/035-env-chemgen-vanacloig-cgt
cd "$PROJECT_ROOT"
echo "Running on $(hostname) at $(date -Is) ($(git rev-parse --short HEAD))"
export PYTHONPATH="$PROJECT_ROOT"
/home/michaelvolk/miniconda3/envs/torchcell/bin/python \
  experiments/035-env-chemgen-vanacloig-cgt/scripts/diagnose_token_collapse.py
echo "finished at $(date -Is)"
