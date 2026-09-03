#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_sweep_arms_b.slurm
# ARM COMPARISON, second half: all five registered arms at seeds 106-110, 25 cells.
# Reasoning and design are in its partner `gh_sweep_arms_a.slurm`, which runs seeds
# 101-105. The seed sets are disjoint so the two jobs never race on the same cached
# split index.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_sweep_arms_b.slurm
#
#SBATCH -p main
#SBATCH --mem=90g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=026-sweep-arms-b
#SBATCH --time=12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/026-metabolism-flux/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/026-metabolism-flux/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/metabolism-flux-module}"
mkdir -p /scratch/projects/torchcell-scratch/experiments/026-metabolism-flux/slurm/output

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

ENV_FILE="$PROJECT_ROOT/.env"
[ -f "$ENV_FILE" ] || ENV_FILE="$HOME/Documents/projects/torchcell/.env"
set -a
. "$ENV_FILE"
set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
echo "env $ENV_FILE; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
# W&B run directories go to scratch, never into the worktree: four concurrent
# jobs write one run directory per cell and they would otherwise accumulate in
# the repo tree as untracked clutter.
export WANDB_DIR=/scratch/projects/torchcell-scratch/experiments/026-metabolism-flux/wandb
mkdir -p "$WANDB_DIR"

python "$PROJECT_ROOT/experiments/026-metabolism-flux/scripts/sweep_flux.py" \
    --grid arms \
    --seeds 106,107,108,109,110 \
    --epochs 30 \
    --max-hours 11 \
    --out sweep_arms_b.json
