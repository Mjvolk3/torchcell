#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_sweep_reg.slurm
# REGULARIZATION GRID on `flux_anchored`: learning rate x weight decay x hidden width,
# two seeds, 24 cells.
#
# WHY. Across the 18 banked 20-epoch runs, validation Pearson peaks at a median epoch of
# 4.5 and the mean of the last five epochs is at or below zero in 13 of them. That is an
# overfitting statement, not an undertraining one, so the lever is regularization rather
# than budget. The incumbent (lr 1e-3, wd 1e-4, hidden 32) is the corner of this grid;
# the grid moves down in learning rate, up in decay, and brackets width both ways. It is
# the only cell of the sweep that could raise the ceiling rather than measure it.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_sweep_reg.slurm
#
#SBATCH -p main
#SBATCH --mem=90g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=026-sweep-reg
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
    --grid reg \
    --arm flux_anchored \
    --seeds 201,202 \
    --epochs 30 \
    --max-hours 11 \
    --out sweep_reg_flux_anchored.json
