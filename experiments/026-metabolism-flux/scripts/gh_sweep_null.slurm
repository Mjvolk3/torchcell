#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_sweep_null.slurm
# CALIBRATE THE NULL. Twelve seeds each of `pooled` and `flux_anchored` with the
# training targets permuted, validation left real. The maximum over epochs of each run
# is one draw from the null distribution of the statistic every arm is reported with.
#
# This is the decisive cell of the overnight sweep. Measured on the smoke run, the
# betaxanthin validation set is n = 353, so a Pearson correlation there has null width
# 1/sqrt(350) = 0.0535, and the arms reported so far span 0.0069 to 0.0867. If the
# permuted null routinely reaches 0.09, no arm has yet measured anything, and that has
# to be known before any of it is written up as a result.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_sweep_null.slurm
#
#SBATCH -p main
#SBATCH --mem=90g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=026-sweep-null
#SBATCH --time=12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/026-metabolism-flux/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/026-metabolism-flux/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/metabolism-flux-module}"
mkdir -p /scratch/projects/torchcell-scratch/experiments/026-metabolism-flux/slurm/output

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

# `load_dotenv()` searches upward from CWD, and a worktree's `.env` is gitignored and
# machine-specific, so resolve it explicitly rather than relying on the search.
ENV_FILE="$PROJECT_ROOT/.env"
[ -f "$ENV_FILE" ] || ENV_FILE="$HOME/Documents/projects/torchcell/.env"
set -a
. "$ENV_FILE"
set +a
# Pin the experiment root to THIS worktree. Results must land beside the scripts that
# made them, not in whichever checkout the shared `.env` happens to name.
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
    --grid null \
    --seeds 301,302,303,304,305,306,307,308,309,310,311,312 \
    --epochs 30 \
    --max-hours 11 \
    --out sweep_null.json
