#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_sweep_arms_a.slurm
# ARM COMPARISON, first half: all five registered arms at seeds 101-105, 25 cells.
# Its partner `gh_sweep_arms_b.slurm` runs seeds 106-110 on a second GPU.
#
# WHY. The banked comparison is three seeds per arm, and the arms are separated by 0.004
# to 0.08 with per-arm standard deviations of 0.007 to 0.083. Nothing in it is resolved.
# The betaxanthin validation set is n = 353, so a single Pearson there has null width
# 0.0535, and ten more seeds per arm brings each to thirteen replicates, a standard error
# near 0.009. That is enough to resolve a real 0.02 separation, and enough to say plainly
# that a smaller one is not resolvable rather than reporting it as an ordering.
#
# The two jobs use DISJOINT seed sets on purpose: the data module caches its split index
# under `index_seed_<n>.json`, so two concurrent jobs sharing a seed would race on the
# same cache file.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_sweep_arms_a.slurm
#
#SBATCH -p main
#SBATCH --mem=90g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=026-sweep-arms-a
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
    --seeds 101,102,103,104,105 \
    --epochs 30 \
    --max-hours 11 \
    --out sweep_arms_a.json
