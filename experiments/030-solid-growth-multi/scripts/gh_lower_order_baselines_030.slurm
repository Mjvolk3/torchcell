#!/bin/bash
# experiments/030-solid-growth-multi/scripts/gh_lower_order_baselines_030.slurm
# [[experiments.030-solid-growth-multi.scripts.gh_lower_order_baselines_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/gh_lower_order_baselines_030.slurm
#
# The lower-order baselines of the 030 build on GilaHyper, CPU only
# (lower_order_baselines_030.py): the term table over 376,732 triples from the 4.86M-row
# closure entry cache, then the ladder on both arms (ridge by direct solve, boosted trees
# over 300k rows, 2,000-resample paired bootstraps). The entry cache and its LabelEntry
# pools are the memory peak; 64 GB is sized from closure_recompute_030's analyze stage on
# the same cache. Boosted trees use every core given.
#
#   mkdir -p experiments/030-solid-growth-multi/slurm/output
#   sbatch experiments/030-solid-growth-multi/scripts/gh_lower_order_baselines_030.slurm
#   STAGE=fit (default all) and WANDB=0 (default 1) can be overridden with --export.
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=030-lower-order-baselines
#SBATCH --time=08:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/experiments/030-solid-growth-multi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/experiments/030-solid-growth-multi/slurm/output/%x_%j.out

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
STAGE="${STAGE:-all}"
WANDB="${WANDB:-1}"
echo "job=$SLURM_JOB_ID name=$SLURM_JOB_NAME host=$(hostname) started=$(date)"
echo "project=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null)) stage=$STAGE wandb=$WANDB"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; source "$PROJECT_ROOT/.env"; set +a
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$PROJECT_ROOT" || exit 1
WANDB_FLAG=""
[[ "$WANDB" == "1" ]] && WANDB_FLAG="--wandb"
python experiments/030-solid-growth-multi/scripts/lower_order_baselines_030.py \
  --stage "$STAGE" $WANDB_FLAG
STATUS=$?
ls -la experiments/030-solid-growth-multi/results/lower_order_baselines_030* \
  experiments/030-solid-growth-multi/results/t11-lower-order-baselines-030.tex 2>/dev/null
echo "finished=$(date) exit=$STATUS"
exit $STATUS
