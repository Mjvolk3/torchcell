#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_optuna_expr.slurm
# GilaHyper EXPRESSION Optuna sweep (per_gene head, PEAK objective). 4 single-GPU workers over
# one shared SQLite study. GilaHyper = plain python (no singularity) + W&B ONLINE. 6h cap.
#
#   cd <repo-or-worktree-root> && sbatch experiments/019-simb-multimodal/scripts/gh_optuna_expr.slurm
#
#SBATCH -p main
#SBATCH --mem=120g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
#SBATCH --job-name=019-expr-peak
#SBATCH --time=24:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out

# Resolve the repo/worktree root from where sbatch was submitted (so worktree code is used).
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell}"
OPTUNA_DIR=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/optuna
mkdir -p "$OPTUNA_DIR"
mkdir -p /scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "$PROJECT_ROOT/experiments" || exit 1

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB__SERVICE_WAIT=600

export CONDITION=expr
export JOINT_BASE_CONFIG=gh_expr_optuna_000
export OPTUNA_STORAGE="sqlite:///$OPTUNA_DIR/optuna_019_expr_peak.db"
export OPTUNA_STUDY_NAME=expr_peak_000
export OPTUNA_N_TRIALS="${OPTUNA_N_TRIALS:-30}"
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 3))

SWEEP=$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py

# Pre-create the study (single- vs multi-objective directions from one place); workers load it.
python $SWEEP --create-only

CUDA_VISIBLE_DEVICES=0 OPTUNA_WORKER_ID=0 python $SWEEP &
CUDA_VISIBLE_DEVICES=1 OPTUNA_WORKER_ID=1 python $SWEEP &
CUDA_VISIBLE_DEVICES=2 OPTUNA_WORKER_ID=2 python $SWEEP &
wait
