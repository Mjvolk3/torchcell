#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_joint_decoder_003.slurm
# GilaHyper JOINT (expr_morph) decoder x distributional sweep (_003).
#
# MULTI-OBJECTIVE (Pareto over expression x morphology). Two questions:
#   1. does standardize-both restore joint-expression to its ~0.11 single-task level
#      (i.e. kill the negative transfer seen in the _002 round)?
#   2. does s3_xattn lift morphology WITHIN the joint model, as it should if the shared
#      pooled readout — not the shared embedding — is what limits morphology?
# Runs on the SAME both-modality instance set as the morph/expr arms (require_modalities in
# cgt_decoder_003), so joint - single is a clean auxiliary-task contrast, not a data-quantity
# artifact. `quantile` is dropped for this arm (the driver restricts it): a K=19 readout over
# BOTH heads is a large projection on a shared 4-GPU node.
#
# GilaHyper = plain python (no singularity) + W&B ONLINE. Self-resubmits until the deadline.
#
#   cd <worktree-root> && sbatch experiments/019-simb-multimodal/scripts/gh_joint_decoder_003.slurm
#
#SBATCH -p main
#SBATCH --mem=120g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=019-joint-dec003
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out

# Resolve the repo/worktree root from where sbatch was submitted (so worktree code is used).
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell}"
OPTUNA_DIR=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/optuna
mkdir -p "$OPTUNA_DIR"
mkdir -p /scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output

# Run until this wall-clock deadline, resubmitting across walltime boundaries.
export DEADLINE="2026-07-29T10:00:00"
source "$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/requeue_until.sh"
REQUEUE_SELF="$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/gh_joint_decoder_003.slurm"
requeue_arm_trap "$REQUEUE_SELF"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "$PROJECT_ROOT/experiments" || exit 1

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB__SERVICE_WAIT=600

export CONDITION=expr_morph
export JOINT_BASE_CONFIG=cgt_decoder_004
export OPTUNA_STORAGE="sqlite:///$OPTUNA_DIR/optuna_019_expr_morph_004.db"
export OPTUNA_STUDY_NAME=expr_morph_004
export OPTUNA_N_TRIALS="${OPTUNA_N_TRIALS:-30}"
export WANDB_PROJECT_SUFFIX=v4
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 4))

SWEEP=$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py

# Pre-create the study (one process, sets multi-objective directions); workers load it.
python $SWEEP --create-only

CUDA_VISIBLE_DEVICES=0 OPTUNA_WORKER_ID=0 python $SWEEP &
CUDA_VISIBLE_DEVICES=1 OPTUNA_WORKER_ID=1 python $SWEEP &
CUDA_VISIBLE_DEVICES=2 OPTUNA_WORKER_ID=2 python $SWEEP &
CUDA_VISIBLE_DEVICES=3 OPTUNA_WORKER_ID=3 python $SWEEP &
wait

# Clean exit before the deadline (workers finished their trial budget): keep going.
requeue_if_before_deadline "$REQUEUE_SELF"
