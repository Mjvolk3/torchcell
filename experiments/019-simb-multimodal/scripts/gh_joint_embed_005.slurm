#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_joint_embed_005.slurm
# GilaHyper JOINT (expr_morph) arm of the _005 NODE-EMBEDDING (cold-start) study.
#
# MULTI-OBJECTIVE (Pareto over expression x morphology), sweeping `node_embeddings` and
# `learnable_embedding` alongside the carried-over decoder/dist axes. Runs on the SAME
# both-modality instance set as the morph and expr arms (`require_modalities`, inherited via
# cgt_embed_005 -> cgt_decoder_004 -> cgt_decoder_003), so joint - single stays a clean
# auxiliary-task contrast rather than a data-quantity artifact.
#
# GilaHyper = plain python (no singularity) + W&B ONLINE. Uses 3 of 4 GPUs: GPU 3 is
# deliberately left free for ad-hoc work. Self-resubmits until the deadline.
#
#   cd <worktree-root> && sbatch experiments/019-simb-multimodal/scripts/gh_joint_embed_005.slurm
#
#SBATCH -p main
#SBATCH --mem=120g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
#SBATCH --job-name=019-joint-embed005
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
REQUEUE_SELF="$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/gh_joint_embed_005.slurm"
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
export JOINT_BASE_CONFIG=cgt_embed_005
export OPTUNA_STORAGE="sqlite:///$OPTUNA_DIR/optuna_019_expr_morph_005.db"
export OPTUNA_STUDY_NAME=expr_morph_005
export OPTUNA_N_TRIALS="${OPTUNA_N_TRIALS:-30}"
export WANDB_PROJECT_SUFFIX=v5
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 3))

SWEEP=$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py

# Pre-create the study (one process, sets multi-objective directions); workers load it.
python $SWEEP --create-only

# 3 workers only: GPU 3 is deliberately LEFT FREE on GilaHyper for ad-hoc work
# (e.g. computing any embedding that still needs building) while the sweep runs.
CUDA_VISIBLE_DEVICES=0 OPTUNA_WORKER_ID=0 python $SWEEP &
CUDA_VISIBLE_DEVICES=1 OPTUNA_WORKER_ID=1 python $SWEEP &
CUDA_VISIBLE_DEVICES=2 OPTUNA_WORKER_ID=2 python $SWEEP &
wait

# Clean exit before the deadline (workers finished their trial budget): keep going.
requeue_if_before_deadline "$REQUEUE_SELF"
