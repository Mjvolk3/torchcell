#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave8_samp_k128_s2705.slurm
# [[experiments.027-betaxanthin-metabolic.wave8-sampling]]
#
# WAVE 8 (2026-09-12), THE DRAW COUNT PUSHED TO THE EXTREME. K = 128 at seed 2705.
#
# WHY 128 AND NOT MORE. K = 16 cost 1.11x per epoch on the GPU (job 1783 at 60.0 s per
# epoch against job 1776's ~54), because the flux layer is roughly 1 percent of a step that
# the transformer backbone dominates and the backbone never sees the extra rows. Fitting
# the CPU benchmark and that point gives a per-epoch cost near 1 + 0.0105 (0.7 K - 1),
# which puts K = 128 at about 1.9x, just inside the 2x per-epoch budget. The measurement
# decides: if this cell's epoch time exceeds 2x, the remaining seeds run at K = 64.
#
# THE PAIR IS THE K = 1 SAMPLED CELL AT THE SAME SEED, not the deterministic one: both arms
# carry --flux-latent-stochastic so the contrast is the draw count alone. Partners are jobs
# 1785 (s2703), 1793 (s2704) and gh_wave8_samp_k1_s2705.
#
# MEMORY. One [B*K, 4131] float32 flux tensor is 258 MiB at batch 128 and K = 128, and the
# alternating projection holds a handful at once, so the request is raised accordingly.
#
#   cd <worktree-root> && bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave8_samp_k128_s2705.slurm
#
#SBATCH -p main
#SBATCH --mem=100g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w8-samp-k128-s2705
#SBATCH --time=10:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB_DIR=/scratch/projects/torchcell-scratch/experiments/027-betaxanthin-metabolic/wandb
export WANDB_PROJECT=torchcell_027_bxfx
export NUM_WORKERS=2
export COLLAPSE_PATIENCE_EPOCHS=200
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave8
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE8_EPOCHS:-80}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_projection --seed 2705 --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union \
    --constraint-warmup-epochs 20 --constraint-scale 0.0 --head-normalize \
    --flux-latent-stochastic --flux-draws 128 \
    --wandb-group gh_wave8_samp --tag _w8samp \
    --out wave8_samp_k128_s2705.json
