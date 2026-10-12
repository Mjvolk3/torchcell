#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave8_samp_k128_e200_s2703.slurm
# [[experiments.027-betaxanthin-metabolic.wave8-sampling]]
#
# WAVE 8 (2026-09-12), THE DEFINITIVE K-DRAW CONTRAST AT 200 EPOCHS. K = 128, seed 2703.
#
# WHY THE 80-EPOCH CELLS DO NOT SETTLE IT. Every one of them selected its epoch at the very
# end of its budget, 74 to 77 of 80, with the training data loss still falling: measured
# 0.5397 to 0.5197 (job 1783), 0.5433 to 0.4761 (1785), 0.5319 to 0.4711 (1789) and 0.5315
# to 0.4773 (1793). A contrast between two arms that are both still improving compares
# where each happened to be when the clock stopped, not what either converges to. The
# 80-epoch pairs stand as the SHORT-BUDGET read and this round is the definitive one.
#
# THE PAIR IS K = 1 SAMPLED AT THE SAME SEED, both arms carrying --flux-latent-stochastic,
# so the contrast is the draw count and nothing else. The two arms of a seed run side by
# side on the two available cards.
#
# 200 EPOCHS, and the collapse guard is set to match so a slow escape is not cut off.
# `select_epoch` is reported per cell: if it again lands at the end of the budget, 200 was
# not enough either and the note says so rather than reporting a converged number.
#
#   cd <worktree-root> && bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave8_samp_k128_e200_s2703.slurm
#
#SBATCH -p main
#SBATCH --mem=100g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w8-samp-k128-e200-s2703
#SBATCH --time=12:00:00
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
EPOCHS="${WAVE8_E200_EPOCHS:-200}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_projection --seed 2703 --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union \
    --constraint-warmup-epochs 20 --constraint-scale 0.0 --head-normalize \
    --flux-latent-stochastic --flux-draws 128 \
    --wandb-group gh_wave8_samp_e200 --tag _w8samp \
    --out wave8_samp_k128_e200_s2703.json
