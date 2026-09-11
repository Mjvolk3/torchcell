#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave5_g6_flux_capacity_headfix.slurm
# [[experiments.027-betaxanthin-metabolic]]
#
# WAVE 5 (2026-09-10), cell G7: G6 with the constraint weight held at EXACTLY ZERO.
#
# THE DISCRIMINATING CELL. G6 (job 1669) has the head fix and is flat: data loss 0.5295 from
# epoch 10 to 28 with a validation prediction spread of 0.0000, where the pooled baseline
# (job 1634) had escaped at epoch 23 and read 0.120 at epoch 28. Two hypotheses remain and
# this separates them. If the pure data term through the flux readout escapes, the constraint
# terms were the blocker. If it stays flat with NO physics at all, the flux readout path
# itself cannot escape and the answer is architectural, not a matter of weighting.
#
# Identical to G6 in every other respect, so the contrast is the constraint scale alone.
#
# WHY, measured rather than guessed. diagnose_head_scale.py instantiated both heads on CPU on
# one real training batch at initialization and read their raw output against a target
# standardized to mean 0 and sd 1: the pooled head starts at mean -0.573 with sd 0.0074, the
# flux head at MEAN +7.030 with sd 0.237. The offset, not the spread, is what the CRPS pays,
# and it comes from the dense term reading 4,131 fluxes whose mean absolute value is 9.87 and
# whose maximum is 61.1 against random weights. That matches the wave-5 cells' epoch-0 data
# loss of 8.5 to 14.3 at a constraint weight of exactly zero (jobs 1663 to 1665), where the
# pooled arm on the same recipe runs 0.49 (job 1634). --head-normalize divides both of the
# head's inputs by the layer's own working flux magnitude, which is a change of units and not
# of the head: same weights, same terms.
#
# Otherwise identical to G1: flux_capacity, predicted kcat tier, union deltaG, warmup 20
# epochs to a final constraint scale of 0.1, CRPS head, graph regularizer, seed 2700.
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_wave5_g6_flux_capacity_headfix.slurm
#
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w5-flux_capacity-nocstr-s2701
#SBATCH --time=04:00:00
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
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave5
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE5_EPOCHS:-80}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_capacity --seed 2701 --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union \
    --constraint-warmup-epochs 20 --constraint-scale 0.0 --head-normalize \
    --wandb-group gh_wave5_000 --tag _w5 \
    --out wave5_g7_flux_capacity_nocstr_s2701.json
