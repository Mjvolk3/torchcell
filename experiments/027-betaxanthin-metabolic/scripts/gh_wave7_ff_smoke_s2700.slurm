#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave7_ff_smoke_s2700.slurm
# [[experiments.027-betaxanthin-metabolic.wave7-flux-force-head]]
#
# WAVE 7 SMOKE (2026-09-11): two epochs of the flux-force channel on the real cell, so the
# dataset, the head, the feasibility block and the results writer are all exercised before a
# five-hour cell is spent on them. Identical to the full cell in every other respect.
#
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave7_ff_smoke_s2700.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w7-ff-smoke-s2700
#SBATCH --time=01:00:00
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
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave7
# The channel. 026's ARMS registry stays the source of truth for the physics switches; the
# parameterization string is passed through by train_bx.py and recorded by the layer in
# flux_meta.coverage.parameterization_raw.
export TC_FLUX_PARAMETERIZATION=fluxforce_k20_box
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; channel=$TC_FLUX_PARAMETERIZATION"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_projection_anchored_fluxforce --seed 2700 --epochs 2 --crps --graph-reg \
    --kcat-tier predicted --thermo-source union_external \
    --constraint-warmup-epochs 20 --constraint-scale 0.0 --head-normalize \
    --wandb-group gh_wave7_ff_smoke --tag _w7ffsmoke --no-checkpoint \
    --out wave7_ff_smoke_fluxforce_k20_box_ue_nocstr_s2700.json
