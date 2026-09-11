#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave6_proj_tdg_nocstr_s2701.slurm
# [[experiments.027-betaxanthin-metabolic.projection-head]]
#
# WAVE 6 (2026-09-11): flux_projection_anchored_tdg, constraint scale 0.0, seed 2701, union_external deltaG.
#
# Anchored thermodynamics on the union deltaG table, the enzyme-capacity bound through the
# predicted per-gene abundance head, the forward protein-pool budget and the Gibbs
# dissipation limit, on top of a flux that is balanced and in-box by construction.
#
# IS THE ZERO SIGNAL THE SEED OR THE HEAD? Job 1725, this arm at seed 2700, is the first
# cell to meet the Genesis criterion (0.98 of the top 100 balanced, in-bounds and downhill)
# and it scores -0.0006 on the pinned test genes against pooled's 0.0717 at the same seed.
# One seed cannot separate "this head carries no phenotype signal" from "this seed did not
# find it". These two cells are that separation.
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_wave6_proj_tdg_nocstr_s2701.slurm
#
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w6-proj-tdg-nocstr-s2701
#SBATCH --time=05:00:00
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
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave6
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE6_EPOCHS:-80}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_projection_anchored_tdg --seed 2701 --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union_external \
    --constraint-warmup-epochs 20 --constraint-scale 0.0 --head-normalize \
    --wandb-group gh_wave6_proj --tag _w6proj \
    --out wave6_proj_flux_projection_anchored_tdg_ue_nocstr_s2701.json
