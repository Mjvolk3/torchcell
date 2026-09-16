#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave6_proj_anchored_s2700.slurm
# [[experiments.027-betaxanthin-metabolic.projection-head]]
#
# WAVE 6 (2026-09-10), cell P3: EVERYTHING ON, at a constraint scale of 0.1.
#
# Anchored thermodynamics on the union deltaG table, the enzyme-capacity bound through the
# predicted per-gene abundance head, the forward protein-pool budget and the Gibbs
# dissipation limit, on top of a flux that is balanced and in-box by construction.
#
# WHAT THIS SEPARATES. Under the BOX parameterization at the same scale, job 1678 escaped
# the mean plateau at epoch 41, reached val Pearson 0.199, and then collapsed to zero
# prediction spread with the weighted constraint term at 1.3 percent of the data loss. The
# wave-5 flux_off cell (job 1677, no thermodynamics, no capacity, no budget) never escaped
# at all, which places the blocking terms on mass balance and the box rather than on the
# physics. Both of those now contribute nothing to the gradient, because the
# parameterization satisfies them. If this cell escapes and STAYS escaped, the collapse was
# the stoichiometric penalties; if it collapses anyway, it is the physics terms and the
# bracket at scale 0.03 and 0.3 is the next read.
#
# Read per epoch: val_pred_sd_ratio against the escape rule in results/wave5_summary.json
# (three consecutive epochs above 0.05), and every feas_* scalar. Compare test Spearman on
# the pinned genes to pooled 0.0717 (job 1674) and the box head at scale 0 0.0747 (job 1673),
# both at seed 2700.
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_wave6_proj_anchored_s2700.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w6-proj-anchored-s2700
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
    --arm flux_projection_anchored --seed 2700 --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union \
    --constraint-warmup-epochs 20 --constraint-scale 0.1 --head-normalize \
    --wandb-group gh_wave6_proj --tag _w6proj \
    --out wave6_proj_flux_projection_anchored_s2700.json
