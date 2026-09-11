#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave6_aa_allon_s2700.slurm
# [[experiments.027-betaxanthin-metabolic.wave6]]
#
# WAVE 6 amino-acid cell, allon, seed 2700.
#
# THE PHENOTYPE MOVES, THE PROTOCOL DOES NOT. --head mulleder19 makes the 19-column
# Mulleder amino-acid profile the only supervised head: the epoch is selected on the MEAN
# per-feature validation Pearson over the 19 columns and the test metrics are per-feature
# Spearman and Pearson on the pinned Merzbacher test genes, with the per-feature n reported
# beside them because the columns are not measured on the same strains.
#
# THE CONSTRAINED READ. Same flux arm as the nocstr cell but with the physics ramped in: constraint scale 0.1 over a 20-epoch linear warmup, which is the wave-5 G1 setting. Wave 5 measured that the constraint terms hold the BETAXANTHIN head on the mean plateau; whether the amino-acid head behaves the same way has not been measured, and this cell is what measures it.
#
# --head-normalize means something DIFFERENT on this head, and the difference is recorded
# here rather than left to be inferred. On FluxScalarHead the flag divides the dense w^T v
# input by the layer's working flux magnitude. FluxMetaboliteHead has no dense term and no
# such switch; it emits a_k log(1 + omega_k) + b_k with a_k initialized to 1, which on this
# dataset starts the flux arm at an epoch-0 data loss of 2.4478 (smoke, seed 9001, same
# recipe) against a target standardized to mean 0 and sd 1. train_bx.py therefore divides
# the INITIAL slope by log1p(flux_scale), which is the identical forward pass at
# initialization; the slope is learnable and free to move from there. The pooled cell takes
# no such flag: its head already starts near the target mean.
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_wave6_aa_allon_s2700.slurm
#
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w6-aa-allon-s2700
#SBATCH --time=05:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
# THE FROZEN PHYSICS SNAPSHOT. `torchcell/metabolism/flux_layer.py` and 026's
# `train_flux.py` are being edited by another agent in this same worktree tonight, and a
# queued cell imports them at the moment it STARTS, not at the moment it is submitted. A
# smoke run at 23:03 on 2026-09-10 caught a mid-write flux_layer.py and died on
# `AttributeError: FluxLayer object has no attribute latent_stochastic`. Worse than the
# crash would be the silent version: seeds pooled into one contrast with half of them run
# on a different flux layer than the other half. So every wave-6 cell reads ONE hash-pinned
# copy taken at 23:05 on 2026-09-10, and train_bx.py records its sha256 in the result file.
export TC_FROZEN=/scratch/projects/torchcell-scratch/frozen-027-wave6
export PYTHONPATH="$TC_FROZEN:$PROJECT_ROOT:$PYTHONPATH"
export TC_FLUX_SCRIPTS_DIR="$TC_FROZEN/026-scripts"
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
EPOCHS="${WAVE5_EPOCHS:-80}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_capacity --kcat-tier predicted --thermo-source union --constraint-warmup-epochs 20 --constraint-scale 0.1 --seed 2700 --epochs "$EPOCHS" --crps --graph-reg \
    --head mulleder19 --head-normalize \
    --wandb-group gh_wave6_aa --tag _w6aa \
    --out wave6_aa_allon_s2700.json
