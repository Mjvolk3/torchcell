#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave6_parity_nocstr_s2705.slurm
# [[experiments.027-betaxanthin-metabolic.wave6]]
#
# WAVE 6 parity replicate, seed 2705: flux_capacity, constraints off, head fix.
#
# THE ARM. Identical recipe to wave-5 G7 (jobs 1673 seed 2700, 1679 seed 2701) and to the
# wave-6 cells at seeds 2702 and 2703 (jobs 1681, 1683): flux_capacity with the predicted
# kcat tier and the union deltaG table, the head fix on, and the constraint scale held at
# exactly 0.0 so the flux readout carries the data term alone. The paired contrast against
# gh_wave6_parity_pooled_s2705.slurm is what the seed is for; the split is pinned and only
# weight initialization moves.
#
# WHAT IS NEW HERE, and it is additive. train_bx.py now writes a "feasibility" block into
# the results JSON at the SELECTED epoch (per-genotype mass-balance residual, box violation
# and second-law violation, with the fraction of the top 100 predicted producers that is
# feasible) and saves the selected-epoch trainable parameters to
# $DATA_ROOT/checkpoints-027/, so the cell can be re-audited after the fact by
# audit_top_strain_feasibility.py. Neither changes what is trained or how the epoch is
# picked, so this cell is still a parity replicate of wave-5 G7.
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_wave6_parity_nocstr_s2705.slurm
#
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w6-nocstr-s2705
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
# crash would be the silent version: six seeds pooled into one contrast, half of them run
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
    --arm flux_capacity --seed 2705 --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union \
    --constraint-warmup-epochs 20 --constraint-scale 0.0 --head-normalize \
    --wandb-group gh_wave6_000 --tag _w6 \
    --out wave5_g7_flux_capacity_nocstr_s2705.json
