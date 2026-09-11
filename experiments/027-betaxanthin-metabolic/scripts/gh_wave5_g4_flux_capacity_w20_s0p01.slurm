#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave5_g4_flux_capacity_w20_s0p01.slurm
# [[experiments.027-betaxanthin-metabolic]]
#
# WAVE 5 (2026-09-10) CONTINGENCY CELL, one GilaHyper GPU, seed 2700, 80 epochs, the same
# recipe as the four primary cells (CRPS head + graph regularizer 6.5e-5, betaxanthin only,
# no collapse abort).
#
# G4. G1 at a hundredth of the configured constraint weight. Run only if G1 is still on the
# mean plateau at epoch 40, in which case a tenth was not weak enough to separate a
# constraint effect from a parameterization effect.
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_wave5_g4_flux_capacity_w20_s0p01.slurm
#
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w5-flux_capacity-warm0.01
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
export NUM_WORKERS=3
export COLLAPSE_PATIENCE_EPOCHS=200
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave5
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE5_EPOCHS:-80}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_capacity --seed 2700 --epochs "$EPOCHS" --crps --graph-reg \
    --kcat-tier predicted --thermo-source union --constraint-warmup-epochs 20 --constraint-scale 0.01 \
    --wandb-group gh_wave5_000 --tag _w5 \
    --out wave5_g4_flux_capacity_w20_s0p01_s2700.json
