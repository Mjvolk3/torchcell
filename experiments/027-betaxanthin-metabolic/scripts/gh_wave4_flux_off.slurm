#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave4_flux_off.slurm
# [[experiments.027-betaxanthin-metabolic]]
#
# WAVE 4 (2026-09-09), arm "flux_off", seed 2700, one GilaHyper GPU, 40 epochs, the recipe
# every commit in the July window escaped the mean plateau under at epoch 21 to 24 (CRPS
# head, graph regularizer 6.5e-5, two layers, betaxanthin only), no collapse abort. Runs
# beside the HEAD pooled control on the same recipe (gh_diag_pooled_both40.slurm), so if the
# pooled arm escapes here the module arms are read at the same budget.
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_wave4_flux_off.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w4-flux_off
#SBATCH --time=01:30:00
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
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave4
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE4_EPOCHS:-40}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_off --seed 2700 --epochs "$EPOCHS" --crps --graph-reg \
    --wandb-group gh_wave4_000 --tag _w4 \
    --out wave4_flux_off_s2700.json
