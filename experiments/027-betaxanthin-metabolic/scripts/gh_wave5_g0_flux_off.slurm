#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave5_g0_flux_off.slurm
# [[experiments.027-betaxanthin-metabolic]]
#
# WAVE 5 (2026-09-10), one GilaHyper GPU, seed 2700, 80 epochs, the recipe the pooled arm
# escapes the mean plateau under (CRPS head + graph regularizer 6.5e-5, betaxanthin only,
# two layers), no collapse abort.
#
# G0. Warmup ALONE, on the arm with no capacity, no budget and no thermodynamics. If the
# flux path escapes here, the schedule is what the flux arms were missing; if it does not,
# the constraint weights are not what holds them on the plateau and the flux parameterization
# itself is.
#
# MEASURED CONTEXT, not a claim about this run: under the same recipe at 40 epochs the two
# flux arms sat at a validation prediction spread of 0.000 with a validation DATA loss of 1
# to 5 (jobs 1636, 1637), while the pooled arm reached spread 0.33 and val Pearson +0.24 at
# a data loss of 0.49 (job 1634).
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_wave5_g0_flux_off.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w5-flux_off-warm0.1
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
# The abort is off in this wave anyway (--abort-on-collapse is not passed); the patience is
# pinned so a stray environment cannot reintroduce the ten-epoch abort that killed every run
# which would later have escaped.
export COLLAPSE_PATIENCE_EPOCHS=200
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave5
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE5_EPOCHS:-80}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_off --seed 2700 --epochs "$EPOCHS" --crps --graph-reg \
    --constraint-warmup-epochs 20 --constraint-scale 0.1 \
    --wandb-group gh_wave5_000 --tag _w5 \
    --out wave5_g0_flux_off_s2700.json
