#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_wave2_flux_capacity.slurm
# [[experiments.027-betaxanthin-metabolic]]
#
# WAVE 2 (2026-09-09), arm "flux_capacity", seed 2700, one GilaHyper GPU, the 019 recipe
# (CRPS head, graph-structure regularizer, and the Mulleder 19-amino-acid co-head) that the
# collapse diagnostics gh_diag_pooled_*
# tested. Four arms run side by side on the four GPUs:
#   pooled          no module
#   flux_off        stoichiometry + GPR availability, no physics
#   flux_capacity   NEW: predicted per-gene enzyme abundance E_g in the capacity bound,
#                   forward pool budget sum MW_g E_g <= p_avail, no thermodynamics
#   flux_abundance  NEW: flux_capacity + anchored thermodynamics (dissipation is a readout)
# The epoch budget is set by the 3-hour window, not by conf/base.yaml's 200, so this is a
# structural comparison at a fixed short budget; selection uses the centered 5-epoch
# validation window and the number reported is the pinned-test Spearman at that epoch.
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_wave2_flux_capacity.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-w2-flux_capacity
#SBATCH --time=02:30:00
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
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-027-wave2
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
EPOCHS="${WAVE2_EPOCHS:-40}"

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm flux_capacity --seed 2700 --epochs "$EPOCHS" --crps --graph-reg --aux-mulleder \
    --abort-on-collapse --wandb-group gh_wave2_000 --tag _w2 \
    --out wave2_flux_capacity_s2700.json
