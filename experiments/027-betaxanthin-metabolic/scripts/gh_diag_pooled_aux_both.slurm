#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_diag_pooled_aux_both.slurm
# [[experiments.027-betaxanthin-metabolic]]
#
# COLLAPSE DIAGNOSTIC, cell "aux_both": the no-module (pooled) arm, seed 2700, 25 epochs, on
# one GilaHyper GPU, flags: "--crps --graph-reg --aux-mulleder". The 144-cell Delta grid produced a constant predictor
# in every cell. Two recipe elements of the 019 run that reached 0.434 (job 1344) were
# missing from the grid's loop: the graph-structure regularizer was never passed to the
# model, and the CRPS head's log_sigma was sliced away before the loss so it fell back to
# MSE. The four gh_diag_pooled_* scripts are the 2x2 over those two switches; every run
# logs the validation prediction spread ratio per epoch and aborts once it stays under
# 1 percent of the target spread past epoch 10.
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_diag_pooled_aux_both.slurm
#
#SBATCH -p main
#SBATCH --mem=60g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-diag-aux_both
#SBATCH --time=02:00:00
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
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
    --arm pooled --seed 2700 --epochs 25 --smoke --crps --graph-reg --aux-mulleder \
    --abort-on-collapse --wandb-group gh_diag_000 --tag _diag_aux_both \
    --out diag_pooled_aux_both_s2700.json
