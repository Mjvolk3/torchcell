#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_pos_control_019_metabolism_000.slurm
# [[experiments.027-betaxanthin-metabolic]]
#
# POSITIVE CONTROL for the collapse diagnostics of 2026-09-09. The 019 Lightning trainer
# with its own gh_metabolism_000 config reached validation Pearson 0.434 on this dataset
# build (job 1344). Every 027-loop cell collapses to a constant on the same build. Running
# the 019 trainer again, single GPU, 15 epochs, says whether the 0.434 still reproduces on
# the current tree; if it does, the defect is confined to train_bx.py, and if it does not,
# something under both of them (dataset, embeddings, model) has changed since July.
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_pos_control_019_metabolism_000.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-pos-ctrl-019
#SBATCH --time=02:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT/experiments" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB__SERVICE_WAIT=600
export WANDB_DIR=/scratch/projects/torchcell-scratch/experiments/027-betaxanthin-metabolic/wandb
mkdir -p "$WANDB_DIR"
echo "host $(hostname) gpu $CUDA_VISIBLE_DEVICES; DATA_ROOT=$DATA_ROOT"

python "$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py" \
  --config-name gh_metabolism_000 \
  trainer.max_epochs=15 trainer.devices=1 data_module.num_workers=3 \
  wandb.project=torchcell_027_bxfx
