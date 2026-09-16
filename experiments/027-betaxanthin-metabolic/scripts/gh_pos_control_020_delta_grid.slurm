#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_pos_control_020_delta_grid.slurm
# [[experiments.027-betaxanthin-metabolic]]
#
# POSITIVE CONTROL 2 for the collapse diagnostics of 2026-09-09. Control 1 (job 1616, the
# July gh_metabolism_000 config on today's tree) collapsed: pred_sd_ratio 2e-4 over 15
# epochs. The model code gained the hard attention mask, the Hadamard pair term and the
# null sink after July, and the 020 Delta grid (August) trained on that code with
# `delta_grid_betaxanthin.yaml`: hard masking on all nine graphs, L=6, KL off. Running that
# config on the 019 trainer, single GPU, 15 epochs, says whether the current code trains
# betaxanthin at all under its own current recipe.
#
#   cd <worktree-root> && sbatch experiments/027-betaxanthin-metabolic/scripts/gh_pos_control_020_delta_grid.slurm
#
#SBATCH -p main
#SBATCH --mem=12g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --job-name=027-pos-ctrl-020
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
  --config-path "$PROJECT_ROOT/experiments/020-cachera-betaxanthin/conf" --config-name delta_grid_betaxanthin \
  trainer.max_epochs=15 trainer.devices=1 data_module.num_workers=3 \
  wandb.project=torchcell_027_bxfx
