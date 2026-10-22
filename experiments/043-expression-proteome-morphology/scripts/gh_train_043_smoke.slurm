#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_train_043_smoke.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_train_043_smoke]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_train_043_smoke.slurm
#
# GPU smoke cell of the 043 trainer: joint_9, HL-Gauss, equal weighting, seed 0, batch 8, one
# card. Epochs = min(50, what fits the 7.6 h budget by the MEASURED first epoch (training and
# evaluation), less a 20 min margin); the projection from the first 30 steps is printed at
# startup and logged (setup/probe_seconds_per_step, setup/projected_train_minutes_per_epoch),
# and the fixed epoch count and registered window after epoch 1 ([budget] line).
#
# Needs the prepared store (gh_train_043_prepare.slurm) and every vector the arm's records
# use in alleles/<encoder>.h5: chain it behind the embedding chain (6106 -> 6115 -> 6121 ->
# 6122). A key with no vector raises KeyError at startup; nothing is substituted.
#
#   sbatch --dependency=afterany:6122 experiments/043-expression-proteome-morphology/scripts/gh_train_043_smoke.slurm
#
# Twins (prepared, not submitted): pass ARM / LOSS / WEIGHTING through the environment, e.g.
#   sbatch --export=ALL,ARM=alone_kemmeren --dependency=afterany:<smoke> .../gh_train_043_smoke.slurm
#   sbatch --export=ALL,LOSS=pinball --dependency=afterany:<previous> .../gh_train_043_smoke.slurm
#   sbatch --export=ALL,LOSS=mse --dependency=afterany:<previous> .../gh_train_043_smoke.slurm
#
# --mem: host-side peak is the prepared store (measured at prepare time) plus one encoder's
# full h5 embedding array while the compact table is gathered (FUDT upstream 1.21 M x 768
# fp16 = 1.9 GB) plus the compact tables before they move to the card (about 5.8 GB if every
# vocabulary row is used); 48g covers it with the forked loader workers. Measure the real
# peak with `sacct -j <id> --format=JobID,MaxRSS -P` and tighten.
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=043-train-smoke
#SBATCH --time=08:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export DATA_ROOT=/scratch/projects/torchcell-scratch
export EXPERIMENT_ROOT="$PWD/experiments"
ARM="${ARM:-joint_9}"
LOSS="${LOSS:-hlgauss}"
WEIGHTING="${WEIGHTING:-equal}"
SEED="${SEED:-0}"
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
echo "arm $ARM  loss $LOSS  weighting $WEIGHTING  seed $SEED"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
python experiments/043-expression-proteome-morphology/scripts/train_043.py \
  arm="$ARM" loss="$LOSS" weighting="$WEIGHTING" seed="$SEED" \
  data.batch_size=8 data.num_workers=4 \
  budget.max_epochs=50 budget.time_budget_hours=7.6 budget.time_margin_minutes=20 \
  "wandb.tags=[smoke]"
echo "done $(date)"
