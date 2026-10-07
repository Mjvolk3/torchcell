#!/bin/bash
#SBATCH --job-name=035-vanacloig
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/038-env-chemgen-vanacloig-cgt-corrected/experiments/038-env-chemgen-vanacloig-cgt-corrected/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/038-env-chemgen-vanacloig-cgt-corrected/experiments/038-env-chemgen-vanacloig-cgt-corrected/slurm/output/%j_%x.out
#SBATCH --no-requeue

# One arm of experiment 035 on one GPU, its folds in sequence.
#
#   sbatch --job-name=035-cgt_env --export=ALL,ARM=cgt_env \
#     experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/gh_train_vanacloig_cgt.slurm
#
# ARM      cgt_genes | cgt_env | embedding_env
# FOLDS    space-separated fold numbers, default "0 1 2 3 4"
# SEED     default 0
# EXTRA    further hydra overrides, e.g. "trainer.max_epochs=1 results_subdir=smoke".
#          No commas: sbatch splits --export on them, so a list override takes one item.
set -euo pipefail

ARM=${ARM:?ARM is required}
FOLDS=${FOLDS:-0 1 2 3 4}
SEED=${SEED:-0}
EXTRA=${EXTRA:-}

echo "Running on $(hostname) at $(date -Is)"
PROJECT_ROOT=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/038-env-chemgen-vanacloig-cgt-corrected
PYTHON=/home/michaelvolk/miniconda3/envs/torchcell/bin/python
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git rev-parse --short HEAD))"
echo "ARM=$ARM FOLDS=$FOLDS SEED=$SEED EXTRA=$EXTRA"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
# the embedding arm runs on CPU (--gres=gpu:0), where nvidia-smi finds no device
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv
fi
export PYTHONPATH="$PROJECT_ROOT"

for fold in $FOLDS; do
  echo "=== fold $fold $(date -Is)"
  # shellcheck disable=SC2086
  $PYTHON experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/train_vanacloig_cgt.py \
    arm="$ARM" fold="$fold" seed="$SEED" $EXTRA
done

echo "finished at $(date -Is)"
