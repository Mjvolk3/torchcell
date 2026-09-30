#!/bin/bash
#SBATCH --job-name=035-fact
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/035-env-chemgen-vanacloig-cgt/experiments/035-env-chemgen-vanacloig-cgt/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/035-env-chemgen-vanacloig-cgt/experiments/035-env-chemgen-vanacloig-cgt/slurm/output/%j_%x.out
#SBATCH --no-requeue

# One sweep file of train_factorized.py on one GPU, its configs in sequence.
#
#   sbatch --job-name=035-r2w1-table --export=ALL,SWEEP=conf/factorized/r2_table.yaml \
#     experiments/035-env-chemgen-vanacloig-cgt/scripts/gh_train_factorized.slurm
#
# SWEEP  path relative to the experiment folder
# ONLY   optional space-separated config names
set -euo pipefail
SWEEP=${SWEEP:?SWEEP is required}
ONLY=${ONLY:-}
PROJECT_ROOT=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/035-env-chemgen-vanacloig-cgt
PYTHON=/home/michaelvolk/miniconda3/envs/torchcell/bin/python
EXP=$PROJECT_ROOT/experiments/035-env-chemgen-vanacloig-cgt
cd "$PROJECT_ROOT"
echo "Running on $(hostname) at $(date -Is), $(git rev-parse --short HEAD), SWEEP=$SWEEP ONLY=$ONLY"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="$PROJECT_ROOT"
if [ -n "$ONLY" ]; then
  # shellcheck disable=SC2086
  $PYTHON "$EXP/scripts/train_factorized.py" --sweep "$EXP/$SWEEP" --only $ONLY
else
  $PYTHON "$EXP/scripts/train_factorized.py" --sweep "$EXP/$SWEEP"
fi
echo "finished at $(date -Is)"
