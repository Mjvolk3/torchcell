#!/bin/bash
#SBATCH --job-name=035-ms
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/038-env-chemgen-vanacloig-cgt-corrected/experiments/038-env-chemgen-vanacloig-cgt-corrected/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/038-env-chemgen-vanacloig-cgt-corrected/experiments/038-env-chemgen-vanacloig-cgt-corrected/slurm/output/%j_%x.out
#SBATCH --no-requeue
# One sweep file of train_multisource.py on one GPU (SWEEP relative to the experiment folder).
set -euo pipefail
SWEEP=${SWEEP:?SWEEP is required}
PROJECT_ROOT=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/038-env-chemgen-vanacloig-cgt-corrected
EXP=$PROJECT_ROOT/experiments/038-env-chemgen-vanacloig-cgt-corrected
cd "$PROJECT_ROOT"
echo "Running on $(hostname) at $(date -Is), $(git rev-parse --short HEAD), SWEEP=$SWEEP"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export PYTHONPATH="$PROJECT_ROOT"
/home/michaelvolk/miniconda3/envs/torchcell/bin/python "$EXP/scripts/train_multisource.py" --sweep "$EXP/$SWEEP"
echo "finished at $(date -Is)"
