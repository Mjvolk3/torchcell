#!/bin/bash
#SBATCH --job-name=035-analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/038-env-chemgen-vanacloig-cgt-corrected/experiments/038-env-chemgen-vanacloig-cgt-corrected/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/038-env-chemgen-vanacloig-cgt-corrected/experiments/038-env-chemgen-vanacloig-cgt-corrected/slurm/output/%j_%x.out
#SBATCH --no-requeue
# One analysis script of experiment 035 on CPU: SCRIPT=<name>.py ARGS="..." (no commas).
set -euo pipefail
SCRIPT=${SCRIPT:?SCRIPT is required}
ARGS=${ARGS:-}
PROJECT_ROOT=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/038-env-chemgen-vanacloig-cgt-corrected
cd "$PROJECT_ROOT"
echo "Running on $(hostname) at $(date -Is), $(git rev-parse --short HEAD), SCRIPT=$SCRIPT ARGS=$ARGS"
export PYTHONPATH="$PROJECT_ROOT"
# shellcheck disable=SC2086
/home/michaelvolk/miniconda3/envs/torchcell/bin/python "experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/$SCRIPT" $ARGS
echo "finished at $(date -Is)"
