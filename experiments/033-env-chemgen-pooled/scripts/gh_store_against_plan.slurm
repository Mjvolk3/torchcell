#!/bin/bash
#SBATCH --job-name=033-plan-check
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Steps 2 and 3 of the post-query pass alone, for a rerun after the analysis changes: the
# cell table that gh_flatten_cells.slurm wrote is read, not rebuilt. Single-threaded; the
# full pass under slurm 2988 peaked at 13.5 G resident and took 6 minutes for these steps.
#
#   sbatch experiments/033-env-chemgen-pooled/scripts/gh_store_against_plan.slurm
set -euo pipefail

echo "Running on $(hostname) at $(date -Is)"
PROJECT_ROOT=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled
PLAN_RESULTS=${PLAN_RESULTS:-/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/031-env-chemgen-vanacloig-hillenmeyer/experiments/031-env-chemgen-inhibitor-tolerance/results}
PYTHON=/home/michaelvolk/miniconda3/envs/torchcell/bin/python
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git rev-parse --short HEAD))"
echo "PLAN_RESULTS=$PLAN_RESULTS ($(git -C "$PLAN_RESULTS" rev-parse --short HEAD))"
export PYTHONPATH="$PROJECT_ROOT"

$PYTHON experiments/033-env-chemgen-pooled/scripts/store_against_plan.py --plan-results "$PLAN_RESULTS"
$PYTHON experiments/033-env-chemgen-pooled/scripts/post_query_figures.py --stable
$PYTHON experiments/033-env-chemgen-pooled/scripts/notes_tex_post_query_tables.py

echo "finished at $(date -Is)"
