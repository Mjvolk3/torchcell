#!/bin/bash
#SBATCH --job-name=033-post-query
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled/experiments/033-env-chemgen-pooled/slurm/output/%j_%x.out
#SBATCH --no-requeue

# The post-query pass over build 001, three steps in one job:
#
#   1. flatten_cells.py       the processed store (6,042,771 entries, 106 G of JSON) into
#                             one parquet row per (genotype, environment) cell, under
#                             $DATA_ROOT/experiments/033-env-chemgen-pooled/cell_table/
#   2. store_against_plan.py  every claim of the 031 plan re-measured on that table,
#                             into experiments/033-env-chemgen-pooled/results/
#   3. post_query_figures.py  the three figures of notes-tex/033-post-query-analysis
#
# Read-only on /db. PLAN_RESULTS is the results directory of experiment 031, which is on
# its own branch (PR #436) until it lands.
#
# Sizing: the smoke run of step 1 (every 3,000th entry, 2,015 entries) took 7.6 CPU
# seconds including two interpreter starts, so the full pass is about 3 CPU hours and
# about 15 minutes on 16 workers. Steps 2 and 3 are single-threaded. The limit is
# deliberately honest so the job can backfill beside the GPU packs.
#
#   sbatch experiments/033-env-chemgen-pooled/scripts/gh_flatten_cells.slurm
set -euo pipefail

echo "Running on $(hostname) at $(date -Is)"
PROJECT_ROOT=/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/033-env-chemgen-pooled
PLAN_RESULTS=${PLAN_RESULTS:-/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/031-env-chemgen-vanacloig-hillenmeyer/experiments/031-env-chemgen-inhibitor-tolerance/results}
PYTHON=/home/michaelvolk/miniconda3/envs/torchcell/bin/python
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git rev-parse --short HEAD))"
echo "PLAN_RESULTS=$PLAN_RESULTS ($(git -C "$PLAN_RESULTS" rev-parse --short HEAD))"
export PYTHONPATH="$PROJECT_ROOT"

echo "=== 1. flatten $(date -Is)"
$PYTHON experiments/033-env-chemgen-pooled/scripts/flatten_cells.py --workers 16
echo "=== 2. store against plan $(date -Is)"
$PYTHON experiments/033-env-chemgen-pooled/scripts/store_against_plan.py --plan-results "$PLAN_RESULTS"
echo "=== 3. figures $(date -Is)"
$PYTHON experiments/033-env-chemgen-pooled/scripts/post_query_figures.py --stable

echo "finished at $(date -Is)"
