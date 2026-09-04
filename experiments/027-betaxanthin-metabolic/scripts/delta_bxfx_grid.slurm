#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/delta_bxfx_grid.slurm
# [[experiments.027-betaxanthin-metabolic]]
#
# THE 027 DECISION RUN: does the enzyme-constrained thermodynamic flux module improve
# betaxanthin prediction over the same CGT without it, and over Flux Cone Learning?
#
# 9 A40x4 nodes as a SLURM ARRAY (0-8), 4 GPUs each, 2 workers per GPU = 72 workers.
# 6 arms x 24 seeds = 144 cells, sharded BY WHOLE SEED so a wall-clock kill leaves every arm
# at the same seed count.
#
#   ssh mjvolk3@login.delta.ncsa.illinois.edu
#   cd /projects/bbub/mjvolk3/torchcell
#   mkdir -p experiments/027-betaxanthin-metabolic/{slurm/output,optuna,results}
#   sbatch experiments/027-betaxanthin-metabolic/scripts/delta_bxfx_grid.slurm
#
# RUN `delta_bxfx_canary.slurm` FIRST and set --time from what it measures. The 20 h below is
# an extrapolation: 70.0 s/epoch measured for `flux_anchored` on GilaHyper, x 200 epochs x 2
# cells per worker x the 019-measured 1.44x two-per-GPU penalty = 11.2 h, then a 1.0-1.5x band
# for the Delta card and num_workers=0. The canary replaces that band with a number.
#
# IF THE CANARY IS SLOWER THAN 1.5x, halve GRID_ROUNDS to 12 rather than cutting epochs: 12
# groups x 1 cell per worker still gives a balanced 12 seeds, in ~5.6 h. Cutting the epoch
# budget would change the experiment, because the selection search widens with it.
#
# THE mkdir IS NOT OPTIONAL -- see the canary's header for why a missing slurm/output kills
# the job at t=0 with no log anywhere. With an ARRAY it kills all nine tasks.
#
# WHY AN ARRAY RATHER THAN `--nodes=9`: nine one-node tasks backfill independently, so the
# experiment starts as capacity appears instead of waiting for nine nodes to free at once.
# The shard rule needs no inter-node communication -- each worker owns a deterministic,
# disjoint slice in its own SQLite file -- so there is nothing an allocation-wide job would
# buy. A task that never starts costs 12 of the 144 cells, and round-major enqueue keeps what
# did run balanced.
#
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbub-delta-gpu
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=180g
#SBATCH --time=20:00:00
#SBATCH --job-name=027-bxfx
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/027-betaxanthin-metabolic/slurm/output/%x_%A_%a.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/027-betaxanthin-metabolic/slurm/output/%x_%A_%a.out
#
# ACCOUNT: bfjt-delta-gpu (12,797 h as of 2026-09-03). In the header on purpose: every older
# Delta launcher in this repo says `--account=bbtp-delta-gpu` on the submit line, and bbtp NO
# LONGER EXISTS on this allocation.
#
# JOB NAME `027-bxfx` is deliberately distinct from the expression sweep `019-expr-v10`
# (Delta array 21796128) that is pending on the same account, so `squeue`, the slurm output
# paths and the W&B project never collide.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-/projects/bbub/mjvolk3/torchcell}/experiments/027-betaxanthin-metabolic/scripts/delta_bxfx_common.sh"

bxfx_env
bxfx_preflight

N_NODES="${BXFX_N_NODES:-9}"
WORKERS_PER_NODE=$((DELTA_GPUS_PER_NODE * GRID_WORKERS_PER_GPU))
SHARD_COUNT=$((N_NODES * WORKERS_PER_NODE))
FIRST_WORKER=$((SLURM_ARRAY_TASK_ID * WORKERS_PER_NODE))

# The deadline every worker measures its remaining budget against. Derived from the job's own
# start time and wall limit rather than hardcoded, so changing --time changes the budget too
# and the two can never disagree.
JOB_SECONDS="${BXFX_JOB_SECONDS:-$((20 * 3600))}"
DEADLINE=$(( $(date +%s) + JOB_SECONDS ))

echo "array task ${SLURM_ARRAY_TASK_ID}/$((N_NODES - 1)): workers ${FIRST_WORKER}..$((FIRST_WORKER + WORKERS_PER_NODE - 1)) of $SHARD_COUNT"
echo "deadline: $(date -d "@$DEADLINE" 2>/dev/null || echo "$DEADLINE")"

bxfx_run_node "$SHARD_COUNT" "$FIRST_WORKER" "$DEADLINE"

echo "=========================================================="
echo "DONE array task ${SLURM_ARRAY_TASK_ID}"
echo "Per-worker results: $EXP_DIR/results/bxfx_w*.json"
echo "Per-gene test predictions: \$DATA_ROOT/test-predictions-027/"
echo "Analyze with: python experiments/027-betaxanthin-metabolic/scripts/analyze_bxfx.py"
echo "=========================================================="
