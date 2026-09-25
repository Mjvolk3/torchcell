#!/bin/bash
#SBATCH --job-name=030-closure
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-solid-growth-multi/experiments/030-solid-growth-multi/slurm/output/%j_%x.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-solid-growth-multi/experiments/030-solid-growth-multi/slurm/output/%j_%x.out
#SBATCH --no-requeue

# The closure recompute on the finished 030 build: one read over the 955 GB processed
# store (every triple, the doubles inside them, every single) into a parquet cache, then
# the identity under the Kuzmin-first policy. Reads the store read-only. Memory is page
# cache plus the entry frame; 32 CPUs at the 4.1 GB MaxMemPerCPU cap is 128 GB, a
# quarter of the node.
#
#   STAGE=scan    sbatch experiments/030-solid-growth-multi/scripts/gh_closure_recompute_030.slurm
#   STAGE=analyze sbatch ...   (or STAGE=all, the default)
#   BUILD=/db/experiments/029-solid-growth-ko-001-ko-build CACHE=... LIMIT=20000  for the smoke test

echo "Running on $(hostname) at $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

PROJECT_ROOT="${PROJECT_ROOT:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-solid-growth-multi}"
cd "$PROJECT_ROOT" || exit 1
echo "PROJECT_ROOT=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"
export PYTHONPATH="$PROJECT_ROOT"
set -a; source "$PROJECT_ROOT/.env"; set +a

STAGE="${STAGE:-all}"
ARGS=()
[[ -n "${BUILD:-}" ]] && ARGS+=(--build "$BUILD")
[[ -n "${CACHE:-}" ]] && ARGS+=(--cache "$CACHE")
[[ -n "${RESULTS:-}" ]] && ARGS+=(--results "$RESULTS")
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit-triples "$LIMIT")
[[ -n "${REF_025:-}" ]] && ARGS+=(--reference-025 "$REF_025")
[[ -n "${LABEL:-}" ]] && ARGS+=(--label "$LABEL")
ARGS+=(--workers "$SLURM_CPUS_PER_TASK")

SCRIPT=experiments/030-solid-growth-multi/scripts/closure_recompute_030.py
if [[ "$STAGE" == "scan" || "$STAGE" == "all" ]]; then
  python "$SCRIPT" scan "${ARGS[@]}" || exit 1
fi
if [[ "$STAGE" == "analyze" || "$STAGE" == "all" ]]; then
  python "$SCRIPT" analyze "${ARGS[@]}" || exit 1
fi
echo "DONE at $(date)"
