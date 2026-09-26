#!/bin/bash
# experiments/tcdb-002-build-speed/scripts/submit_bench.sh
# [[experiments.tcdb-002-build-speed.scripts.submit_bench]]
#
# Freeze the worktree into a wheel, then submit one benchmark arm with that wheel.
#   submit_bench.sh <round> <arm> [cpus] [mem] [extra sbatch args...]
# e.g. submit_bench.sh r0 baseline 48 192G
# The commit is recorded as a tag; a dirty tree is refused so the tag means something.
set -euo pipefail
ROUND="${1:?round (r0, r1, ...)}"; ARM="${2:?arm name}"
CPUS="${3:-48}"; MEM="${4:-192G}"; shift 4 2>/dev/null || shift $#
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BENCH_ROOT="${BENCH_ROOT:-/scratch/projects/torchcell-scratch/tcdb-002-bench}"
PY=~/miniconda3/envs/torchcell/bin/python
if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]; then
    echo "worktree $SRC has uncommitted tracked changes; commit before submitting an arm" >&2
    exit 1
fi
COMMIT=$(git -C "$SRC" rev-parse --short=8 HEAD)
STAMP=$(date +%Y%m%d-%H%M%S)
WHEEL_DIR="$BENCH_ROOT/wheels/${STAMP}_${ROUND}_${ARM}_${COMMIT}"
mkdir -p "$WHEEL_DIR" "$BENCH_ROOT/slurm"
"$PY" -m pip wheel --no-deps --no-build-isolation -w "$WHEEL_DIR" "$SRC" >/dev/null
ls "$WHEEL_DIR"/torchcell-*.whl >/dev/null
chmod -R a+rX "$WHEEL_DIR"
sbatch --export=ALL,WHEEL_DIR="$WHEEL_DIR",ROUND="$ROUND",ARM="$ARM",COMMIT="$COMMIT",KG_CONFIG="${KG_CONFIG:-kg_bench_ladder}",KEEP_CSV="${KEEP_CSV:-0}" \
    -J "tcdb002-${ROUND}-${ARM}" --cpus-per-task="$CPUS" --mem="$MEM" "$@" \
    "$SRC/experiments/tcdb-002-build-speed/scripts/gh_bench_generate.slurm"
echo "wheel: $WHEEL_DIR (commit $COMMIT)"
