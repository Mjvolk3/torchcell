#!/bin/bash
# experiments/tcdb-002-build-speed/scripts/submit_bench.sh
# [[experiments.tcdb-002-build-speed.scripts.submit_bench]]
#
# Freeze the worktree into a wheel, then submit one benchmark arm with that wheel.
#   submit_bench.sh <round> <arm> [cpus] [mem] [extra sbatch args...]
# e.g. submit_bench.sh r0 baseline 48 192G
#      KG_OVERRIDES="adapters.inprocess_max_records=25000" submit_bench.sh r1 inproc-small 48 192G
# The commit is recorded as a tag; a dirty tree is refused so the tag means something.
# KG_OVERRIDES (hydra overrides) is the arm's deviation from the ladder config.
# TIME (sbatch --time, default 1:00:00) must be honest: GilaHyper is one node, so
# backfill starts a job next to the GPU packs only if its limit ends before the next
# pack's reservation. At 12:00:00 arms 2930-2934 sat pending for hours with 32 CPUs
# idle; at 1:00:00 the Bloom arm started within seconds. A full build sets TIME=12:00:00.
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
# The BioCypher schema config is frozen with the wheel: the container reads it from
# the mounted dir, and an arm that adds a node class (r9: interned constant) must run
# against the schema of the commit it was built from, not the build tree's copy.
cp -r "$SRC/biocypher" "$WHEEL_DIR/biocypher"
chmod -R a+rX "$WHEEL_DIR"
export KG_OVERRIDES="${KG_OVERRIDES:-}"
sbatch --export=ALL,WHEEL_DIR="$WHEEL_DIR",ROUND="$ROUND",ARM="$ARM",COMMIT="$COMMIT",KG_CONFIG="${KG_CONFIG:-kg_bench_ladder}",KEEP_CSV="${KEEP_CSV:-0}",PROFILE="${PROFILE:-0}" \
    -J "tcdb002-${ROUND}-${ARM}" --cpus-per-task="$CPUS" --mem="$MEM" --time="${TIME:-1:00:00}" "$@" \
    "$SRC/experiments/tcdb-002-build-speed/scripts/gh_bench_generate.slurm"
echo "wheel: $WHEEL_DIR (commit $COMMIT)"
