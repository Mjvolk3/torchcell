#!/bin/bash
# experiments/030-solid-growth-multi/scripts/perlmutter_chain_030.sh
# [[experiments.030-solid-growth-multi.scripts.perlmutter_chain_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/perlmutter_chain_030.sh
#
# Submit ONE 030 run on Perlmutter as a chain of 24 h jobs (perlmutter_cgt_030.slurm):
# link 1 starts fresh after the jobs in AFTER, every later link waits on the previous one
# with --dependency=afterany and resumes its `<run>-last.ckpt` (RESUME_FROM_JOB). Run on
# the Perlmutter login node from the 030 code tree:
#
#   AFTER=60163698:60163696 bash perlmutter_chain_030.sh <job-name> <config> <seed> <s3r|s3q> <links> [hydra overrides...]
#
# At the measured 4 x A100 pace (36 min/epoch on IGB mmli) a 24 h link holds about 38
# epochs, so 200 epochs is 6 links; the last link ends early when max_epochs is reached.
set -euo pipefail
NAME="${1:?job name}"; CONFIG="${2:?config}"; SEED="${3:?seed}"; STORE="${4:?s3r|s3q}"; LINKS="${5:?links}"
shift 5
OVERRIDES=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POOL_STORE_REL="data/torchcell/experiments/030-solid-growth-multi/001-multi-build-${STORE}"
DEP="${AFTER:-}"
PREV=""
for i in $(seq 1 "$LINKS"); do
  if [[ -n "$PREV" ]]; then
    ARGS=(--parsable -J "${NAME}-L${i}" --dependency="afterany:${PREV}"
          --export=ALL,POOL_STORE_REL="$POOL_STORE_REL",RESUME_FROM_JOB="$PREV")
  elif [[ -n "$DEP" ]]; then
    ARGS=(--parsable -J "${NAME}-L${i}" --dependency="afterany:${DEP}"
          --export=ALL,POOL_STORE_REL="$POOL_STORE_REL")
  else
    ARGS=(--parsable -J "${NAME}-L${i}" --export=ALL,POOL_STORE_REL="$POOL_STORE_REL")
  fi
  JOB=$(sbatch "${ARGS[@]}" "$HERE/perlmutter_cgt_030.slurm" "$CONFIG" "+seed=$SEED" "${OVERRIDES[@]}")
  echo "link $i: job $JOB (${NAME}-L${i}) after ${PREV:-${DEP:-nothing}}"
  PREV="$JOB"
done
