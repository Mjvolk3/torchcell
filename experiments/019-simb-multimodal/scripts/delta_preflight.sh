#!/bin/bash
# experiments/019-simb-multimodal/scripts/delta_preflight.sh
# [[experiments.019-simb-multimodal.scripts.metabolism_grid_runner]]
#
# RUN THIS ON A DELTA LOGIN NODE BEFORE SUBMITTING. It checks every path the four metabolism
# grid jobs read or write, plus the interpreter, and exits non-zero on the first real problem.
#
#   cd /projects/bbub/mjvolk3/torchcell
#   bash experiments/019-simb-multimodal/scripts/delta_preflight.sh
#
# WHY IT EXISTS: the failure modes here are all silent, late, or expensive.
#   * a missing slurm/output dir kills the job at t=0 with NO LOG ANYWHERE;
#   * a wrong DATA_ROOT makes `Neo4jCellDataset` attempt a REBUILD -- on a compute node with
#     no Neo4j -- rather than erroring cleanly;
#   * a missing embedding surfaces ~20 minutes in, inside the first trial, after the dataset
#     has already loaded.
# Each of those costs a queue slot to discover. This costs ten seconds.
#
# Every path below is derived from `train_cgt_multitask.run_training`, not from memory:
#   dataset_root  = $DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/<dataset_tag>
#   genome/go     = $DATA_ROOT/data/sgd/genome, $DATA_ROOT/data/go
#   graphs        = $DATA_ROOT/data/string, $DATA_ROOT/data/tflink
#   query         = $EXPERIMENT_ROOT/019-simb-multimodal/queries/<query_file>
#   pinned split  = $EXPERIMENT_ROOT/<data_module.pinned_test_split_file>
#   writes        = $DATA_ROOT/{models/checkpoints,wandb-experiments,test-predictions}/<group>
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DELTA_DATA_ROOT="${DELTA_DATA_ROOT:-/scratch/bbub/mjvolk3/torchcell}"
DELTA_CONDA_BASE="${DELTA_CONDA_BASE:-/work/hdd/bbub/miniconda3}"
DELTA_CONDA_ENV="${DELTA_CONDA_ENV:-torchcell}"
DELTA_PY="$DELTA_CONDA_BASE/envs/$DELTA_CONDA_ENV/bin/python"
EXPS=(020-cachera-betaxanthin 021-ozaydin-beta-carotene 022-mulleder-metabolome
  023-metabolome-betaxanthin-joint)

fail=0
ok() { printf '  OK      %-62s %s\n' "$1" "${2:-}"; }
bad() {
  printf '  MISSING %-62s %s\n' "$1" "${2:-}"
  fail=1
}
note() { printf '  ..      %-62s %s\n' "$1" "${2:-}"; }

echo "=========================================================="
echo "project   : $PROJECT_ROOT"
echo "DATA_ROOT : $DELTA_DATA_ROOT"
echo "python    : $DELTA_PY"
echo "=========================================================="

echo
echo "-- 1. interpreter"
if [[ -x "$DELTA_PY" ]]; then
  ok "$DELTA_PY" "$("$DELTA_PY" -c 'import sys;print("py"+".".join(map(str,sys.version_info[:2])))' 2>/dev/null)"
  if "$DELTA_PY" -c 'import torch, torch_geometric, lightning, optuna' 2>/dev/null; then
    ok "imports torch / torch_geometric / lightning / optuna"
  else
    bad "python imports" "install with the ENV'S OWN pip, not a bare pip"
  fi
else
  bad "$DELTA_PY" "set DELTA_CONDA_BASE / DELTA_CONDA_ENV"
fi

echo
echo "-- 2. data the jobs READ (under DATA_ROOT)"
# THE dataset. `Neo4jCellDataset.processed_file_names` is "lmdb", so PyG skips `process()`
# only when processed/lmdb exists. Absent it, the dataset rebuilds -- from a database that is
# not reachable from a compute node.
DS="$DELTA_DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer"
if [[ -d "$DS" ]]; then
  ok "fig6_pigment_transfer" "$(du -sh "$DS" 2>/dev/null | cut -f1)"
  if [[ -e "$DS/processed/lmdb" ]]; then
    ok "  processed/lmdb" "present -- the Neo4j build is skipped"
  else
    bad "  processed/lmdb" "WITHOUT THIS THE JOB TRIES TO REBUILD FROM NEO4J"
  fi
else
  bad "fig6_pigment_transfer" "$DS"
fi
# Only prot_T5_all is needed: `node_embeddings` is frozen, so the other seven embedding trees
# (~810 MB) that sync_delta_fig6.sh still ships are not required by these jobs.
for p in data/scerevisiae/protT5_embedding data/sgd/genome data/go data/string data/tflink; do
  if [[ -d "$DELTA_DATA_ROOT/$p" ]]; then
    ok "$p" "$(du -sh "$DELTA_DATA_ROOT/$p" 2>/dev/null | cut -f1)"
  else
    bad "$p"
  fi
done

echo
echo "-- 3. files the jobs READ (in the repo)"
for f in 019-simb-multimodal/queries/fig6_pigment_transfer.cql \
  020-cachera-betaxanthin/results/merzbacher_nested_split.json; do
  if [[ -f "$PROJECT_ROOT/experiments/$f" ]]; then ok "experiments/$f"; else bad "experiments/$f"; fi
done

echo
echo "-- 4. dirs the jobs WRITE"
# slurm cannot create its own --output directory; a missing one is the single most common way
# a Delta submission fails with nothing to look at.
for e in "${EXPS[@]}"; do
  for d in slurm/output optuna results; do
    if [[ -d "$PROJECT_ROOT/experiments/$e/$d" ]]; then
      ok "experiments/$e/$d"
    else
      bad "experiments/$e/$d" "mkdir -p experiments/$e/{slurm/output,optuna,results}"
    fi
  done
done
# These three the code creates itself; only the parent must be writable.
if [[ -w "$DELTA_DATA_ROOT" ]]; then
  ok "$DELTA_DATA_ROOT writable" "models/checkpoints, wandb-experiments, test-predictions"
else
  bad "$DELTA_DATA_ROOT not writable"
fi

echo
echo "-- 5. headroom"
# 96 runs x 3 retained checkpoints x ~5 MB is ~1.5 GB, plus a wandb dir per run. Small, but a
# full filesystem fails late and without a useful message.
avail=$(df -BG --output=avail "$DELTA_DATA_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "$avail" && "$avail" -ge 20 ]]; then
  ok "free space on DATA_ROOT" "${avail}G"
else
  note "free space on DATA_ROOT" "${avail:-?}G -- want >=20G for checkpoints + wandb"
fi

echo
echo "=========================================================="
if [[ "$fail" == 0 ]]; then
  echo "PREFLIGHT PASSED -- safe to sbatch."
else
  echo "PREFLIGHT FAILED -- fix the MISSING lines above before submitting." >&2
fi
exit "$fail"
