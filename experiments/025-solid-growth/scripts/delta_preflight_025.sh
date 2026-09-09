#!/bin/bash
# experiments/025-solid-growth/scripts/delta_preflight_025.sh
# [[experiments.025-solid-growth.scripts.delta_cgt]]
#
# RUN ON A DELTA LOGIN NODE, FROM THE REPO ROOT, BEFORE SUBMITTING. Checks every path the
# sweep reads or writes, the interpreter, and the Taiga mount, and exits non-zero on the
# first real problem. Ten seconds here against a queue slot per silent failure.
#
#   cd /scratch/bbub/mjvolk3/torchcell && git pull && bash experiments/025-solid-growth/scripts/delta_preflight_025.sh
set -uo pipefail
# The Taiga export is the LOWERCASE path (NCSA SUP-29573, verified 2026-09-09); the
# capitalized spelling in the ticket does not exist on the server.
TAIGA_ZHAO5="/taiga/illinois/eng/chbe/zhao5"

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DELTA_DATA_ROOT="${DELTA_DATA_ROOT:-/scratch/bbub/mjvolk3/torchcell}"
DELTA_CONDA_BASE="${DELTA_CONDA_BASE:-/work/hdd/bbub/miniconda3}"
DELTA_CONDA_ENV="${DELTA_CONDA_ENV:-torchcell}"
DELTA_PY="$DELTA_CONDA_BASE/envs/$DELTA_CONDA_ENV/bin/python"
BUILD="$DELTA_DATA_ROOT/data/torchcell/experiments/010-kuzmin-tmi/001-small-build-schema-v2"

fail=0
ok() { printf '  OK      %-70s %s\n' "$1" "${2:-}"; }
bad() { printf '  MISSING %-70s %s\n' "$1" "${2:-}"; fail=1; }
note() { printf '  ..      %-70s %s\n' "$1" "${2:-}"; }

echo "project   : $PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null) on $(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null))"
echo "DATA_ROOT : $DELTA_DATA_ROOT"

echo; echo "-- 1. interpreter"
if [[ -x "$DELTA_PY" ]]; then
  ok "$DELTA_PY" "$("$DELTA_PY" -c 'import sys;print("py"+".".join(map(str,sys.version_info[:2])))')"
  "$DELTA_PY" -c 'import torch, torch_geometric, lightning, lmdb, wandb, hydra' 2>/dev/null \
    && ok "imports torch / torch_geometric / lightning / lmdb / wandb / hydra" \
    || bad "python imports" "install with the ENV'S OWN pip"
  f=$(PYTHONPATH="$PROJECT_ROOT" "$DELTA_PY" -c 'import torchcell;print(torchcell.__file__)' 2>/dev/null)
  [[ "$f" == "$PROJECT_ROOT"* ]] && ok "torchcell imports from this checkout" || bad "torchcell import" "resolves to ${f:-nothing}"
else
  bad "$DELTA_PY" "set DELTA_CONDA_BASE / DELTA_CONDA_ENV"
fi

echo; echo "-- 2. the 010 build (ship with sync_delta_010_build.sh from GilaHyper)"
for p in processed/lmdb processed/gene_set.json processed/experiment_types.json processed/label_df.parquet data_module_cache; do
  [[ -e "$BUILD/$p" ]] && ok "$BUILD/$p" || bad "$BUILD/$p"
done
[[ -d "$BUILD" ]] && note "size" "$(du -sh "$BUILD" 2>/dev/null | cut -f1)"

echo; echo "-- 3. graph roots (shipped for 019)"
for p in data/sgd/genome data/go data/string data/tflink; do
  [[ -d "$DELTA_DATA_ROOT/$p" ]] && ok "$DELTA_DATA_ROOT/$p" || bad "$DELTA_DATA_ROOT/$p"
done

echo; echo "-- 4. index artifacts and configs in this checkout"
for a in subset_010build_all_indices.json.gz pinned_splits_010build_seed_42.json.gz query_pair_disjoint_splits_010build_seed_42.json.gz; do
  [[ -f "$PROJECT_ROOT/experiments/025-solid-growth/results/$a" ]] && ok "results/$a" || bad "results/$a" "git pull, or run make_010build_index_artifacts.py"
done
for c in cgt_010b_r_kl_005 cgt_010b_r_mask_006 cgt_010b_q_kl_007; do
  [[ -f "$PROJECT_ROOT/experiments/025-solid-growth/conf/$c.yaml" ]] && ok "conf/$c.yaml" || bad "conf/$c.yaml"
done

echo; echo "-- 5. writes"
out="$DELTA_DATA_ROOT/experiments/025-solid-growth/slurm/output"
[[ -d "$out" ]] && ok "$out" || bad "$out" "mkdir -p it: SLURM opens --output before the job script runs"
for p in models/checkpoints wandb-experiments; do
  [[ -d "$DELTA_DATA_ROOT/$p" ]] && ok "$DELTA_DATA_ROOT/$p" || note "$DELTA_DATA_ROOT/$p" "created on first run"
done

echo; echo "-- 6. Taiga (informational; the sweep does not depend on it)"
if [[ -d "$TAIGA_ZHAO5" ]]; then
  ok "$TAIGA_ZHAO5" "$(df -h "$TAIGA_ZHAO5" 2>/dev/null | tail -1 | awk '{print $4" free"}')"
  b025="$DELTA_DATA_ROOT/data/torchcell/experiments/025-solid-growth/001-full-build"
  [[ -e "$b025/processed/lmdb/data.mdb" ]] \
    && ok "$b025 -> $(readlink -f "$b025")" "$(stat -c %s "$b025/processed/lmdb/data.mdb") B" \
    || note "$b025" "025 full build not linked; sync_taiga_025_build.sh + symlink"
else
  note "$TAIGA_ZHAO5" "not visible from this node"
fi

echo; echo "-- 7. accounts"
command -v accounts >/dev/null 2>&1 && accounts 2>/dev/null | grep -E "delta-gpu" || note "accounts" "run it by hand"

echo
if [[ $fail -eq 0 ]]; then echo "PREFLIGHT OK"; else echo "PREFLIGHT FAILED"; exit 1; fi
