#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/alcf/aurora_run_cell_043.sh
# [[experiments.043-expression-proteome-morphology.scripts.alcf.aurora_run_cell_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/aurora_run_cell_043.sh
#
# One cell of round aurora_2026_10_24 on one Aurora tile (ZE_AFFINITY_MASK set by the
# caller): the stage-1 run on prep_v4, then (fine-tune 1) its Kemmeren fine-tune from the stage-1 last.pt
# (20 epochs, lr 3e-4, 30 warmup steps, effective batch 8 = micro 2 x accumulate 4, or
# micro 1 x accumulate 8 when stage 1 ran at micro-batch 1). Both
# stages carry a fixed resume_id (a1024_<cell> and a1024_<cell>_ft), so a rerun PBS job
# (-r y) resumes each from its last.pt and the same W&B run; a finished stage reruns
# nothing. Unchunked SDPA at micro-batch 2 or 1 (Aurora jobs 8959230 and 8959263: the
# one-call path at batch 8 exceeds the 64 GB tile and page-faults; the chunked path goes
# non-finite, open).
#
#   aurora_run_cell_043.sh <cell id> <fine-tune 0|1> <hydra overrides of stage 1 ...>
#
# The fine-tune keeps the stage-1 loss, seed, weighting and model section (init_from
# requires an equal model section) and sets its own arm, batch, lr, warmup and epochs.
set -euo pipefail
cell=$1 finetune=$2; shift 2
rid="${RID_PREFIX:-a1024}_${cell}"
: "${PY:?}" "${DATA_ROOT:?}" "${TRAIN:?}" "${TILES_DIR:?node-local tile claims, set by the grid job}"
ulimit -c 0  # a fault's core file is 10-12 GB; two are kept from PBS 8959387 for ALCF
ROOT=$DATA_ROOT/experiments/043-expression-proteome-morphology/checkpoints_aurora_2026_10_24
common=(data.store_version=v2 data.prep_version=prep_v4 device=xpu data.num_workers=4
  data.sampler=records data.mark_cap=null budget.time_budget_hours=null
  "wandb.tags=[aurora_2026_10_24,aurora]" wandb.round=aurora_2026_10_24
  "++budget.checkpoint_every_steps=${CKPT_EVERY:-1000}")

# A GPU page fault (intel-compute-runtime "Segmentation fault from GPU ... NotPresent")
# aborts a process at no particular batch (PBS 8959387/8959388: 7 of 24 cells within 85
# min; the census aurora_batch_census_043.py finds no heavy micro-batch at the faults), and
# afterwards every launch on that tile dies at its first step (exit 139) while a plain
# matmul still runs. So a failed attempt marks its tile bad and the retry claims a free,
# unmarked tile of the node (claims are mkdir locks in $TILES_DIR, node-local); with none
# free the cell stops. A retry resumes from last.pt, written every CKPT_EVERY optimizer
# steps (default 1,000, about 15 min at micro-batch 2), under the same resume_id, so W&B
# drops the step rows the retry repeats.
ATTEMPTS=${ATTEMPTS:-5}
tile=${ZE_AFFINITY_MASK:?}
mkdir -p "$TILES_DIR"
mkdir "$TILES_DIR/claim_$tile" 2>/dev/null || true  # the grid job assigns distinct tiles
next_tile() {
  local t
  for t in 0 1 2 3 4 5 6 7 8 9 10 11; do
    [ -e "$TILES_DIR/bad_$t" ] && continue
    mkdir "$TILES_DIR/claim_$t" 2>/dev/null && { echo "$t"; return 0; }
  done
  return 1
}
attempt() {
  local stage=$1 n rc
  shift
  for ((n = 1; n <= ATTEMPTS; n++)); do
    ZE_AFFINITY_MASK=$tile "$PY" "$TRAIN" "$@" && return 0
    rc=$?
    echo "[cell $cell] $stage attempt $n of $ATTEMPTS failed (exit $rc) on tile $tile at $(date -u +%FT%TZ)"
    mkdir -p "$TILES_DIR/bad_$tile"
    [ "$n" -lt "$ATTEMPTS" ] || break
    tile=$(next_tile) || { echo "[cell $cell] no free healthy tile left; stopping"; return 1; }
    echo "[cell $cell] moving to tile $tile"
  done
  return 1
}

echo "[cell $cell] stage 1 on tile $tile at $(date): $*"
attempt "stage 1" "${common[@]}" "$@" resume_id="$rid" paths.checkpoint_root="$ROOT/$cell"

[ "$finetune" = 1 ] || { echo "[cell $cell] done (no fine-tune) at $(date)"; exit 0; }
shopt -s nullglob
ckpts=("$ROOT/$cell"/*/"$rid"/last.pt)
[ "${#ckpts[@]}" = 1 ] || { echo "[cell $cell] expected one stage-1 last.pt, found ${#ckpts[@]}" >&2; exit 4; }
keep=()
micro=2
for o in "$@"; do
  case "$o" in
    loss=*|seed=*|weighting=*|model.*) keep+=("$o") ;;
    data.batch_size=1) micro=1 ;;  # a cell that fits a tile only at micro-batch 1
  esac
done
echo "[cell $cell] fine-tune from ${ckpts[0]} at $(date): ${keep[*]}"
attempt "fine-tune" "${common[@]}" "${keep[@]}" arm=alone_kemmeren init_from="${ckpts[0]}" \
  data.batch_size=$micro ++optim.accumulate_steps=$((8 / micro)) optim.lr=3.0e-4 optim.warmup_steps=30 \
  budget.max_epochs=20 budget.eval_every_epochs=1 resume_id="${rid}_ft" \
  paths.checkpoint_root="$ROOT/${cell}_ft"
echo "[cell $cell] done at $(date)"
