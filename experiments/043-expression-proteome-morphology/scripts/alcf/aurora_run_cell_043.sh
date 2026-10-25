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
# row-level options ahead of the hydra overrides: rid_prefix=<p> (a resumed run keeps its
# old id) and ckpt_every=none (a run whose last.pt predates budget.checkpoint_every_steps;
# the key would make config_matches refuse it)
prefix=${RID_PREFIX:-a1024}
ckpt="++budget.checkpoint_every_steps=${CKPT_EVERY:-1000}"
args=()
for o in "$@"; do
  case "$o" in
    rid_prefix=*) prefix=${o#rid_prefix=} ;;
    ckpt_every=none) ckpt="" ;;
    *) args+=("$o") ;;
  esac
done
set -- "${args[@]}"
rid="${prefix}_${cell}"
: "${PY:?}" "${DATA_ROOT:?}" "${TRAIN:?}" "${TILES_DIR:?node-local tile claims, set by the grid job}"
ulimit -c 0  # a fault's core file is 10-12 GB; two are kept from PBS 8959387 for ALCF
ROOT=$DATA_ROOT/experiments/043-expression-proteome-morphology/checkpoints_aurora_2026_10_24
# ROUND, TAGS, STACK, NUM_WORKERS (env, set by the stack-test env scripts of round
# aurora_2026_10_25): W&B round label, tag list, wandb.stack label and loader workers. Their
# defaults are round aurora_2026_10_24's values. All four are placement keys
# (train_043._strip_placement), so a run resumes from a last.pt written under other values.
ROUND=${ROUND:-aurora_2026_10_24}
TAGS=${TAGS:-[aurora_2026_10_24,aurora]}
common=(data.store_version=v2 data.prep_version=prep_v4 device=xpu
  "data.num_workers=${NUM_WORKERS:-4}" data.sampler=records data.mark_cap=null
  budget.time_budget_hours=null "wandb.tags=$TAGS" "wandb.round=$ROUND")
[ -n "${STACK:-}" ] && common+=("++wandb.stack=$STACK")
[ -n "$ckpt" ] && common+=("$ckpt")

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
LIVE=$(dirname "$TRAIN")/alcf/wandb_run_live_043.py
# A cell queued on two routes (this Aurora job and a Polaris lane copy, same resume_id) trains
# on one: a stage whose W&B run is live elsewhere (state running, heartbeat under 10 min) is
# not started. A run killed by an earlier job still reads live until its heartbeat ages past
# 10 min, so the check is retried every 2 min for up to 14 min (as run_cell_043.sh does).
free_or_wait() {  # free_or_wait <run id>: 0 = free to start, 1 = live elsewhere
  local id=$1 k
  for k in 0 1 2 3 4 5 6 7; do
    "$PY" "$LIVE" "$id" || return 0
    [ "$k" = 7 ] && return 1
    sleep 120
  done
}
attempt() {
  local stage=$1 n rc id
  shift
  for o in "$@"; do case "$o" in resume_id=*) id=${o#resume_id=} ;; esac; done
  free_or_wait "$id" || { echo "[cell $cell] $stage: $id is live on another route; not started here"; return 3; }
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
attempt "stage 1" "${common[@]}" "$@" resume_id="$rid" paths.checkpoint_root="$ROOT/$cell" || exit $?

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
