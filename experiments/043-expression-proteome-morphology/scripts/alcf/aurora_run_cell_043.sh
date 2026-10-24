#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/alcf/aurora_run_cell_043.sh
# [[experiments.043-expression-proteome-morphology.scripts.alcf.aurora_run_cell_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/aurora_run_cell_043.sh
#
# One cell of round aurora_2026_10_24 on one Aurora tile (ZE_AFFINITY_MASK set by the
# caller): the stage-1 run on prep_v4, then its Kemmeren fine-tune from the stage-1 last.pt
# (20 epochs, lr 3e-4, 30 warmup steps, effective batch 8 = micro 2 x accumulate 4, or
# micro 1 x accumulate 8 when stage 1 ran at micro-batch 1). Both
# stages carry a fixed resume_id (a1024_<cell> and a1024_<cell>_ft), so a rerun PBS job
# (-r y) resumes each from its last.pt and the same W&B run; a finished stage reruns
# nothing. Unchunked SDPA at micro-batch 2 or 1 (Aurora jobs 8959230 and 8959263: the
# one-call path at batch 8 exceeds the 64 GB tile and page-faults; the chunked path goes
# non-finite, open).
#
#   aurora_run_cell_043.sh <cell id> <hydra overrides of stage 1 ...>
#
# The fine-tune keeps the stage-1 loss, seed, weighting and model section (init_from
# requires an equal model section) and sets its own arm, batch, lr, warmup and epochs.
set -euo pipefail
cell=$1; shift
rid="a1024_${cell}"
: "${PY:?}" "${DATA_ROOT:?}" "${TRAIN:?}"
ROOT=$DATA_ROOT/experiments/043-expression-proteome-morphology/checkpoints_aurora_2026_10_24
common=(data.store_version=v2 data.prep_version=prep_v4 device=xpu data.num_workers=4
  data.sampler=records data.mark_cap=null budget.time_budget_hours=null
  "wandb.tags=[aurora_2026_10_24,aurora]" wandb.round=aurora_2026_10_24)

echo "[cell $cell] stage 1 on tile ${ZE_AFFINITY_MASK:-?} at $(date): $*"
"$PY" "$TRAIN" "${common[@]}" "$@" resume_id="$rid" paths.checkpoint_root="$ROOT/$cell"

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
"$PY" "$TRAIN" "${common[@]}" "${keep[@]}" arm=alone_kemmeren init_from="${ckpts[0]}" \
  data.batch_size=$micro ++optim.accumulate_steps=$((8 / micro)) optim.lr=3.0e-4 optim.warmup_steps=30 \
  budget.max_epochs=20 budget.eval_every_epochs=1 resume_id="${rid}_ft" \
  paths.checkpoint_root="$ROOT/${cell}_ft"
echo "[cell $cell] done at $(date)"
