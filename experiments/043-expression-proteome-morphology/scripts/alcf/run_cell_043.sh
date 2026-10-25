#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/alcf/run_cell_043.sh
# [[experiments.043-expression-proteome-morphology.scripts.alcf.run_cell_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/run_cell_043.sh
#
# One cell of the ontology round (round ontology_2026_10_23): the stage-1 run, then, when
# FINETUNE=1, its Kemmeren fine-tune from the stage-1 last.pt (20 epochs, lr 3e-4, 30 warmup
# steps, batch 8: the settings of GilaHyper job 6311). Both stages carry a fixed resume_id
# (<cell> and <cell>_ft), so a requeued PBS job resumes each from its last.pt and the same
# W&B run; a finished stage reruns nothing. Called by polaris_train_043_ontology_grid.pbs
# with CUDA_VISIBLE_DEVICES already set.
#
#   run_cell_043.sh <cell id> <finetune 0|1> <hydra overrides of stage 1 ...>
#
# The fine-tune keeps the stage-1 loss, seed, weighting and model section (init_from
# requires an equal model section) and sets its own arm, batch, lr, warmup and epochs.
set -euo pipefail
cell=$1 finetune=$2; shift 2
# the W&B run id and checkpoint directory carry the round, so no other round reuses them
rid="o1023_${cell}"
: "${PY:?}" "${DATA_ROOT:?}"
TRAIN=experiments/043-expression-proteome-morphology/scripts/train_043.py
ROOT=$DATA_ROOT/experiments/043-expression-proteome-morphology/checkpoints_ontology_2026_10_23
common=(data.store_version=v2 data.prep_version=prep_v4 device=cuda data.num_workers=4
  data.sampler=records data.mark_cap=null budget.time_budget_hours=null
  "wandb.tags=[ontology_2026_10_23,polaris]" wandb.round=ontology_2026_10_23)

LIVE=experiments/043-expression-proteome-morphology/scripts/alcf/wandb_run_live_043.py
# A cell is queued on several routes (preemptable, capacity, prod chunks) under one run id.
# A stage whose W&B run is live elsewhere (state running, heartbeat under 10 min) is not
# started. A run killed by the previous chunk's walltime still reads live until its
# heartbeat ages past 10 min, so the check is retried every 2 min for up to 14 min before
# the stage is given up to the other route.
free_or_wait() {  # free_or_wait <run id>: 0 = free to start, 1 = live elsewhere
  local id=$1 k
  for k in 0 1 2 3 4 5 6 7; do
    "$PY" "$LIVE" "$id" || return 0
    [ "$k" = 7 ] && return 1
    sleep 120
  done
}
# A stage that completed writes a DONE marker; later chunks skip it without loading data.
DONE1="$ROOT/$cell/STAGE1_DONE" DONEFT="$ROOT/${cell}_ft/FINETUNE_DONE"

if [ -e "$DONE1" ]; then
  echo "[cell $cell] stage 1 already finished ($DONE1)"
else
  free_or_wait "$rid" || { echo "[cell $cell] $rid is live on another route: not started here"; exit 0; }
  echo "[cell $cell] stage 1 on card ${CUDA_VISIBLE_DEVICES:-?} host $(hostname) job ${PBS_JOBID:-?} at $(date): $*"
  "$PY" "$TRAIN" "${common[@]}" "$@" resume_id="$rid" paths.checkpoint_root="$ROOT/$cell"
  date > "$DONE1"
fi

if [ "$finetune" = 1 ]; then
  if [ -e "$DONEFT" ]; then echo "[cell $cell] fine-tune already finished"; exit 0; fi
  shopt -s nullglob
  ckpts=("$ROOT/$cell"/*/"$rid"/last.pt)
  [ "${#ckpts[@]}" = 1 ] || { echo "[cell $cell] expected one stage-1 last.pt, found ${#ckpts[@]}" >&2; exit 4; }
  keep=()
  for o in "$@"; do
    case "$o" in loss=*|seed=*|weighting=*|model.*) keep+=("$o") ;; esac
  done
  free_or_wait "${rid}_ft" || { echo "[cell $cell] ${rid}_ft is live on another route"; exit 0; }
  echo "[cell $cell] fine-tune from ${ckpts[0]} at $(date): ${keep[*]}"
  "$PY" "$TRAIN" "${common[@]}" "${keep[@]}" arm=alone_kemmeren init_from="${ckpts[0]}" \
    data.batch_size=8 optim.lr=3.0e-4 optim.warmup_steps=30 budget.max_epochs=20 \
    budget.eval_every_epochs=1 resume_id="${rid}_ft" paths.checkpoint_root="$ROOT/${cell}_ft"
  mkdir -p "$ROOT/${cell}_ft" && date > "$DONEFT"
fi
echo "[cell $cell] done at $(date)"
