#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/alcf/run_cell_043.sh
# [[experiments.043-expression-proteome-morphology.scripts.alcf.run_cell_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/run_cell_043.sh
#
# One cell on one Polaris A100 (CUDA_VISIBLE_DEVICES set by the caller): the stage-1 run,
# then, when FINETUNE=1, its Kemmeren fine-tune from the stage-1 last.pt. Both stages carry
# a fixed resume_id, so any later job resumes each from its last.pt and the same W&B run; a
# finished stage writes a DONE marker and is skipped without loading data.
#
#   run_cell_043.sh <cell id> <finetune 0|1> [row options] <hydra overrides of stage 1 ...>
#
# Row options (ahead of the hydra overrides):
#   profile=ontology (default): round ontology_2026_10_23, run id o1023_<cell>, fine-tune at
#     batch 8 (lr 3e-4, warmup 30, 20 epochs: the settings of GilaHyper job 6311).
#   profile=aurora: a cell of round aurora_2026_10_24 moved to Polaris (owner, 2026-10-25),
#     run id a1024b_<cell> (rid_prefix=<p> overrides), the Aurora common overrides with
#     device=cuda, its checkpoint root, so it resumes from the Aurora last.pt copied to
#     eagle (config_matches ignores device and wandb.tags); the fine-tune keeps the
#     stage-1 micro-batch (effective batch 8). ckpt_every=none drops
#     budget.checkpoint_every_steps for a run whose last.pt predates that key.
# CELL_DONE (env, optional): a file written when every stage of the cell is finished
# (rank_cell_043.sh's claim loop reads it).
set -euo pipefail
cell=$1 finetune=$2; shift 2
profile=ontology prefix="" ckpt="++budget.checkpoint_every_steps=1000"
args=()
for o in "$@"; do
  case "$o" in
    profile=*) profile=${o#profile=} ;;
    rid_prefix=*) prefix=${o#rid_prefix=} ;;
    ckpt_every=none) ckpt="" ;;
    *) args+=("$o") ;;
  esac
done
set -- "${args[@]}"
: "${PY:?}" "${DATA_ROOT:?}"
TRAIN=experiments/043-expression-proteome-morphology/scripts/train_043.py
EXP=$DATA_ROOT/experiments/043-expression-proteome-morphology
case "$profile" in
  ontology)
    rid="${prefix:-o1023}_${cell}"
    ROOT=$EXP/checkpoints_ontology_2026_10_23
    common=(data.store_version=v2 data.prep_version=prep_v4 device=cuda data.num_workers=4
      data.sampler=records data.mark_cap=null budget.time_budget_hours=null
      "wandb.tags=[ontology_2026_10_23,polaris]" wandb.round=ontology_2026_10_23)
    ;;
  aurora)
    rid="${prefix:-a1024b}_${cell}"
    ROOT=$EXP/checkpoints_aurora_2026_10_24
    common=(data.store_version=v2 data.prep_version=prep_v4 device=cuda data.num_workers=4
      data.sampler=records data.mark_cap=null budget.time_budget_hours=null
      "wandb.tags=[aurora_2026_10_24,polaris]" wandb.round=aurora_2026_10_24)
    [ -n "$ckpt" ] && common+=("$ckpt")
    ;;
  *) echo "[cell $cell] unknown profile $profile" >&2; exit 2 ;;
esac

LIVE=experiments/043-expression-proteome-morphology/scripts/alcf/wandb_run_live_043.py
# A cell queued on several routes runs under one run id. A stage whose W&B run is live
# elsewhere (state running, heartbeat under 10 min) is not started. A run killed by an
# earlier job's walltime still reads live until its heartbeat ages past 10 min, so the
# check is retried every 2 min for up to 14 min before the stage is given up.
free_or_wait() {  # free_or_wait <run id>: 0 = free to start, 1 = live elsewhere
  local id=$1 k
  for k in 0 1 2 3 4 5 6 7; do
    "$PY" "$LIVE" "$id" || return 0
    [ "$k" = 7 ] && return 1
    sleep 120
  done
}
DONE1="$ROOT/$cell/STAGE1_DONE_${rid}" DONEFT="$ROOT/${cell}_ft/FINETUNE_DONE_${rid}"
cell_done() { [ -n "${CELL_DONE:-}" ] && date > "$CELL_DONE"; echo "[cell $cell] done at $(date)"; }

if [ -e "$DONE1" ]; then
  echo "[cell $cell] stage 1 already finished ($DONE1)"
else
  free_or_wait "$rid" || { echo "[cell $cell] $rid is live on another route: not started here"; exit 3; }
  echo "[cell $cell] stage 1 ($rid) on card ${CUDA_VISIBLE_DEVICES:-?} host $(hostname) job ${PBS_JOBID:-?} at $(date): $*"
  "$PY" "$TRAIN" "${common[@]}" "$@" resume_id="$rid" paths.checkpoint_root="$ROOT/$cell"
  mkdir -p "$ROOT/$cell" && date > "$DONE1"
fi
[ "$finetune" = 1 ] || { cell_done; exit 0; }
[ -e "$DONEFT" ] && { echo "[cell $cell] fine-tune already finished"; cell_done; exit 0; }

shopt -s nullglob
ckpts=("$ROOT/$cell"/*/"$rid"/last.pt)
[ "${#ckpts[@]}" = 1 ] || { echo "[cell $cell] expected one stage-1 last.pt, found ${#ckpts[@]}" >&2; exit 4; }
keep=()
micro=8
for o in "$@"; do
  case "$o" in
    loss=*|seed=*|weighting=*|model.*) keep+=("$o") ;;
    data.batch_size=1) [ "$profile" = aurora ] && micro=1 ;;
    data.batch_size=2) [ "$profile" = aurora ] && micro=2 ;;
  esac
done
ft_batch=(data.batch_size=8)
[ "$micro" != 8 ] && ft_batch=(data.batch_size=$micro ++optim.accumulate_steps=$((8 / micro)))
free_or_wait "${rid}_ft" || { echo "[cell $cell] ${rid}_ft is live on another route"; exit 3; }
echo "[cell $cell] fine-tune from ${ckpts[0]} at $(date): ${keep[*]} ${ft_batch[*]}"
"$PY" "$TRAIN" "${common[@]}" "${keep[@]}" arm=alone_kemmeren init_from="${ckpts[0]}" \
  "${ft_batch[@]}" optim.lr=3.0e-4 optim.warmup_steps=30 budget.max_epochs=20 \
  budget.eval_every_epochs=1 resume_id="${rid}_ft" paths.checkpoint_root="$ROOT/${cell}_ft"
mkdir -p "$ROOT/${cell}_ft" && date > "$DONEFT"
cell_done
