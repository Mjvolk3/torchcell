#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_train_043_sweep_v2.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_train_043_sweep_v2]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_train_043_sweep_v2.slurm
#
# One GPU cell of the 2026-10-24 GilaHyper sweep (round ghyper_sweep_2026_10_24): the
# gh_train_043_overnight_v2.slurm launcher on STORE v2 with the sweep knobs exposed. The
# architecture stays the virtual-cell CGT (composite gene token always on, one model over
# cells, no per-dataset heads); the knobs are how it is optimized, how targets are
# preprocessed, the loss family, the trunk's S/M layer pattern and the widths.
#
# Fixed epochs: the trainer runs exactly MAX_EPOCHS (no trainer time budget) and a
# `timeout` at STOP_HOURS stops it instead (exit 124 = PARTIAL; last.pt is the last finished
# epoch). Every run carries RESUME_ID, which names its checkpoint directory AND its W&B run
# id, so a chained fine-tune finds the stage-1 last.pt and gates on its W&B history.
#
# Environment knobs (--export cannot carry a comma inside one value):
#   RESUME_ID (required)  ARM (all)  LOSS (hlgauss)  WEIGHTING (equal)  SEED (0)
#   SAMPLER (records)  MAX_EPOCHS (80)  STOP_HOURS (11.5)  WINDOW_START_FRAC (0.875, the
#   overnight round's: the last 10 of 80 epochs)  BATCH (8)
#   LAYER_PATTERN  e.g. SMMSS (letters, no commas; unset = the default trunk SMSS)
#   HIDDEN  FFN_MULT (trunk)  HEAD_HIDDEN  TRANSFORM (zscore|quantile_normal|none)
#   STATS_BY (dataset|profile)  WEIGHT_BY (dataset|profile)  SCHEDULE (constant|cosine)
#   LR  WARMUP_STEPS  PREP_VERSION (prep_v4 for ontology or profile)  CONDITIONING (name|ontology)
#   DROPOUT (model.dropout: trunk attention + FFN, operator, heads)  TOKEN_DROPOUT
#   WEIGHT_DECAY  SIGMA_BINS (HL-Gauss smoothing in bin widths)  (2026-10-24 regularization)
#   BALANCE_POWER (with SAMPLER=balanced: record weight (1/n_dataset)^p; 2026-10-25)
#   MAX_VISITS (with BALANCE_POWER: cap draws per epoch at MAX_VISITS x n_records)
#   INJECT (ontology token: head | operator | trunk_and_operator; 2026-10-25)
#   INIT_FROM_ID: stage 2; the stage-1 RESUME_ID. The job first requires the stage-1 W&B run
#     to have logged val/kemmeren/loss (scripts/check_wandb_run_keys.py), then initializes
#     from <checkpoints>/*/<INIT_FROM_ID>/last.pt. A stage-2 job must pass the same model,
#     loss and target knobs as its stage 1 (the trainer refuses otherwise).
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=043-sweep
#SBATCH --time=12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export DATA_ROOT=/scratch/projects/torchcell-scratch
export EXPERIMENT_ROOT="$PWD/experiments"
PROJECT=torchcell_043-expression-proteome-morphology
CKPT="$DATA_ROOT/experiments/043-expression-proteome-morphology/checkpoints"
: "${RESUME_ID:?set RESUME_ID (checkpoint dir and W&B run id)}"
ARM="${ARM:-all}"
LOSS="${LOSS:-hlgauss}"
WEIGHTING="${WEIGHTING:-equal}"
SEED="${SEED:-0}"
SAMPLER="${SAMPLER:-records}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
STOP_HOURS="${STOP_HOURS:-11.5}"
BATCH="${BATCH:-8}"
TAGS="v2_store,ghyper_sweep_2026_10_24"
EXTRA=("budget.registered_window.start_frac=${WINDOW_START_FRAC:-0.875}")
if [ -n "${LAYER_PATTERN:-}" ]; then
  EXTRA+=("model.layer_pattern=[$(echo "$LAYER_PATTERN" | sed 's/./&,/g; s/,$//')]")
fi
if [ -n "${HIDDEN:-}" ]; then EXTRA+=("model.hidden_dim=${HIDDEN}"); fi
if [ -n "${FFN_MULT:-}" ]; then EXTRA+=("model.trunk_ffn_mult=${FFN_MULT}"); fi
if [ -n "${HEAD_HIDDEN:-}" ]; then EXTRA+=("model.head_hidden=${HEAD_HIDDEN}"); fi
if [ -n "${DROPOUT:-}" ]; then EXTRA+=("model.dropout=${DROPOUT}"); fi
if [ -n "${BALANCE_POWER:-}" ]; then EXTRA+=("data.balance_power=${BALANCE_POWER}"); fi
if [ -n "${MAX_VISITS:-}" ]; then EXTRA+=("data.max_visits_per_record_per_epoch=${MAX_VISITS}"); fi
if [ -n "${TOKEN_DROPOUT:-}" ]; then EXTRA+=("model.token_dropout=${TOKEN_DROPOUT}"); fi
if [ -n "${WEIGHT_DECAY:-}" ]; then EXTRA+=("optim.weight_decay=${WEIGHT_DECAY}"); fi
if [ -n "${SIGMA_BINS:-}" ]; then EXTRA+=("loss.hl_gauss.sigma_bins=${SIGMA_BINS}"); fi
if [ -n "${TRANSFORM:-}" ]; then EXTRA+=("data.target_transform=${TRANSFORM}"); fi
if [ -n "${STATS_BY:-}" ]; then EXTRA+=("data.target_stats_by=${STATS_BY}"); fi
if [ -n "${WEIGHT_BY:-}" ]; then EXTRA+=("weighting.by=${WEIGHT_BY}"); fi
if [ -n "${SCHEDULE:-}" ]; then EXTRA+=("optim.schedule=${SCHEDULE}"); fi
if [ -n "${LR:-}" ]; then EXTRA+=("optim.lr=${LR}"); fi
if [ -n "${WARMUP_STEPS:-}" ]; then EXTRA+=("optim.warmup_steps=${WARMUP_STEPS}"); fi
if [ -n "${PREP_VERSION:-}" ]; then EXTRA+=("data.prep_version=${PREP_VERSION}"); fi
if [ -n "${CONDITIONING:-}" ]; then EXTRA+=("model.conditioning.source=${CONDITIONING}"); fi
if [ -n "${INJECT:-}" ]; then EXTRA+=("model.conditioning.inject=${INJECT}"); fi
# space-separated hydra overrides with no env knob of their own (arm.exclude_from_training=[hu]
# for a leave-one-dataset-out cell, 2026-10-26)
if [ -n "${EXTRA_OVERRIDES:-}" ]; then read -r -a more <<< "${EXTRA_OVERRIDES}"; EXTRA+=("${more[@]}"); fi
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
if [ -n "${INIT_FROM_ID:-}" ]; then
  python scripts/check_wandb_run_keys.py "zhao-group/${PROJECT}/${INIT_FROM_ID}" \
    --require val/kemmeren/loss --min-rows 2
  mapfile -t found < <(ls -d "$CKPT"/*/"$INIT_FROM_ID" 2>/dev/null)
  if [ "${#found[@]}" -ne 1 ] || [ ! -f "${found[0]}/last.pt" ]; then
    echo "stage-1 checkpoint for ${INIT_FROM_ID}: ${found[*]:-none}" >&2; exit 3
  fi
  EXTRA+=("init_from=${found[0]}/last.pt")
fi
STOP=(timeout --signal=TERM --kill-after=120 "$(python -c "print(int(${STOP_HOURS} * 3600))")")
echo "resume_id $RESUME_ID  arm $ARM  loss $LOSS  weighting $WEIGHTING  seed $SEED  epochs $MAX_EPOCHS  stop ${STOP_HOURS} h  batch $BATCH  extra ${EXTRA[*]}  tags $TAGS"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
rc=0
"${STOP[@]}" python experiments/043-expression-proteome-morphology/scripts/train_043.py \
  data.store_version=v2 resume_id="$RESUME_ID" \
  arm="$ARM" loss="$LOSS" weighting="$WEIGHTING" seed="$SEED" \
  data.batch_size="$BATCH" data.num_workers=4 data.sampler="$SAMPLER" data.mark_cap=null \
  budget.max_epochs="$MAX_EPOCHS" budget.time_budget_hours=null \
  "${EXTRA[@]}" "wandb.tags=[${TAGS}]" wandb.round=ghyper_sweep_2026_10_24 \
  wandb.label_knobs=true || rc=$?
echo "done $(date)  exit ${rc} (124 = stopped at STOP_HOURS: PARTIAL)"
exit "$rc"
