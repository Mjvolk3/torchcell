#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_train_043_overnight_v2.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_train_043_overnight_v2]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_train_043_overnight_v2.slurm
#
# One GPU cell of the 043 trainer on STORE v2 (raw_043_v2 -> prep_v3, global_split_v2):
# the gh_train_043_smoke.slurm launcher with data.store_version=v2 passed on the command
# line (the yaml default flip is the store-v2 commit's, not this script's) and the W&B
# tags of the 2026-10-23 overnight round. Every mark (MARK_CAP null), plain residual.
#
# Environment knobs (all optional): ARM, LOSS, WEIGHTING, SEED, SAMPLER, MAX_EPOCHS,
# MAX_STEPS (step-matched alone runs), INIT_FROM (stage 2: a stage-1 last.pt), LR,
# WARMUP_STEPS, EVAL_EVERY, TIME_BUDGET_HOURS, DEADLINE (epoch seconds; the trainer's
# budget becomes min(TIME_BUDGET_HOURS, hours left)), RESERVE_HOURS (hours held back from
# DEADLINE for a successor on the same card), TAG_SET (smoke | overnight),
# WINDOW_START_FRAC (registered window start; 0.8 default), FIXED_EPOCHS=1 (no trainer
# epoch cut: the run keeps MAX_EPOCHS, so two seeds share one epoch count, and a
# `timeout` at DEADLINE - RESERVE_HOURS stops it instead; a stopped run is PARTIAL and its
# last.pt is the last finished epoch).
# --export cannot carry a comma inside one value, so tags are chosen by TAG_SET.
#
# Round overnight_2026_10_23 (owner grant 2026-10-23: four GilaHyper cards overnight, every
# job ends by 07:30 CDT 2026-10-24 = DEADLINE 1792845000). The submitted lines, the
# arithmetic and the job ids are in the experiment note's dated section.
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=043-v2-train
#SBATCH --time=01:00:00
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
ARM="${ARM:-all}"
LOSS="${LOSS:-hlgauss}"
WEIGHTING="${WEIGHTING:-equal}"
SEED="${SEED:-0}"
SAMPLER="${SAMPLER:-records}"
TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-0.9}"
RESERVE_HOURS="${RESERVE_HOURS:-0}"
if [ -n "${DEADLINE:-}" ]; then
  LEFT=$(python -c "import time; print(round(max(0.1, (${DEADLINE} - time.time()) / 3600 - ${RESERVE_HOURS}), 2))")
  TIME_BUDGET_HOURS=$(python -c "print(min(${TIME_BUDGET_HOURS}, ${LEFT}))")
fi
EVAL_EVERY="${EVAL_EVERY:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-2}"
TIME_MARGIN_MINUTES="${TIME_MARGIN_MINUTES:-10}"
MARK_CAP="${MARK_CAP:-null}"
case "${TAG_SET:-overnight}" in
  smoke) TAGS="v2_store,smoke" ;;
  overnight) TAGS="v2_store,overnight_2026_10_23" ;;
  *) echo "unknown TAG_SET ${TAG_SET}" >&2; exit 2 ;;
esac
EXTRA=()
if [ -n "${WINDOW_START_FRAC:-}" ]; then EXTRA+=("budget.registered_window.start_frac=${WINDOW_START_FRAC}"); fi
STOP=()
if [ "${FIXED_EPOCHS:-0}" = "1" ]; then
  : "${DEADLINE:?FIXED_EPOCHS needs DEADLINE}"
  TIME_BUDGET_HOURS=null
  STOP=(timeout --signal=TERM --kill-after=120 "$(python -c "import time; print(max(60, int(${DEADLINE} - time.time() - 3600 * ${RESERVE_HOURS})))")")
fi
if [ -n "${MAX_STEPS:-}" ]; then EXTRA+=("budget.max_steps=${MAX_STEPS}"); fi
if [ -n "${INIT_FROM:-}" ]; then EXTRA+=("init_from=${INIT_FROM}"); fi
if [ -n "${LR:-}" ]; then EXTRA+=("optim.lr=${LR}"); fi
if [ -n "${WARMUP_STEPS:-}" ]; then EXTRA+=("optim.warmup_steps=${WARMUP_STEPS}"); fi
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  $(date)"
echo "fixed_epochs ${FIXED_EPOCHS:-0}  stop ${STOP[*]:-none}  window_start ${WINDOW_START_FRAC:-0.8}"
echo "store v2  arm $ARM  loss $LOSS  weighting $WEIGHTING  sampler $SAMPLER  seed $SEED  mark_cap $MARK_CAP  max_epochs $MAX_EPOCHS  max_steps ${MAX_STEPS:-none}  init_from ${INIT_FROM:-none}  lr ${LR:-default}  warmup ${WARMUP_STEPS:-default}  eval_every $EVAL_EVERY  budget ${TIME_BUDGET_HOURS} h  tags $TAGS"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
rc=0
"${STOP[@]}" python experiments/043-expression-proteome-morphology/scripts/train_043.py \
  data.store_version=v2 \
  arm="$ARM" loss="$LOSS" weighting="$WEIGHTING" seed="$SEED" \
  data.batch_size=8 data.num_workers=4 data.sampler="$SAMPLER" data.mark_cap="$MARK_CAP" \
  budget.max_epochs="$MAX_EPOCHS" budget.time_budget_hours="$TIME_BUDGET_HOURS" budget.time_margin_minutes="$TIME_MARGIN_MINUTES" \
  budget.eval_every_epochs="$EVAL_EVERY" "${EXTRA[@]}" \
  "wandb.tags=[${TAGS}]" || rc=$?
echo "done $(date)  exit ${rc} (124 = stopped at the deadline: PARTIAL)"
exit "$rc"
