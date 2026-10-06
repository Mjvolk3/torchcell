#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_profile_cgt_expr.slurm
# [[experiments.019-simb-multimodal.scripts.gh_profile_cgt_expr]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/gh_profile_cgt_expr
#
# WHERE DOES AN EPOCH GO (2026-10-06). One v21 reference run per card under Lightning's
# simple profiler, four epochs with the eval-mode train pass every epoch, so the per-hook
# table at the end of the log splits the epoch into dataloader fetch, training_step,
# backward, optimizer step and the validation hooks. Cells vary the two things the
# 2026-10-05 benchmark could not separate: data-loader workers and batch size.
#
#   cell 0  batch 32, zero workers        (the Delta setting of July)
#   cell 1  batch 32, three persistent workers
#   cell 2  batch 64, three persistent workers
#   cell 3  batch 128, three persistent workers
#
# The learning-rate is NOT rescaled with the batch here: this job times the step, it does
# not score it. perf/epoch_seconds and the profiler table are the readouts; W&B project
# torchcell_019_profile.
#
#   sbatch --array=0-3 --export=ALL,W5_PROJECT_ROOT=$PWD \
#       experiments/019-simb-multimodal/scripts/gh_profile_cgt_expr.slurm
#
#SBATCH -p main
#SBATCH --mem=60000M
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=019-profile
#SBATCH --time=00:40:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out

set -euo pipefail

OUT_DIR=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output
mkdir -p "$OUT_DIR"

PROJECT_ROOT="${W5_PROJECT_ROOT:?set W5_PROJECT_ROOT to the checkout being profiled}"
ARM_SH="$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh"
TASK="${SLURM_ARRAY_TASK_ID:?this script must be submitted with --array}"
EPOCHS="${PROFILE_EPOCHS:-4}"

PACK=1
case "$TASK" in
  0) BATCH=32;  WORKERS=0 ;;
  1) BATCH=32;  WORKERS=3 ;;
  2) BATCH=64;  WORKERS=3 ;;
  3) BATCH=128; WORKERS=3 ;;
  # Packed cells: four runs share the card (the production packing), so the per-run
  # epoch time here is the one a round pays. Submit with --cpus-per-task=16.
  4) BATCH=32;  WORKERS=3; PACK=4 ;;
  5) BATCH=64;  WORKERS=3; PACK=4 ;;
  6) BATCH=128; WORKERS=3; PACK=4 ;;
  # The original per-strain operator loop, for the before/after of the batched form.
  7) BATCH=32;  WORKERS=3; OPERATOR=loop ;;
  8) BATCH=32;  WORKERS=3; PACK=4; OPERATOR=loop ;;
  # The kernel-level view: Lightning's PyTorch profiler over the training step, two
  # epochs, so the per-op table says which CUDA kernels the 0.2 s step is made of.
  9) BATCH=32;  WORKERS=3; PROFILER=pytorch; EPOCHS=2 ;;
  # Materialized splits (`+data_module.materialize=true`): every sample fetched once at
  # setup, no loader work in the epoch, zero workers in training.
  10) BATCH=32;  WORKERS=0; MATERIALIZE=1 ;;
  11) BATCH=32;  WORKERS=0; MATERIALIZE=1; PACK=4 ;;
  12) BATCH=64;  WORKERS=0; MATERIALIZE=1; PACK=4 ;;
  # NEXT-SPEED CELLS (2026-10-06 morning), on the fast code, 40 epochs with the eval-mode
  # train pass at the round's interval of 10, read from W&B perf/epoch_seconds. After the
  # fixes the cost is the operator's feed-forward block over batch x 6,607 gene tokens:
  # cell 13 narrows it from 4x to 1x the width, cell 14 is its control under the same
  # packing, cell 15 asks what batch 256 costs in time and card memory.
  13) BATCH=128; WORKERS=2; MATERIALIZE=1; PACK=3; EPOCHS=40; TEVAL=10; SHAPE="model.perturbation_head.ffn_mult=1" ;;
  14) BATCH=128; WORKERS=2; MATERIALIZE=1; PACK=3; EPOCHS=40; TEVAL=10 ;;
  15) BATCH=256; WORKERS=2; MATERIALIZE=1; PACK=1; EPOCHS=40; TEVAL=10 ;;
  *) echo "unknown task $TASK (0..15)" >&2; exit 1 ;;
esac
export TORCHCELL_PERT_OPERATOR="${OPERATOR:-batched}"
PROFILER="${PROFILER:-simple}"
CELL="profile_b${BATCH}_w${WORKERS}_p${PACK}_${TORCHCELL_PERT_OPERATOR}_${PROFILER}${MATERIALIZE:+_mat}"

# `+trainer.profiler`: the key is not in the config, so hydra needs the append form.
COMMON="trainer.max_epochs=$EPOCHS trainer.train_eval_every=${TEVAL:-1} trainer.early_stopping.enabled=false +trainer.profiler=$PROFILER data_module.batch_size=$BATCH data_module.num_workers=$WORKERS wandb.project=torchcell_019_profile ${SHAPE:-}"
if [[ -n "${MATERIALIZE:-}" ]]; then
  COMMON="$COMMON +data_module.materialize=true"
fi
if [[ "$WORKERS" -gt 0 ]]; then
  COMMON="$COMMON data_module.persistent_workers=true"
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

export TORCHCELL_CONFIG=cgt_expr_v21_small
export TORCHCELL_EXTRA_OVERRIDES="$COMMON wandb.tags=[profile,$CELL]"
export PYTHONPATH="$PROJECT_ROOT"

echo "=========================================================="
echo "host          : $(hostname)"
echo "job           : ${SLURM_ARRAY_JOB_ID:-none} task=${TASK} cell=$CELL"
echo "project root  : $PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD))"
echo "config        : $TORCHCELL_CONFIG"
echo "gpu           : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "overrides     : $TORCHCELL_EXTRA_OVERRIDES"
echo "=========================================================="

( while true; do
    printf '%s  ' "$(date +%T)"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    sleep 30
  done ) &
SAMPLER=$!

PIDS=()
for ((k = 0; k < PACK; k++)); do
  LOG="$OUT_DIR/019-profile_${SLURM_ARRAY_JOB_ID}_${TASK}_${CELL}_r${k}.out"
  echo "  launching S_ref_s0 copy $k -> $LOG"
  bash "$ARM_SH" S_ref_s0 0 > "$LOG" 2>&1 &
  PIDS+=("$!")
  sleep 20
done
STATUS=0
for p in "${PIDS[@]}"; do wait "$p" || STATUS=1; done
kill "$SAMPLER" 2>/dev/null || true
echo "  exit $STATUS"
exit "$STATUS"
