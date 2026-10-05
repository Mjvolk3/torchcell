#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_small_model_bench.slurm
# [[experiments.019-simb-multimodal.scripts.gh_small_model_bench]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/gh_small_model_bench
#
# THROUGHPUT BENCHMARK OF SCALED-DOWN TRUNKS (2026-10-05), for sizing single-GPU Delta jobs.
# Four cells, one per GilaHyper RTX 6000 Ada, each a pack of the v19 arms (K_prot, K_expr,
# K_joint on the both-label store, config cgt_expr_v19_joint_clean) under one model shape:
#
#   cell 0  the v19 model as run: four-part 3,328-d stack, width 90, six layers, 4 runs/card
#   cell 1  ProtT5 alone (1,024-d), width 90, six layers, 4 runs/card
#   cell 2  ProtT5 alone, width 90, three layers, 6 runs/card
#   cell 3  ProtT5 alone, width 54, two layers, 8 runs/card
#
# What it measures: perf/epoch_seconds per run (W&B project torchcell_019_small_bench) and
# card memory sampled every minute into the task log. What it cannot measure in a three-hour
# window: convergence. The 2026-10-04 audit found wall time independent of parameter count
# between 0.68 and 1.45 million parameters and a fifth of each epoch in the eval-mode train
# pass; depth and packing are the two levers it did not time, and those are the axes here.
#
# ZERO DATA-LOADER WORKERS IN EVERY CELL, deliberately. Delta runs at num_workers=0 (spawned
# workers re-import torch from Lustre; delta_grid_common.sh), so a pack sized on GilaHyper
# numbers taken with workers would be sized wrong. GilaHyper's NVMe still makes these
# optimistic for Lustre; the A40 is also slower than this card (hypothesis, untimed: about
# 1.5 to 1.8x per epoch). Both corrections go on top of what this logs.
#
#   sbatch --array=0-3 --export=ALL,W5_PROJECT_ROOT=$PWD \
#       experiments/019-simb-multimodal/scripts/gh_small_model_bench.slurm
#
#SBATCH -p main
#SBATCH --mem=60000M
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=019-smallbench
#SBATCH --time=01:45:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out

set -euo pipefail

OUT_DIR=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output
mkdir -p "$OUT_DIR"

PROJECT_ROOT="${W5_PROJECT_ROOT:?set W5_PROJECT_ROOT to the checkout being benchmarked}"
if [ ! -d "$PROJECT_ROOT/torchcell" ]; then
  echo "FATAL: PROJECT_ROOT=$PROJECT_ROOT has no torchcell/ package dir." >&2
  exit 1
fi
ARM_SH="$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh"

TASK="${SLURM_ARRAY_TASK_ID:?this script must be submitted with --array}"
EPOCHS="${BENCH_EPOCHS:-30}"

# Every cell: v19 store and arms, 30 epochs, eval-mode train pass every 10 as in v19,
# early stopping off, zero workers, its own W&B project so nothing lands in v19.
COMMON="trainer.max_epochs=$EPOCHS trainer.early_stopping.enabled=false data_module.num_workers=0 wandb.project=torchcell_019_small_bench"
case "$TASK" in
  0) SHAPE="" ; RUNS=(K_prot_s0 K_expr_s0 K_joint_s0 K_joint_s1); CELL=full_w90_l6_p4 ;;
  1) SHAPE="cell_dataset.node_embeddings=[prot_T5_all]"
     RUNS=(K_prot_s0 K_expr_s0 K_joint_s0 K_joint_s1); CELL=t5_w90_l6_p4 ;;
  2) SHAPE="cell_dataset.node_embeddings=[prot_T5_all] model.num_transformer_layers=3"
     RUNS=(K_prot_s0 K_expr_s0 K_joint_s0 K_prot_s1 K_expr_s1 K_joint_s1); CELL=t5_w90_l3_p6 ;;
  3) SHAPE="cell_dataset.node_embeddings=[prot_T5_all] model.num_transformer_layers=2 model.hidden_channels=54"
     RUNS=(K_prot_s0 K_expr_s0 K_joint_s0 K_prot_s1 K_expr_s1 K_joint_s1 K_prot_s2 K_expr_s2); CELL=t5_w54_l2_p8 ;;
  # SECOND WAVE (same day, after cells 0 to 3 read out at epoch 1): every cell above cost
  # about 146 s per epoch at zero workers whether 4 or 8 runs shared the card and whether the
  # trunk had two or six layers, and card memory was 8 to 10 GB per run at every shape. So the
  # main process, not the GPU, bounds the epoch, and the lever to time is the data-loader
  # worker count with PERSISTENT workers (the spawn import is then paid once, not per epoch,
  # which is the Delta objection to workers). Cells 4 and 5 time that on the ProtT5-only
  # six-layer trunk, the shape the v21 round adopts.
  4) SHAPE="cell_dataset.node_embeddings=[prot_T5_all]"
     RUNS=(K_prot_s0 K_expr_s0 K_joint_s0 K_joint_s1); CELL=t5_w90_l6_p4_w3; WORKERS=3 ;;
  5) SHAPE="cell_dataset.node_embeddings=[prot_T5_all]"
     RUNS=(K_prot_s0 K_expr_s0 K_joint_s0 K_prot_s1 K_expr_s1); CELL=t5_w90_l6_p5_w2; WORKERS=2 ;;
  # Cells 4 and 5 read 35 to 40 s per epoch against 69 to 75 at zero workers (five per card
  # ran out of memory, confirming four as the ceiling). Cell 6 is Delta's shape: one A40's
  # share of a gpuA40x4 node is 8 CPUs, so four runs get one worker each. Submitted with
  # --cpus-per-task=8 so the CPU budget matches.
  6) SHAPE="cell_dataset.node_embeddings=[prot_T5_all]"
     RUNS=(K_prot_s0 K_expr_s0 K_joint_s0 K_joint_s1); CELL=t5_w90_l6_p4_w1; WORKERS=1 ;;
  *) echo "unknown task $TASK (0..6)" >&2; exit 1 ;;
esac
if [[ -n "${WORKERS:-}" ]]; then
  COMMON="${COMMON/data_module.num_workers=0/data_module.num_workers=$WORKERS data_module.persistent_workers=true}"
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

export TORCHCELL_CONFIG=cgt_expr_v19_joint_clean
export TORCHCELL_EXTRA_OVERRIDES="$COMMON $SHAPE wandb.tags=[small-bench,$CELL]"
export PYTHONPATH="$PROJECT_ROOT"

echo "=========================================================="
echo "host          : $(hostname)"
echo "job           : ${SLURM_ARRAY_JOB_ID:-none} task=${TASK} cell=$CELL"
echo "project root  : $PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD))"
echo "config        : $TORCHCELL_CONFIG"
echo "gpu           : ${CUDA_VISIBLE_DEVICES:-unset}  (${#RUNS[@]} runs co-resident)"
echo "overrides     : $TORCHCELL_EXTRA_OVERRIDES"
echo "=========================================================="

# Card memory and utilization once a minute, so the VRAM per pack is measured, not inferred.
( while true; do
    printf '%s  ' "$(date +%T)"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    sleep 60
  done ) &
SAMPLER=$!

PIDS=(); LABELS=()
for ARM in "${RUNS[@]}"; do
  LOG="$OUT_DIR/019-smallbench_${SLURM_ARRAY_JOB_ID}_${TASK}_${CELL}_${ARM}.out"
  echo "  launching $ARM seed=0 -> $LOG"
  bash "$ARM_SH" "$ARM" 0 > "$LOG" 2>&1 &
  PIDS+=("$!"); LABELS+=("$ARM")
done

STATUS=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "  OK     ${LABELS[$i]}"; else echo "  FAILED ${LABELS[$i]}"; STATUS=1; fi
done
kill "$SAMPLER" 2>/dev/null || true
exit "$STATUS"
