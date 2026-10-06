#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_expr_v22_fast.slurm
# [[experiments.019-simb-multimodal.scripts.gh_expr_v22_fast]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/gh_expr_v22_fast
#
# v22, THE FAST-MODEL ROUND (2026-10-06, GilaHyper, four RTX 6000 Ada). The v21 single-head
# expression model on the fast code path of the same night (batched perturbation operator,
# splits held in memory), one ARM per card (see the case block for why). Every arm is the
# v21 reference (cgt_expr_v21_small, arm S_ref) with one change:
#
#   wave 1
#     F_ref       nothing: batch 32, six layers, width 90, lr 3e-4. Checks the fast code
#                 against v21's S_ref on Delta (same config, same split seeds, old code).
#     F_b128      batch 128, lr 6e-4  (square-root scaling of the learning rate)
#     F_b128lr4   batch 128, lr 1.2e-3 (linear scaling)
#     F_l4w180    four layers, width 180
#
# WHY THESE. The profile (gh_profile_cgt_expr.slurm, jobs 3302 to 3313) put about 70 percent
# of a step's GPU time in the encoder's self-attention over all 6,607 genes, which runs once
# per STEP whatever the batch, so batch 128 takes 9 encoder passes an epoch against 35 and
# the model step falls from 8.2 to 5.6 s an epoch on an unshared card. Whether batch 128
# LEARNS the same is what F_b128 and F_b128lr4 measure; no batch other than 32 has been
# scored on this task. F_l4w180 is the wide, shallower trunk: no width above 90 has run
# past 151 epochs (round_leaderboards.csv).
#
# SCORING, as v21 (pre-registered there): mean of val/expression/pearson_per_feature over
# epochs 1,000 to 1,200, paired by split seed against the reference (F_ref here on split
# seeds 0 to 2, v21's S_ref on Delta for all twelve). Two to four split seeds cannot reach
# p < 0.05 on a sign test; this round sizes effects and throughput, it does not decide.
#
#   sbatch --array=0-3 --export=ALL,V22_PROJECT_ROOT=$PWD \
#       experiments/019-simb-multimodal/scripts/gh_expr_v22_fast.slurm
#   V22_WAVE=2 selects the second arm set (see the case block).
#
#SBATCH -p main
#SBATCH --mem=60000M
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=019-v22w1-fast
#SBATCH --time=08:45:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out

set -euo pipefail

OUT_DIR=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output
mkdir -p "$OUT_DIR"

PROJECT_ROOT="${V22_PROJECT_ROOT:?set V22_PROJECT_ROOT to the checkout to run}"
ARM_SH="$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh"
TASK="${SLURM_ARRAY_TASK_ID:?this script must be submitted with --array (the pack index)}"
WAVE="${V22_WAVE:-1}"
EPOCHS="${V22_EPOCHS:-1200}"

# arm name -> overrides against the v21 reference
declare -A ARM_OVERRIDES
ARM_OVERRIDES[F_ref]=""
ARM_OVERRIDES[F_b128]="data_module.batch_size=128 regression_task.optimizer.lr=6.0e-4"
ARM_OVERRIDES[F_b128lr4]="data_module.batch_size=128 regression_task.optimizer.lr=1.2e-3"
ARM_OVERRIDES[F_l4w180]="model.num_transformer_layers=4 model.hidden_channels=180"

# ONE ARM PER CARD. The first submission (job 3315, cancelled ten minutes in) put the four
# arms on each card, one card per split seed. The four processes time-slice the card, so a
# batch-128 run's nine heavy steps took as long as a batch-32 run's 35 light ones (19 to 22
# s per epoch for both) and nothing would have reached 1,200 epochs inside the window. A
# card of batch-128 runs alone is the configuration whose speed the profile measured.
# Batch-32 arms run two or three to a card for the same reason. F_ref's split seeds 0 to 11
# also exist as v21's S_ref on Delta (same config and seeds, loop operator).
case "$WAVE:$TASK" in
  1:0) ARM=F_b128;    SPLITS=(0 1 2 3) ;;
  1:1) ARM=F_b128lr4; SPLITS=(0 1 2 3) ;;
  1:2) ARM=F_ref;     SPLITS=(0 1 2) ;;
  1:3) ARM=F_l4w180;  SPLITS=(0 1) ;;
  *) echo "unknown wave:task $WAVE:$TASK" >&2; exit 1 ;;
esac

COMMON="trainer.max_epochs=$EPOCHS trainer.early_stopping.enabled=false +data_module.materialize=true data_module.num_workers=2 data_module.persistent_workers=true wandb.project=torchcell_019_expr_v22"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

export TORCHCELL_CONFIG=cgt_expr_v21_small
export PYTHONPATH="$PROJECT_ROOT"

echo "=========================================================="
echo "host          : $(hostname)"
echo "job           : ${SLURM_ARRAY_JOB_ID:-none} task=${TASK} wave=${WAVE}"
echo "project root  : $PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD))"
echo "config        : $TORCHCELL_CONFIG"
echo "gpu           : ${CUDA_VISIBLE_DEVICES:-unset}  arm: $ARM  split seeds: ${SPLITS[*]}"
echo "common        : $COMMON"
echo "=========================================================="

( while true; do
    printf '%s  ' "$(date +%T)"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    sleep 120
  done ) &
SAMPLER=$!

PIDS=(); LABELS=()
for SPLIT in "${SPLITS[@]}"; do
  LOG="$OUT_DIR/019-v22_${SLURM_ARRAY_JOB_ID}_${ARM}_s${SPLIT}.out"
  echo "  launching $ARM split $SPLIT -> $LOG   [${ARM_OVERRIDES[$ARM]}]"
  TORCHCELL_EXTRA_OVERRIDES="$COMMON ${ARM_OVERRIDES[$ARM]} wandb.tags=[v22,$ARM,split$SPLIT]" \
    bash "$ARM_SH" "S_ref_s${SPLIT}" 0 > "$LOG" 2>&1 &
  PIDS+=("$!"); LABELS+=("${ARM}_s${SPLIT}")
  sleep 30
done

STATUS=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "  OK     ${LABELS[$i]}"; else echo "  FAILED ${LABELS[$i]}"; STATUS=1; fi
done
kill "$SAMPLER" 2>/dev/null || true
exit "$STATUS"
