#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_expr_wave5.slurm
# [[experiments.019-simb-multimodal.scripts.gh_expr_wave5]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/gh_expr_wave5
#
# GilaHyper half of WAVE 5. Two stages, submitted SEPARATELY so the long-budget runs and the
# short paired runs do not share a GPU and contaminate each other's per-epoch timing:
#
#   # H1 -- is training saturated? 2 GPUs, 4 runs at 2/GPU, long cap.
#   sbatch --array=0-1%2 --export=ALL,W5_STAGE=h1 \
#       experiments/019-simb-multimodal/scripts/gh_expr_wave5.slurm
#
#   # Mask depth -- 2 GPUs, 12 runs at 2/GPU over 3 rounds.
#   sbatch --array=0-5%2 --export=ALL,W5_STAGE=maskdepth \
#       experiments/019-simb-multimodal/scripts/gh_expr_wave5.slurm
#
# WHY THE TWO STAGES GET DISJOINT GPUS. Packing changes per-epoch time (measured: 31 s/epoch
# alone versus 47 s/epoch at 2 runs/GPU pre-SDPA-fix). A 2000-epoch run co-resident with a
# rotating series of 400-epoch runs would see its neighbour change several times, so its
# trajectory would encode the scheduler rather than the model. Each stage is internally
# uniform: 2 runs per GPU, every run, start to finish.
#
# CONFIG IS cgt_expr_011 FOR BOTH: the partition is pinned (split_seed 0) so `seed` varies
# initialization only, and `traineval/*` gives the eval-mode train fit that H1 needs.
#
#SBATCH -p main
#SBATCH --mem=110g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=019-wave5
#SBATCH --time=23:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out

set -euo pipefail

mkdir -p /scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output

# EXPLICIT root, not SLURM_SUBMIT_DIR: submit-dir is wherever sbatch happened to be invoked,
# and the whole wave's validity rests on every task importing the same source.
PROJECT_ROOT="${W5_PROJECT_ROOT:-/home/michaelvolk/Documents/projects/torchcell.worktrees/019-decoder-pair-routing}"
ARM_SH="$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh"

STAGE="${W5_STAGE:?set W5_STAGE=h1|maskdepth via --export=ALL,W5_STAGE=...}"
export TORCHCELL_CONFIG=cgt_expr_011

# SEEDS ARE INIT-ONLY NOW. cgt_expr_011 pins data_module.split_seed=0, so these vary weight
# initialization against a FIXED validation draw. That is the whole point: the old scheme had
# one knob doing both, and the between-seed sd (0.0444) was 7.7x the across-arm sd (0.0058).
case "$STAGE" in
  # H1: 4 runs, 2000-epoch cap, eval-mode train metric every epoch. NO early stopping -- we
  # want the whole train curve, and ModelCheckpoint already retains best-on-val-loss so a
  # long cap costs nothing on the val side. At ~19 s/epoch plus the traineval pass this is
  # ~13 h; the 23 h walltime leaves margin without needing a requeue chain.
  h1)        ARMS=(H1_ref H1_nodrop); SEEDS=(0 1)
             EXTRA_OV="trainer.max_epochs=2000 trainer.train_eval_every=1" ;;
  # Mask depth: W_ref co-launched as the in-pool paired reference. 400 epochs brackets every
  # val-loss optimum observed in wave 4b (42-272) with margin.
  maskdepth) ARMS=(W_ref G_l3 G_l13); SEEDS=(0 1 2 3)
             EXTRA_OV="trainer.max_epochs=400 trainer.train_eval_every=5" ;;
  *) echo "unknown W5_STAGE '$STAGE' (expected h1|maskdepth)" >&2; exit 1 ;;
esac

JOBS=()
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    JOBS+=("$arm:$seed")
  done
done

RUNS_PER_GPU=2
TASK="${SLURM_ARRAY_TASK_ID:?this script must be submitted with --array}"
START=$((TASK * RUNS_PER_GPU))
if [[ "$START" -ge "${#JOBS[@]}" ]]; then
  echo "task $TASK starts past the end of stage '$STAGE' (${#JOBS[@]} runs)" >&2
  exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

# Split the CPU allocation across co-resident runs (reserve 2 for overhead). Giving each the
# full allocation oversubscribes the cores and makes the interference non-uniform across tasks.
NUM_WORKERS=$(((SLURM_CPUS_PER_TASK - 2) / RUNS_PER_GPU))

# Early stopping OFF for every wave-5 arm: it selects a different epoch per arm (wave 4b's
# val-loss argmins landed 42-272, and consistently later for the sink), so it puts
# epoch-selection into the comparison. Fixed budget + a pre-registered reducer instead.
export TORCHCELL_EXTRA_OVERRIDES="$EXTRA_OV trainer.early_stopping.enabled=false data_module.num_workers=$NUM_WORKERS"

OUT_DIR=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output

echo "=========================================================="
echo "host          : $(hostname)"
echo "job           : ${SLURM_ARRAY_JOB_ID:-none} task=${TASK}"
echo "stage         : $STAGE  (runs ${START}..$((START + RUNS_PER_GPU - 1)) of ${#JOBS[@]})"
echo "project root  : $PROJECT_ROOT"
echo "config        : $TORCHCELL_CONFIG"
echo "gpu           : ${CUDA_VISIBLE_DEVICES:-unset}  ($RUNS_PER_GPU runs co-resident)"
echo "overrides     : $TORCHCELL_EXTRA_OVERRIDES"
echo "=========================================================="

PIDS=()
LABELS=()
for k in $(seq 0 $((RUNS_PER_GPU - 1))); do
  IDX=$((START + k))
  [[ "$IDX" -ge "${#JOBS[@]}" ]] && continue
  JOB="${JOBS[$IDX]}"
  ARM="${JOB%%:*}"
  SEED="${JOB##*:}"
  LOG="$OUT_DIR/019-wave5_${SLURM_ARRAY_JOB_ID}_${TASK}_${ARM}_seed${SEED}.out"
  echo "  launching $ARM seed=$SEED -> $LOG"
  bash "$ARM_SH" "$ARM" "$SEED" > "$LOG" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$ARM:$SEED")
done

# Fail the task if EITHER run fails: a half-completed task leaves an arm missing a replicate,
# which breaks the pairing for that replicate.
STATUS=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  OK     ${LABELS[$i]}"
  else
    echo "  FAILED ${LABELS[$i]}" >&2
    STATUS=1
  fi
done
exit "$STATUS"
