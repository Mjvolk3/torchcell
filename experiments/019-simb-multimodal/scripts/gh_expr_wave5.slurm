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

STAGE="${W5_STAGE:?set W5_STAGE=h1|maskdepth|h2gh via --export=ALL,W5_STAGE=...}"
export TORCHCELL_CONFIG=cgt_expr_011

# SEEDS ARE INIT-ONLY NOW. cgt_expr_011 pins data_module.split_seed=0, so these vary weight
# initialization against a FIXED validation draw. That is the whole point: the old scheme had
# one knob doing both, and the between-seed sd (0.0444) was 7.7x the across-arm sd (0.0058).
case "$STAGE" in
  # H1: 4 runs, 2000-epoch cap, NO early stopping -- we want the whole train curve, and
  # ModelCheckpoint already retains best-on-val-loss so a long cap costs nothing on the val side.
  #
  # train_eval_every=5, NOT 1, and the reason is measured. The first submission (job 1417) ran
  # every=1 and clocked 85 s/epoch -- reaching only epoch 17 in 24:04 -- against 36 s/epoch for
  # the every=5 maskdepth stage in the same window. So the eval-mode train pass costs ~65 s on
  # its own, which is unsurprising in hindsight: it sweeps the 1244-strain TRAIN split against
  # validation's 155, i.e. ~8x the rows. At 85 s/epoch the 23 h walltime truncates at ~970
  # epochs, short of the cap; at ~36 s/epoch the full 2000 fit with margin. Sampling the
  # saturation curve every 5 epochs still yields 400 points, which is far more than a
  # "has it stopped moving" question needs.
  h1)        ARMS=(H1_ref H1_nodrop); SEEDS=(0 1)
             EXTRA_OV="trainer.max_epochs=2000 trainer.train_eval_every=5" ;;
  # Mask depth: W_ref co-launched as the in-pool paired reference. 400 epochs brackets every
  # val-loss optimum observed in wave 4b (42-272) with margin.
  maskdepth) ARMS=(W_ref G_l3 G_l13); SEEDS=(0 1 2 3)
             EXTRA_OV="trainer.max_epochs=400 trainer.train_eval_every=5" ;;
  # H2 REPLICATED ON GILAHYPER HARDWARE. The powered H2 test runs on mmli A100s; this is the
  # SAME three arms at 4 replicates on RTX 6000 Ada with its own co-launched W_ref. It is not
  # pooled with the mmli runs -- different GPU type, different packing, so a cross-pool
  # comparison would be confounded (_007 pooled A100 with Ada and their means differ, 0.0242
  # vs 0.0182). Its value is INDEPENDENT REPLICATION: a mechanism that shows the same sign on
  # two hardware pools with separate references is a far stronger claim than one pool at n=8.
  # 300 epochs to match the mmli budget, so the two are at least reduced the same way.
  h2gh)      ARMS=(W_ref X_mix X_mix_rezero); SEEDS=(0 1 2 3)
             EXTRA_OV="trainer.max_epochs=300 trainer.train_eval_every=5" ;;
  # THROUGHPUT BENCH -- the launch blocker, not science. Wave 6 targets 5,000 epochs in
  # 24 h = 17.3 s/epoch, and the best rate measured so far is 32 s/epoch (2 runs/GPU,
  # 7 workers); 3 workers gave 270 s/epoch with the A100 at 0% util, i.e. dataloader-starved.
  #
  # The lever being tested is BATCH SIZE. The encoder runs on the WILDTYPE graph at batch 1,
  # so its cost C_enc is paid once per STEP regardless of B, and per-epoch cost is
  # ceil(n/B) * (C_enc + B*C_pert). At the current B=8 that is ~155 steps to process 1,244
  # strains -- the dominant cost paid 155 times. B=32 should cut the C_enc term ~4x.
  #
  # Each task is ONE (batch, packing) cell, 60 epochs, and the answer is read from
  # `perf/epoch_seconds` (median over the last 40 epochs) rather than job elapsed, so
  # dataset load and startup are excluded.
  #
  #   sbatch --array=0-3%4 --export=ALL,W5_STAGE=bench<N> ... one cell per task, see below.
  bench_b8_p2)  ARMS=(W_ref W_ref); SEEDS=(0); STAGE_RUNS_PER_GPU=2
                EXTRA_OV="trainer.max_epochs=60 trainer.train_eval_every=0 data_module.batch_size=8" ;;
  bench_b32_p2) ARMS=(W_ref W_ref); SEEDS=(0); STAGE_RUNS_PER_GPU=2
                EXTRA_OV="trainer.max_epochs=60 trainer.train_eval_every=0 data_module.batch_size=32" ;;
  bench_b8_p3)  ARMS=(W_ref W_ref W_ref); SEEDS=(0); STAGE_RUNS_PER_GPU=3
                EXTRA_OV="trainer.max_epochs=60 trainer.train_eval_every=0 data_module.batch_size=8" ;;
  bench_b32_p3) ARMS=(W_ref W_ref W_ref); SEEDS=(0); STAGE_RUNS_PER_GPU=3
                EXTRA_OV="trainer.max_epochs=60 trainer.train_eval_every=0 data_module.batch_size=32" ;;
  *) echo "unknown W5_STAGE '$STAGE' (expected h1|maskdepth|h2gh|bench_*)" >&2; exit 1 ;;
esac

JOBS=()
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    JOBS+=("$arm:$seed")
  done
done

RUNS_PER_GPU="${STAGE_RUNS_PER_GPU:-2}"
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
