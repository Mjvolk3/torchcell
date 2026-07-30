#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_expr_wave3.slurm
# [[experiments.019-simb-multimodal.scripts.gh_expr_wave3]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/gh_expr_wave3
#
# GilaHyper job array for WAVE 3 of the expression round. One task = one GPU running
# RUNS_PER_GPU=2 training processes concurrently; the %4 throttle keeps 4 tasks, so the
# box runs 8 models at once.
#
#   sbatch --array=0-3%4 --export=ALL,WAVE3_STAGE=ladder \
#       experiments/019-simb-multimodal/scripts/gh_expr_wave3.slurm
#   sbatch --array=0-9%4 --export=ALL,WAVE3_STAGE=arch \
#       experiments/019-simb-multimodal/scripts/gh_expr_wave3.slurm
#
#   (array size = ceil(runs / RUNS_PER_GPU): ladder 8 runs -> 0-3, arch 20 runs -> 0-9)
#
# WHY TWO PER GPU. A single run uses 20,026 of 46,068 MiB and does not saturate an Ada
# 6000 -- the encoder runs at BATCH 1 on the wildtype graph, so a lot of the step is small
# kernels. Measured: 31 s/epoch alone vs 47 s/epoch when two share a GPU, i.e. 23.5 s per
# run-epoch, ~1.45x the throughput. Two runs also fit in memory with ~6 GB spare.
#
# WHY WAVE 3 EXISTS. Everything before it was scored with a broken instrument:
# score_decoder_arms.py derived the arm from only `free_gene_dim` +
# `perturbation_propagation`, so C0/D1/D2/E0/F1/G1/H0/GEARS/S* all resolved to the literal
# string "A0_baseline" and were averaged INTO the baseline. On top of that a source edit at
# 2026-07-28 22:26:23 split the runs into two code versions that were then pooled, and the
# per-arm epoch budget was set by early stopping firing on a metric that swings +/-0.05
# epoch to epoch. Wave 3 re-measures from scratch on one frozen source, one budget, one
# packing, with a co-launched within-seed reference.
#
# STAGE ORDER IS NOT ARBITRARY. `ladder` runs FIRST and alone, because the model never fits
# its own TRAINING set: train/expression/pearson_per_feature ends at 0.19-0.246 and is still
# climbing at the cap, train/expression/loss falls 8% over 196 epochs, and train/grad_norm
# sits at 0.02-0.13 against a clip threshold of 10.0 so clipping never engages. Every
# architecture delta measured so far (<=0.03) was taken on a model explaining ~5% of train
# variance. Running the architecture arms before settling the optimizer would measure them
# all in that starved regime.
#
# PACKING IS UNIFORM AND MUST STAY THAT WAY. What the protocol requires is that packing be
# IDENTICAL for every run in a comparison set -- the paired reference is only a valid
# denominator if it ran under the same conditions as the arms subtracted from it. Two per
# GPU satisfies that as long as EVERY run in the stage is packed two-up. Do not mix.
#
# Sizing (node: 128 CPUs, 502 GB): 16 CPUs x 4 tasks = 64 of 128, and 110g x 4 = 440 of
# 502 GB reserved. Reserved is not used -- four concurrent runs measured ~114 GB resident
# in total, i.e. ~25 GB each -- but SLURM schedules on the reservation, so it has to fit.
#
#SBATCH -p main
#SBATCH --mem=110g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=019-wave3
#SBATCH --time=12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out

set -euo pipefail

mkdir -p /scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output

# Resolve the worktree root so the BRANCH's code runs, not the primary checkout's -- the
# editable install otherwise wins (memory: worktree-build-needs-pythonpath). Prefer an
# EXPLICIT root over SLURM_SUBMIT_DIR: submit-dir is whatever directory sbatch happened to
# be invoked from, and this wave's entire validity rests on every task importing the same
# source, so it is not something to leave to the caller's cwd.
PROJECT_ROOT="${WAVE3_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell}}"
ARM_SH="$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh"

STAGE="${WAVE3_STAGE:?set WAVE3_STAGE=ladder|arch via --export=ALL,WAVE3_STAGE=...}"

# Seeds are {0, 1, 2, 42}, NOT {0,1,2,3}: index_seed_{0,1,2,42}.json already exist under
# fig3_core/data_module_cache/, whereas seed 3 would have concurrent tasks racing to build
# and write the same split cache.
case "$STAGE" in
  ladder) ARMS=(L0_lr3e4 L1_lr1e3 L2_lr3e3 L3_lr1e3_cosine); SEEDS=(0 1) ;;
  arch)   ARMS=(A1_ref A2_prop_sparse A3_deep2 A4_nullsink A5_sham); SEEDS=(0 1 2 42) ;;
  # WAVE 4a -- the two arms that need NO new code, at FOUR seeds so this is a verdict on the
  # null-sink retry rather than a screen. Runs on cgt_expr_010, i.e. hard graph masking frozen.
  # The row-ablation and FiLM arms follow as wave 4b with their own co-launched reference: a
  # reference is only a valid denominator for arms launched alongside it under the same
  # conditions, so it is re-run per wave rather than borrowed across waves.
  wave4a) ARMS=(R_ref N_sink0); SEEDS=(0 1 2 42); export TORCHCELL_CONFIG=cgt_expr_010 ;;
  # WAVE 4b -- wave4a relaunched with the sink's magnitude confound removed and three defects
  # fixed in the masking path that wave4a ran with: the fused SDPA branch was gated on having
  # a mask (so 3 of 4 layers still materialized [1,9,6608,6608]), attention dropout silently
  # ran at p=0 on the masked layer versus p=0.1 elsewhere, and attention_mask.layers was never
  # validated against encoder depth. Because SDPA consumes RNG differently from the manual
  # softmax path, these runs are NOT comparable to wave3/wave4a at an identical seed -- hence
  # the separate stage tag and scorer wave boundary.
  wave4b) ARMS=(R_ref N_sink_mm); SEEDS=(0 1 2 42); export TORCHCELL_CONFIG=cgt_expr_010 ;;
  *) echo "unknown WAVE3_STAGE '$STAGE' (expected ladder|arch|wave4a)" >&2; exit 1 ;;
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

# num_workers from the SLURM CPU allocation, SPLIT across the co-resident runs (reserve 2
# for overhead). Giving each process the full allocation would oversubscribe the cores and
# make the two runs interfere in a way that is not uniform across tasks.
NUM_WORKERS=$(((SLURM_CPUS_PER_TASK - 2) / RUNS_PER_GPU))

# WAVE-WIDE overrides, applied identically to every arm. A budget that differs between arms
# is a confound: it scores "how long did the plateau look flat" alongside "how good is the
# mechanism". 300 epochs, not 200 -- measured ROLL5-argmax is 171 for one A0 seed-0
# replicate and 232 for H0_factor, so a 200 cap truncates arm-dependently.
export TORCHCELL_EXTRA_OVERRIDES="trainer.max_epochs=300 trainer.early_stopping.enabled=false data_module.num_workers=$NUM_WORKERS"

OUT_DIR=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output

echo "=========================================================="
echo "host          : $(hostname)"
echo "job           : ${SLURM_ARRAY_JOB_ID:-none} task=${TASK}"
echo "stage         : $STAGE  (runs ${START}..$((START + RUNS_PER_GPU - 1)) of ${#JOBS[@]})"
echo "project root  : $PROJECT_ROOT"
echo "source        : $(git -C "$PROJECT_ROOT" rev-parse --short HEAD) + working tree"
echo "gpu           : ${CUDA_VISIBLE_DEVICES:-unset}  ($RUNS_PER_GPU runs co-resident)"
echo "overrides     : $TORCHCELL_EXTRA_OVERRIDES"
echo "=========================================================="

# Both co-resident runs inherit this task's CUDA_VISIBLE_DEVICES, so they share the one
# allocated GPU. Each gets its OWN log -- interleaving two training streams into the array
# task's stdout would make both unreadable and break the epoch parsing the scorer relies on.
PIDS=()
LABELS=()
for k in $(seq 0 $((RUNS_PER_GPU - 1))); do
  IDX=$((START + k))
  [[ "$IDX" -ge "${#JOBS[@]}" ]] && continue
  JOB="${JOBS[$IDX]}"
  ARM="${JOB%%:*}"
  SEED="${JOB##*:}"
  LOG="$OUT_DIR/019-wave3_${SLURM_ARRAY_JOB_ID}_${TASK}_${ARM}_seed${SEED}.out"
  echo "  launching $ARM seed=$SEED -> $LOG"
  bash "$ARM_SH" "$ARM" "$SEED" > "$LOG" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$ARM:$SEED")
done

# Fail the task if EITHER run fails -- a silently half-completed task would leave the wave
# with an arm missing a seed, which breaks the within-seed pairing for that arm.
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
