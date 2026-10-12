#!/bin/bash
# experiments/019-simb-multimodal/scripts/delta_morph_v25_small.slurm
# [[experiments.019-simb-multimodal.scripts.delta_morph_v25_small]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/delta_morph_v25_small
#
# THE INDUCTIVE MORPHOLOGY ROUND v25 ON DELTA (2026-10-12): the dimensions cabbi is not
# running, on the bbub-delta-gpu account (the 019 project's own, idle on 2026-10-12). ONE
# A40 per job, every job an independent submission, each a pack of PACK runs sharing the
# card (delta_expr_v21_small.slurm holds the reasons: single-card jobs backfill, packs make
# the budget). conf/cgt_morph_v24.yaml carries the design; the arms live in
# gh_expr_008_arm.sh; the recipe of a run comes in through RUN_OV, per run.
#
# STAGE morph_v25d (24 runs, 8 packs of 3): on the two readouts that escaped the plateau on
# cabbi at lr 3e-4 (M_pool, the CLS plus the gene pool; M_zs, the CLS plus the deleted
# gene's token), the batch size (32 and 64 against the 128 of every cabbi cell: four and
# two times the optimizer steps per epoch) and the initialization seed (1 and 2 at batch
# 128, the escape rate of a readout that escapes), split seeds 0 to 2, the seeds whose
# split caches the Delta fig3_core store carries (a fourth would be written by the first
# run that needs it, and two packs of one split would race for it).
#
# ONE-TIME SETUP on Delta (the user, Duo):
#   mkdir -p /work/hdd/bbub/mjvolk3/slurm-logs/019-morph-v25
#   git -C /scratch/bbub/mjvolk3/torchcell-v11 pull      # job-owned clone of this branch
#   # the store is already there: /scratch/bbub/mjvolk3/torchcell/.../fig3_core
#
# SUBMIT, one sbatch per pack, from the clone:
#   R=/scratch/bbub/mjvolk3/torchcell-v11
#   H=$(git -C $R rev-parse HEAD); D=$(git -C $R diff HEAD | sha256sum | cut -d" " -f1)
#   for k in $(seq 0 7); do
#     sbatch --array=$k --account=bbub-delta-gpu -J 019-morph2w4-delta-p$k \
#       --export=ALL,V25_STAGE=morph_v25d,V25_PROJECT_ROOT=$R,TORCHCELL_SOURCE_GIT_HASH=$H,TORCHCELL_SOURCE_DIFF_SHA256=$D \
#       $R/experiments/019-simb-multimodal/scripts/delta_morph_v25_small.slurm
#   done
# N_PACKS is printed by `V25_STAGE=morph_v25d V25_LIST=1 bash <this script>` (no sbatch).
# BUDGET: V25_EPOCHS, default 500 (2026-10-12): the loss finishes learning by epoch 50, an
# escape happens by epoch 400 and the validation peak of an escaped run sits at epochs 400
# to 600, so the registered window of v25 is epochs 400 to 499. Materializing the splits
# costs about 1.8 h per run on Lustre before epoch 1 (measured on pack 0, 2026-10-12).
#
#SBATCH --job-name=019-morph-v25
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbub-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60g
#SBATCH --time=1-12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/hdd/bbub/mjvolk3/slurm-logs/019-morph-v25/%x_%A_%a.out
#SBATCH --error=/work/hdd/bbub/mjvolk3/slurm-logs/019-morph-v25/%x_%A_%a.out

set -euo pipefail

PROJECT_ROOT="${V25_PROJECT_ROOT:?set V25_PROJECT_ROOT to the job-owned clone}"
OUT_DIR="${V25_LOG_DIR:-/work/hdd/bbub/mjvolk3/slurm-logs/019-morph-v25}"
STAGE="${V25_STAGE:?set V25_STAGE=canary|morph_v25d}"
STAGE_CONFIG=cgt_morph_v24
# PACK: runs sharing the one card. Three: a batch-128 morphology run holds about 12 GB
# (v22 measured 12.2 GB on the expression trunk at this batch), so three fit the A40's
# 44.4 GiB and four would not.
PACK="${V25_PACK:-3}"
# One persistent worker per run (delta_expr_v21_small.slurm, measured 2026-10-05).
WORKERS="${V25_WORKERS:-1}"

BASE_OV="trainer.max_epochs=${V25_EPOCHS:-500} data_module.batch_size=128 wandb.project=torchcell_019_morph_v25 regression_task.optimizer.lr=3e-4"
case "$STAGE" in
  canary) JOBS=(); RUN_OV=()
          for i in $(seq 1 "$PACK"); do
            JOBS+=("M_pool_s0:0"); RUN_OV+=("wandb.tags=[v25d,M_pool,lr3e-4,b128,split0,init0,canary]")
          done
          BASE_OV="trainer.max_epochs=${V25_CANARY_EPOCHS:-3} trainer.train_eval_every=1 data_module.batch_size=128 wandb.project=torchcell_019_morph_v25_canary regression_task.optimizer.lr=3e-4" ;;
  morph_v25d) JOBS=(); RUN_OV=()
          # Pack-major by split so a card's three co-residents share one split cache.
          for s in 0 1 2; do
            for arm in M_pool M_zs; do for b in 32 64; do
              JOBS+=("${arm}_s${s}:0")
              RUN_OV+=("data_module.batch_size=${b} wandb.tags=[v25d,${arm},lr3e-4,b${b},split${s},init0]")
            done; done
            for arm in M_pool M_zs; do for init in 1 2; do
              JOBS+=("${arm}_s${s}:${init}")
              RUN_OV+=("wandb.tags=[v25d,${arm},lr3e-4,b128,split${s},init${init}]")
            done; done
          done ;;
  *) echo "unknown V25_STAGE '$STAGE' (expected canary|morph_v25d)" >&2; exit 1 ;;
esac
N_PACKS=$(( (${#JOBS[@]} + PACK - 1) / PACK ))
if [[ -n "${V25_LIST:-}" ]]; then
  echo "stage $STAGE: ${#JOBS[@]} runs, PACK=$PACK, N_PACKS=$N_PACKS"
  for k in $(seq 0 $((N_PACKS-1))); do echo "  pack $k: ${JOBS[*]:$((k*PACK)):$PACK}"; done
  exit 0
fi

mkdir -p "$OUT_DIR"
if [ ! -d "$PROJECT_ROOT/torchcell" ]; then
  echo "FATAL: PROJECT_ROOT=$PROJECT_ROOT has no torchcell/ package dir." >&2; exit 1
fi
TASK="${SLURM_ARRAY_TASK_ID:?submit with --array=<pack index>}"
START=$((TASK * PACK))
if [[ "$START" -ge "${#JOBS[@]}" ]]; then
  echo "pack $TASK starts past the end of stage '$STAGE' (${#JOBS[@]} runs)" >&2; exit 1
fi

# ---- Delta environment (delta_grid_common.sh and delta_expr_v21_small.slurm hold the reasons)
ENV_PY=/work/hdd/bbub/miniconda3/envs/torchcell/bin/python
export DATA_ROOT="${DELTA_DATA_ROOT:-/scratch/bbub/mjvolk3/torchcell}"
STORE="$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig3_core"
if [ ! -d "$STORE/processed" ]; then
  echo "FATAL: no built fig3_core under DATA_ROOT=$DATA_ROOT (run sync_delta_fig3_core.sh from GilaHyper)" >&2; exit 1
fi
f=protT5_embedding/processed/prot_t5_xl_uniref50_all.pt
[ -f "$DATA_ROOT/data/scerevisiae/$f" ] || { echo "FATAL: embedding file missing: $DATA_ROOT/data/scerevisiae/$f" >&2; exit 1; }
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
export ASSET_IMAGES_DIR="$PROJECT_ROOT/notes/assets/images"
export MPLSTYLE_PATH="$PROJECT_ROOT/torchcell/torchcell.mplstyle"
mkdir -p "$ASSET_IMAGES_DIR"
export WANDB_MODE=online
grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null || { echo "FATAL: no api.wandb.ai entry in ~/.netrc" >&2; exit 1; }
[ -f "$PROJECT_ROOT/experiments/019-simb-multimodal/conf/$STAGE_CONFIG.yaml" ] || { echo "FATAL: $STAGE_CONFIG.yaml missing in $PROJECT_ROOT (branch $(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>&1))" >&2; exit 1; }

SRC_HASH="${TORCHCELL_SOURCE_GIT_HASH:-$(git -C "$PROJECT_ROOT" rev-parse HEAD)}"
SRC_DIFF="${TORCHCELL_SOURCE_DIFF_SHA256:-$(git -C "$PROJECT_ROOT" diff HEAD | sha256sum | cut -d" " -f1)}"
export TORCHCELL_SOURCE_GIT_HASH="$SRC_HASH" TORCHCELL_SOURCE_DIFF_SHA256="$SRC_DIFF"

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHCELL_PYTHON="$ENV_PY"
export TORCHCELL_CONFIG="$STAGE_CONFIG"
if [[ "$WORKERS" -gt 0 ]]; then
  WORKER_OV="data_module.num_workers=$WORKERS data_module.persistent_workers=true"
else
  WORKER_OV="data_module.num_workers=0"
fi
MAT_OV="+data_module.materialize=true"
ARM_SH="$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh"

echo "=========================================================="
echo "host          : $(hostname)   partition: ${SLURM_JOB_PARTITION:-?}  account: ${SLURM_JOB_ACCOUNT:-?}"
echo "job           : ${SLURM_ARRAY_JOB_ID:-none} pack=$TASK of $N_PACKS   stage=$STAGE  PACK=$PACK"
echo "project root  : $PROJECT_ROOT"
echo "source        : $SRC_HASH (diff ${SRC_DIFF:0:12})"
echo "config        : $STAGE_CONFIG   base overrides: $BASE_OV $WORKER_OV $MAT_OV"
echo "DATA_ROOT     : $DATA_ROOT"
echo "=========================================================="

( while true; do printf '%s  ' "$(date +%T)"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader; sleep 300; done ) &
SAMPLER=$!

PIDS=(); LABELS=(); RC=0
for k in $(seq 0 $((PACK - 1))); do
  IDX=$((START + k)); [[ "$IDX" -ge "${#JOBS[@]}" ]] && continue
  JOB="${JOBS[$IDX]}"; ARM="${JOB%%:*}"; SEED="${JOB##*:}"
  LOG="$OUT_DIR/019-morph-v25_${SLURM_ARRAY_JOB_ID}_${TASK}_${ARM}_seed${SEED}_${k}.out"
  echo "  launching $ARM seed=$SEED [${RUN_OV[$IDX]}] -> $LOG"
  # The run's recipe rides on the stage-wide overrides; the wandb.tags of RUN_OV replace
  # the arm script's, as the IGB launcher's SWEEP_OV does, so the view and readout scripts
  # read arm, split and init from separate tags.
  ( export TORCHCELL_EXTRA_OVERRIDES="$BASE_OV $WORKER_OV $MAT_OV trainer.early_stopping.enabled=false ${RUN_OV[$IDX]}"
    cd "$PROJECT_ROOT/experiments" && bash "$ARM_SH" "$ARM" "$SEED" ) > "$LOG" 2>&1 &
  PIDS+=("$!"); LABELS+=("$ARM:$SEED")
  sleep "${DELTA_LAUNCH_STAGGER_S:-45}"
done
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "  OK     ${LABELS[$i]}"; else echo "  FAILED ${LABELS[$i]}"; RC=1; fi
done
kill "$SAMPLER" 2>/dev/null || true
echo "  pack $TASK complete"
exit "$RC"
