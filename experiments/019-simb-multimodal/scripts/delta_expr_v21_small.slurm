#!/bin/bash
# experiments/019-simb-multimodal/scripts/delta_expr_v21_small.slurm
# [[experiments.019-simb-multimodal.scripts.delta_expr_v21_small]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/delta_expr_v21_small
#
# THE SMALL-TRUNK SINGLE-CHANGE ROUND ON DELTA (2026-10-05): ONE A40 per job, every job an
# independent submission (its own job id, no --dependency, no array throttle), each job a
# pack of PACK runs of one arm sharing the card. conf/cgt_expr_v21_small.yaml carries the
# design; the S_* arms live in gh_expr_008_arm.sh.
#
# WHY SINGLE-GPU INDEPENDENT JOBS. A whole-node job waits for four free cards at once and a
# chained array inherits one submit time, so later tasks lose age priority to everything
# queued since (the IGB cabbi lesson of 2026-10-01). One card per job backfills into gaps
# and every job accrues its own priority from its own submit time.
#
# PACKING is the lever that makes twelve split seeds fit two days on one card. v19 timing
# on an A40 at three runs per card was 35 s per epoch (8.5 GPU hours per 1,200-epoch run);
# the GilaHyper benchmark of 2026-10-05 (gh_small_model_bench.slurm, job 3267) times the
# ProtT5-only trunk at 4, 6 and 8 runs per card, and PACK is set from it. Twelve seeds of one
# arm are then two or three jobs of PACK runs each.
#
# ONE-TIME SETUP on Delta (the user, Duo):
#   mkdir -p /work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v21
#   git -C /work/hdd/bbub/mjvolk3/torchcell-v11 pull        # job-owned clone of this branch
#   # the store: bash experiments/019-simb-multimodal/scripts/sync_delta_store.sh fig3_proteome
#   # from GilaHyper, into /scratch/bbub/mjvolk3/torchcell (the root every launcher reads)
#
# SUBMIT, one sbatch per pack, from the clone (the account is the user's call; bfjt held the
# largest balance on 2026-10-04):
#   R=/work/hdd/bbub/mjvolk3/torchcell-v11
#   H=$(git -C $R rev-parse HEAD); D=$(git -C $R diff HEAD | sha256sum | cut -d" " -f1)
#   for k in $(seq 0 $((N_PACKS-1))); do
#     sbatch --array=$k --account=bfjt-delta-gpu -J 019-v21w1-small-p$k \
#       --export=ALL,V21_STAGE=round,V21_PROJECT_ROOT=$R,TORCHCELL_SOURCE_GIT_HASH=$H,TORCHCELL_SOURCE_DIFF_SHA256=$D \
#       $R/experiments/019-simb-multimodal/scripts/delta_expr_v21_small.slurm
#   done
# N_PACKS is printed by `V21_STAGE=round V21_LIST=1 bash <this script>` (no sbatch).
# A canary first: `--array=0 --time=02:00:00 --export=ALL,V21_STAGE=canary,...` runs PACK
# copies of S_ref for three epochs with a test pass, which is what proves the store, the
# embedding file, the config and the card memory before 84 runs are queued.
#
#SBATCH --job-name=019-v21-small
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfjt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60g
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v21/%x_%A_%a.out
#SBATCH --error=/work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v21/%x_%A_%a.out

set -euo pipefail

PROJECT_ROOT="${V21_PROJECT_ROOT:?set V21_PROJECT_ROOT to the job-owned clone}"
OUT_DIR="${V21_LOG_DIR:-/work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v21}"
STAGE="${V21_STAGE:?set V21_STAGE=canary|round}"
STAGE_CONFIG=cgt_expr_v21_small
# PACK: runs sharing the one card. FOUR, from the GilaHyper benchmark of 2026-10-05: 8 to
# 10 GB of card memory per run at every trunk shape, so four fit the A40's 44.4 GiB with
# room and five do not. Delta's own v11 round (one run per A40, zero workers) logged 81 to
# 134 s per epoch, and the benchmark's zero-worker runs cost 78 to 81 s each with four per
# card, so the main process bounds a run on both machines and four per card should cost
# about what one does (hypothesis until the canary logs perf/epoch_seconds): 1,200 epochs
# in roughly 27 to 45 h, inside the two-day limit.
PACK="${V21_PACK:-4}"
# WORKERS per run, PERSISTENT. The Delta rule of zero workers (delta_grid_common.sh) was
# measured with non-persistent workers, where every epoch re-spawns interpreters that import
# torch from Lustre. With persistent workers the spawn is paid once at startup. The GilaHyper
# benchmark of 2026-10-05 measured 35 to 40 s per epoch at four runs with two or three
# workers each against 69 to 75 s at zero, so workers halve the round. One worker per run
# fits the 8 CPUs an A40's share of a gpuA40x4 node carries (4 mains + 4 workers). The
# canary's epoch 1 time is the startup cost on Lustre; if it runs to hours, resubmit the
# round with V21_WORKERS=0 (1,200 epochs then take 27 to 45 h, still inside two days at
# the low end). Measured on GilaHyper at exactly this shape (four runs, one worker each,
# eight CPUs, job 3273): 58 to 64 s per epoch, about 20 h per 1,200-epoch pack; with three
# workers each on sixteen CPUs 35 to 38 s. If gpuA40x4 grants --cpus-per-task=16 for one
# card, submit with that and V21_WORKERS=3.
WORKERS="${V21_WORKERS:-1}"
# Flipped to 0 in the commit that fixed the trunk shape from the benchmark (2026-10-05).
BENCH_PENDING="${V21_BENCH_PENDING:-0}"

# Nine arms: the seven single changes plus the two higher-rank operator forms
# (S_basis64, S_hadam) that, with S_sink against S_ref, settle the operator on twelve seeds.
ARMS_ROUND=(S_ref S_mask S_sink S_prop2 S_nodrop S_stack S_prot S_basis64 S_hadam)
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11)
case "$STAGE" in
  canary) JOBS=(); for i in $(seq 1 "$PACK"); do JOBS+=("S_ref_s$((i-1)):0"); done
          EXTRA_OV="trainer.max_epochs=3 trainer.train_eval_every=1 wandb.project=torchcell_019_expr_v21_canary" ;;
  round)  if [[ "$BENCH_PENDING" == "1" ]]; then
            echo "REFUSED: the trunk shape in $STAGE_CONFIG.yaml is a placeholder until the GilaHyper benchmark reads out; set V21_BENCH_PENDING=0 in the commit that fixes it." >&2
            exit 1
          fi
          # Arm-major: pack k holds PACK consecutive seeds of ONE arm, so a card's co-residents
          # are the same arm and every arm's twelve seeds run under identical packing.
          JOBS=(); for a in "${ARMS_ROUND[@]}"; do for s in "${SEEDS[@]}"; do JOBS+=("${a}_s${s}:0"); done; done
          EXTRA_OV="" ;;
  *) echo "unknown V21_STAGE '$STAGE' (expected canary|round)" >&2; exit 1 ;;
esac
N_PACKS=$(( (${#JOBS[@]} + PACK - 1) / PACK ))
if [[ -n "${V21_LIST:-}" ]]; then
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

# ---- Delta environment (delta_grid_common.sh and delta_expr_v11_emb.slurm hold the reasons)
ENV_PY=/work/hdd/bbub/miniconda3/envs/torchcell/bin/python
export DATA_ROOT="${DELTA_DATA_ROOT:-/scratch/bbub/mjvolk3/torchcell}"
STORE="$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig3_proteome"
if [ ! -d "$STORE/processed" ]; then
  echo "FATAL: no built fig3_proteome under DATA_ROOT=$DATA_ROOT (run sync_delta_store.sh fig3_proteome from GilaHyper)" >&2; exit 1
fi
for f in protT5_embedding/processed/prot_t5_xl_uniref50_all.pt \
         fudt_embedding/processed/species_upstream.pt \
         fudt_embedding/processed/species_downstream.pt \
         calm_embedding/processed/calm.pt; do
  [ -f "$DATA_ROOT/data/scerevisiae/$f" ] || { echo "FATAL: embedding file missing: $DATA_ROOT/data/scerevisiae/$f" >&2; exit 1; }
done
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
# Early stopping off as in every round; workers per the header.
if [[ "$WORKERS" -gt 0 ]]; then
  WORKER_OV="data_module.num_workers=$WORKERS data_module.persistent_workers=true"
else
  WORKER_OV="data_module.num_workers=0"
fi
export TORCHCELL_EXTRA_OVERRIDES="$EXTRA_OV $WORKER_OV trainer.early_stopping.enabled=false"
ARM_SH="$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh"

echo "=========================================================="
echo "host          : $(hostname)   partition: ${SLURM_JOB_PARTITION:-?}  account: ${SLURM_JOB_ACCOUNT:-?}"
echo "job           : ${SLURM_ARRAY_JOB_ID:-none} pack=$TASK of $N_PACKS   stage=$STAGE  PACK=$PACK"
echo "project root  : $PROJECT_ROOT"
echo "source        : $SRC_HASH (diff ${SRC_DIFF:0:12})"
echo "config        : $STAGE_CONFIG   overrides: $TORCHCELL_EXTRA_OVERRIDES"
echo "DATA_ROOT     : $DATA_ROOT"
echo "=========================================================="

( while true; do printf '%s  ' "$(date +%T)"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader; sleep 300; done ) &
SAMPLER=$!

PIDS=(); LABELS=(); RC=0
for k in $(seq 0 $((PACK - 1))); do
  IDX=$((START + k)); [[ "$IDX" -ge "${#JOBS[@]}" ]] && continue
  JOB="${JOBS[$IDX]}"; ARM="${JOB%%:*}"; SEED="${JOB##*:}"
  LOG="$OUT_DIR/019-v21_${SLURM_ARRAY_JOB_ID}_${TASK}_${ARM}_seed${SEED}.out"
  echo "  launching $ARM seed=$SEED -> $LOG"
  ( cd "$PROJECT_ROOT/experiments" && bash "$ARM_SH" "$ARM" "$SEED" ) > "$LOG" 2>&1 &
  PIDS+=("$!"); LABELS+=("$ARM:$SEED")
  # Stagger so the Lustre import storm of one run warms the client cache for the next.
  sleep "${DELTA_LAUNCH_STAGGER_S:-45}"
done
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "  OK     ${LABELS[$i]}"; else echo "  FAILED ${LABELS[$i]}"; RC=1; fi
done
kill "$SAMPLER" 2>/dev/null || true
echo "  pack $TASK complete"
exit "$RC"
