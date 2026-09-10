#!/bin/bash
# experiments/019-simb-multimodal/scripts/delta_expr_v11_emb.slurm
# [[experiments.019-simb-multimodal.scripts.delta_expr_v11_emb]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/delta_expr_v11_emb
#
# THE INPUT-RICHNESS ROUND on Delta: four embedding stacks x 3 seeds = 12 runs, ONE run per
# A40 for the whole job, four per node, three whole-node array tasks. Every operational
# lesson here is inherited from delta_expr_v10_grid.slurm and delta_grid_common.sh, which
# explain them at length; this file keeps the gates and the short reason for each.
#
#   # ONE-TIME SETUP. Slurm opens --output BEFORE the script body runs, so the log directory
#   # must exist, and it sits OUTSIDE every git checkout so no branch switch can remove it:
#   mkdir -p /work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v11
#   # The job runs from a checkout it OWNS (grid 21796128 died when a shared checkout was
#   # switched to another branch mid-flight):
#   git clone --branch multimodal-phenotype-retrospective \
#       https://github.com/Mjvolk3/torchcell /work/hdd/bbub/mjvolk3/torchcell-v11
#
#   # THE ROUND. The account is bbub-delta-gpu because the 025 jobs queued the same day sit
#   # on bfjt; two arrays on one account share a fairshare pool and hold each other back.
#   H=$(git -C /work/hdd/bbub/mjvolk3/torchcell-v11 rev-parse HEAD)
#   D=$(git -C /work/hdd/bbub/mjvolk3/torchcell-v11 diff HEAD | sha256sum | cut -d" " -f1)
#   sbatch --array=0-2 --account=bbub-delta-gpu \
#       --export=ALL,V11_STAGE=round,V11_PROJECT_ROOT=/work/hdd/bbub/mjvolk3/torchcell-v11,TORCHCELL_SOURCE_GIT_HASH=$H,TORCHCELL_SOURCE_DIFF_SHA256=$D \
#       /work/hdd/bbub/mjvolk3/torchcell-v11/experiments/019-simb-multimodal/scripts/delta_expr_v11_emb.slurm
#
# NO DELTA CANARY. The widest stack (E_full, 6,912-d input) is proven to compose, build and
# step by a fast_dev_run on GilaHyper against the same embedding files before submission,
# which is what a Delta canary would test at the price of a 3 h backfill slot and a 43 min
# startup. The arms differ from v10 only in `cell_dataset.node_embeddings`.
#
# RESOURCE SHAPE is the one Delta accepts for this work: one whole gpuA40x4 node, 32 CPUs
# to 4 GPUs, 180 GB, 2 days, no --constraint.
#
#SBATCH --job-name=019-expr-v11
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbub-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=180g
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v11/%x_%A_%a.out
#SBATCH --error=/work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v11/%x_%A_%a.out

set -euo pipefail

PROJECT_ROOT="${V11_PROJECT_ROOT:?set V11_PROJECT_ROOT to the job-owned clone}"
OUT_DIR="${V11_LOG_DIR:-/work/hdd/bbub/mjvolk3/slurm-logs/019-expr-v11}"
mkdir -p "$OUT_DIR"
if [ ! -d "$PROJECT_ROOT/torchcell" ]; then
  echo "FATAL: PROJECT_ROOT=$PROJECT_ROOT has no torchcell/ package dir." >&2
  exit 1
fi

STAGE="${V11_STAGE:?set V11_STAGE=canary|round via --export=ALL,V11_STAGE=...}"
STAGE_CONFIG=cgt_expr_v11_emb

# ---------------------------------------------------------------- THE ARMS
# Each arm is a node_embeddings list; the config explains the four. Names are the W&B arm
# tag and the log-file label.
ARM_NAMES=(E_calm E_ptt5 E_calm_ptt5 E_full)
ARM_EMB=(
  "[calm]"
  "[prot_T5_all]"
  "[calm,prot_T5_all]"
  "[nt_window_five_prime_1003,nt_window_three_prime_300,calm,prot_T5_all]"
)

case "$STAGE" in
  # PRE-FLIGHT on Delta, kept for the case where the GilaHyper smoke test is not enough:
  # the narrowest and widest arms at fast_dev_run. Submit with --time=03:00:00 --array=0
  # --gpus-per-node=2, since startup dominates (see the v10 launcher).
  canary) ARMS=(0 3); SEEDS=(0)
          EXTRA_OV="trainer.fast_dev_run=true" ;;
  # THE ROUND. Seed-major, so each whole node runs one seed of all four arms: the node is a
  # seed-level block and every arm contrast is within node.
  round)  ARMS=(0 1 2 3); SEEDS=(0 1 2)
          EXTRA_OV="" ;;
  *) echo "unknown V11_STAGE '$STAGE' (expected canary|round)" >&2; exit 1 ;;
esac

JOBS=()
for seed in "${SEEDS[@]}"; do
  for a in "${ARMS[@]}"; do
    JOBS+=("$a:$seed")
  done
done

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
PER_TASK="$GPUS_PER_NODE"
TASK="${SLURM_ARRAY_TASK_ID:?this script must be submitted with --array}"
START=$((TASK * PER_TASK))
if [[ "$START" -ge "${#JOBS[@]}" ]]; then
  echo "task $TASK starts past the end of stage '$STAGE' (${#JOBS[@]} runs)" >&2
  exit 1
fi

# ZERO DATALOADER WORKERS ON DELTA: spawn workers re-import the stack off Lustre and stall
# (delta_grid_common.sh, canaries 21785783 and 21788244).
NUM_WORKERS="${DELTA_NUM_WORKERS:-0}"

echo "=========================================================="
echo "host          : $(hostname)"
echo "partition     : ${SLURM_JOB_PARTITION:-unknown}  account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "job           : ${SLURM_ARRAY_JOB_ID:-none} task=${TASK}"
echo "stage         : $STAGE  (runs ${START}..$((START + PER_TASK - 1)) of ${#JOBS[@]}, 1 run/GPU x $GPUS_PER_NODE GPUs)"
echo "project root  : $PROJECT_ROOT"
echo "config        : $STAGE_CONFIG"
echo "num_workers   : $NUM_WORKERS  (of $SLURM_CPUS_PER_TASK cpus)"
echo "=========================================================="

# SOURCE PROVENANCE from the submit shell (compute nodes may lack git); an unattributable
# run must not be pooled with attributable ones.
SRC_HASH="${TORCHCELL_SOURCE_GIT_HASH:-}"
SRC_DIFF="${TORCHCELL_SOURCE_DIFF_SHA256:-}"
if [[ -z "$SRC_HASH" ]] && command -v git >/dev/null 2>&1; then
  SRC_HASH="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
  SRC_DIFF="$(git -C "$PROJECT_ROOT" diff HEAD | sha256sum | cut -d" " -f1)"
fi
if [[ -n "$SRC_HASH" && -n "$SRC_DIFF" ]]; then
  echo "source        : $SRC_HASH (diff ${SRC_DIFF:0:12})"
  export TORCHCELL_SOURCE_GIT_HASH="$SRC_HASH" TORCHCELL_SOURCE_DIFF_SHA256="$SRC_DIFF"
else
  echo "source        : UNAVAILABLE -- no git here and no hash exported at submit time"
fi

# The env python directly: /work/hdd/bbub/miniconda3 is a prefix install with no
# bin/activate, and `source` there fails silently into system python.
ENV_PY=/work/hdd/bbub/miniconda3/envs/torchcell/bin/python
# DATA_ROOT is /scratch. Neo4jCellDataset skips the database only when processed/ exists,
# so a root without the build makes a compute node try to REBUILD with no Neo4j.
export DATA_ROOT="${DELTA_DATA_ROOT:-/scratch/bbub/mjvolk3/torchcell}"
if [ ! -d "$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig3_core/processed" ]; then
  echo "FATAL: no built fig3_core under DATA_ROOT=$DATA_ROOT" >&2
  exit 1
fi
# EVERY EMBEDDING FILE THE ROUND READS, checked before a GPU is touched. The v10 grid found
# a width mismatch only at model construction; a missing .pt would surface the same way,
# 43 minutes into each run.
for f in nucleotide_transformer_embedding/processed/nt_window_five_prime_1003.pt \
         nucleotide_transformer_embedding/processed/nt_window_three_prime_300.pt \
         calm_embedding/processed/calm.pt \
         protT5_embedding/processed/prot_t5_xl_uniref50_all.pt; do
  if [ ! -f "$DATA_ROOT/data/scerevisiae/$f" ]; then
    echo "FATAL: embedding file missing: $DATA_ROOT/data/scerevisiae/$f" >&2
    exit 1
  fi
done

# THE JOB CARRIES ITS OWN ENVIRONMENT instead of a gitignored .env (grid 21814726 died on
# KeyError: 'EXPERIMENT_ROOT' in a fresh clone). W&B is ONLINE on Delta, from ~/.netrc.
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
export ASSET_IMAGES_DIR="$PROJECT_ROOT/notes/assets/images"
export MPLSTYLE_PATH="$PROJECT_ROOT/torchcell/torchcell.mplstyle"
export WANDB_MODE=online
mkdir -p "$ASSET_IMAGES_DIR"
for v in EXPERIMENT_ROOT ASSET_IMAGES_DIR MPLSTYLE_PATH; do
  p="${!v}"
  if [ ! -e "$p" ]; then
    echo "FATAL: $v=$p does not exist." >&2
    exit 1
  fi
done
if ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
  echo "FATAL: no api.wandb.ai entry in ~/.netrc and WANDB_MODE=online; run 'wandb login'." >&2
  exit 1
fi
STAGE_CONFIG_FILE="$PROJECT_ROOT/experiments/019-simb-multimodal/conf/$STAGE_CONFIG.yaml"
if [ ! -f "$STAGE_CONFIG_FILE" ]; then
  echo "FATAL: $STAGE_CONFIG_FILE is missing (PROJECT_ROOT on branch $(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>&1))." >&2
  exit 1
fi
export PYTHONWARNINGS=ignore
export PYTHONPATH="$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB__SERVICE_WAIT=600
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT" || exit 1

# IMPORT PROVENANCE GATE: the editable install must not shadow PROJECT_ROOT.
RESOLVED="$("$ENV_PY" -c 'import torchcell, os; print(os.path.realpath(os.path.dirname(torchcell.__file__)))')"
EXPECTED="$(cd "$PROJECT_ROOT" && pwd -P)/torchcell"
if [[ "$RESOLVED" != "$EXPECTED" ]]; then
  echo "FATAL: torchcell imports from $RESOLVED but PROJECT_ROOT is $EXPECTED." >&2
  exit 1
fi
echo "import check  : torchcell -> $RESOLVED (matches PROJECT_ROOT)"

TRAIN=experiments/019-simb-multimodal/scripts/train_cgt_multitask.py
RC=0
PIDS=()
LABELS=()
for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
  IDX=$((START + gpu))
  [[ "$IDX" -ge "${#JOBS[@]}" ]] && continue
  JOB="${JOBS[$IDX]}"
  A="${JOB%%:*}"
  SEED="${JOB##*:}"
  NAME="${ARM_NAMES[$A]}"
  LOG="$OUT_DIR/019-expr-v11_${SLURM_ARRAY_JOB_ID}_${TASK}_${NAME}_seed${SEED}.out"
  echo "  [gpu $gpu] $NAME seed=$SEED -> $LOG"
  echo "    node_embeddings: ${ARM_EMB[$A]}"
  # CUDA_VISIBLE_DEVICES pins ONE card per run. The embedding list is passed as one quoted
  # Hydra override so the commas survive the shell.
  CUDA_VISIBLE_DEVICES="$gpu" "$ENV_PY" "$TRAIN" --config-name "$STAGE_CONFIG" \
      "cell_dataset.node_embeddings=${ARM_EMB[$A]}" \
      $EXTRA_OV \
      "data_module.num_workers=$NUM_WORKERS" \
      "trainer.early_stopping.enabled=false" \
      "wandb.tags=[v11,$NAME,seed$SEED,stage-$STAGE]" \
      "seed=$SEED" \
      > "$LOG" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$NAME:$SEED")
  # STAGGER the imports so the node's Lustre client cache is warm for runs 2 to 4.
  [[ "$gpu" -lt $((GPUS_PER_NODE - 1)) ]] && sleep "${DELTA_LAUNCH_STAGGER_S:-60}"
done

for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "FAILED: ${LABELS[$i]}" >&2
    RC=1
  fi
done
echo "  task $TASK complete"
exit "$RC"
