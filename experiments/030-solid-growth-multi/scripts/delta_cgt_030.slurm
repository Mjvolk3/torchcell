#!/bin/bash
# experiments/030-solid-growth-multi/scripts/delta_cgt_030.slurm
# [[experiments.030-solid-growth-multi.scripts.delta_cgt_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/delta_cgt_030.slurm
#
# Delta (NCSA) launcher for ONE 030 per-entry CellGraphTransformer run from the S3Q POOL
# STORE, one GPU by default. The 025 Delta launcher (delta_cgt.slurm) with the data
# source changed: instead of copying the 554 GB build to node-local disk, this copies the
# 7.5 GB pool store (make_pool_store_030.py: the arm's 1.12M records, zlib, under the full
# build's keys, with its split caches) from the Taiga mount to node-local disk in about a
# minute, and points `dataset.root_rel` at the copy. Genome, GO, STRING, TFLink and the
# four embedding builds are read from DATA_ROOT on /scratch as every 025 run did.
#
#   sbatch --account=bgcg-delta-gpu --time=48:00:00 -J 030-scl-N45L2-D4-s1 \
#       experiments/030-solid-growth-multi/scripts/delta_cgt_030.slurm cgt_030_s3_q_tok_embfit_004 \
#       subset.indices=subset_S3Q_frac4_indices_030.json.gz transforms.fit_stats_strict=false \
#       model.hidden_channels=45 model.num_transformer_layers=2 +seed=1
#
# CONTINUATION. `RESUME_FROM_JOB=<slurm job id>` makes this job resume the full training
# state of that job's `<run>-last.ckpt` (found under $DATA_ROOT/models/checkpoints/
# *-<jobid>_*/, the group the 030 script names after the host and job), so a run longer
# than 48 h is a chain of jobs submitted with --dependency=afterany. The checkpoint must
# exist when this job STARTS, which afterany guarantees.
#
# WORKERS. 025 ran NUM_WORKERS=0 on Delta because spawned workers re-import the stack from
# Lustre (38 min once). With the store on node-local disk the read is cheap but the per-
# record graph processing is not, and one rank with no workers would be CPU-bound; 6
# workers is the default here, a measurement to revisit after the calibration job.
#
# ONE-TIME: mkdir -p /scratch/bbub/mjvolk3/torchcell/experiments/030-solid-growth-multi/slurm/output
#SBATCH --mem=80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --constraint="scratch"
#SBATCH --account=bgcg-delta-gpu
#SBATCH --job-name=030-CGT
#SBATCH --time=48:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell/experiments/030-solid-growth-multi/slurm/output/%x_%j.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell/experiments/030-solid-growth-multi/slurm/output/%x_%j.out

CONFIG_NAME="${1:?usage: sbatch delta_cgt_030.slurm <config_name> [hydra overrides...]}"
shift
OVERRIDES=("$@")

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
DELTA_DATA_ROOT="${DELTA_DATA_ROOT:-/scratch/bbub/mjvolk3/torchcell}"
DELTA_CONDA_BASE="${DELTA_CONDA_BASE:-/work/hdd/bbub/miniconda3}"
DELTA_CONDA_ENV="${DELTA_CONDA_ENV:-torchcell}"
DELTA_PY="$DELTA_CONDA_BASE/envs/$DELTA_CONDA_ENV/bin/python"
NUM_WORKERS="${NUM_WORKERS:-6}"
GPUS="${GPUS:-1}"
EXP_REL="experiments/030-solid-growth-multi"
# The pool store, as synced to Taiga by `REL=... sync_taiga_025_build.sh` from GilaHyper
# (no Duo on that route). Delta mounts Taiga read-only at this path.
POOL_STORE_SRC="${POOL_STORE_SRC:-/taiga/illinois/eng/chbe/zhao5/mjvolk3/projects/torchcell/data/torchcell/experiments/030-solid-growth-multi/001-multi-build-s3q}"
RESUME_FROM_JOB="${RESUME_FROM_JOB:-}"

echo "=========================================================="
echo "job         : $SLURM_JOB_ID  name=$SLURM_JOB_NAME  host=$(hostname)"
echo "config      : $CONFIG_NAME  overrides: ${OVERRIDES[*]:-none}"
echo "project     : $PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"
echo "DATA_ROOT   : $DELTA_DATA_ROOT"
echo "pool store  : $POOL_STORE_SRC"
echo "workers     : $NUM_WORKERS   gpus: $GPUS   resume_from_job: ${RESUME_FROM_JOB:-none}"
echo "=========================================================="

# ---- PREFLIGHT ----
[[ -x "$DELTA_PY" ]] || { echo "ERROR: interpreter not found: $DELTA_PY" >&2; exit 1; }
[[ -f "$PROJECT_ROOT/$EXP_REL/conf/$CONFIG_NAME.yaml" ]] || {
  echo "ERROR: config not found: $PROJECT_ROOT/$EXP_REL/conf/$CONFIG_NAME.yaml" >&2; exit 1; }
[[ -f "$POOL_STORE_SRC/processed/lmdb/STORE.json" ]] || {
  echo "ERROR: pool store missing: $POOL_STORE_SRC/processed/lmdb/STORE.json" >&2
  echo "       on GilaHyper: REL=data/torchcell/experiments/030-solid-growth-multi/001-multi-build-s3q bash experiments/025-solid-growth/scripts/sync_taiga_025_build.sh" >&2
  exit 1; }
N_INDEX=$(find "$POOL_STORE_SRC/data_module_cache" -maxdepth 1 -name 'index_seed_*.json' 2>/dev/null | wc -l)
[[ "$N_INDEX" -gt 0 ]] || { echo "ERROR: pool store has no split cache (index_seed_*.json)" >&2; exit 1; }
for d in data/sgd/genome data/go/go.obo data/string data/tflink \
         data/scerevisiae/fudt_embedding/processed/species_upstream.pt \
         data/scerevisiae/fudt_embedding/processed/species_downstream.pt \
         data/scerevisiae/calm_embedding/processed/calm.pt \
         data/scerevisiae/protT5_embedding/processed/prot_t5_xl_uniref50_all.pt; do
  [[ -e "$DELTA_DATA_ROOT/$d" ]] || { echo "ERROR: missing $DELTA_DATA_ROOT/$d" >&2; exit 1; }
done
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---- STAGE THE POOL STORE ONTO NODE-LOCAL DISK ----
LOCAL_STORE="${SLURM_TMPDIR:-/tmp}/$SLURM_JOB_ID/001-multi-build-s3q"
mkdir -p "$LOCAL_STORE"
trap 'rm -rf "${SLURM_TMPDIR:-/tmp}/$SLURM_JOB_ID"' EXIT
t0=$(date +%s)
# The .lock files beside the index JSONs are not copied: filelock opens them O_NOFOLLOW
# and recreates them locally; the copy is the only reader anyway.
rsync -a --exclude='*.lock' "$POOL_STORE_SRC/" "$LOCAL_STORE/" || { echo "ERROR: stage-in failed" >&2; exit 1; }
mkdir -p "$LOCAL_STORE/raw/lmdb"
echo "staged pool store in $(( $(date +%s) - t0 )) s: $(du -sh "$LOCAL_STORE" | cut -f1) -> $LOCAL_STORE"
cat "$LOCAL_STORE/processed/lmdb/STORE.json"

# ---- CONTINUATION ----
if [[ -n "$RESUME_FROM_JOB" ]]; then
  mapfile -t CKPTS < <(ls -t "$DELTA_DATA_ROOT"/models/checkpoints/*-"${RESUME_FROM_JOB}"_*/*-last.ckpt 2>/dev/null)
  [[ "${#CKPTS[@]}" -ge 1 ]] || { echo "ERROR: no *-last.ckpt under $DELTA_DATA_ROOT/models/checkpoints/*-${RESUME_FROM_JOB}_*/" >&2; exit 1; }
  echo "resuming from ${CKPTS[0]}"
  OVERRIDES+=("+resume.ckpt_path=${CKPTS[0]}")
fi

cd "$PROJECT_ROOT/experiments" || exit 1
export DATA_ROOT="$DELTA_DATA_ROOT"
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
export ASSET_IMAGES_DIR="$PROJECT_ROOT/notes/assets/images"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
# Delta compute nodes have internet, so W&B is ONLINE.
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=600

"$DELTA_PY" - <<PY || exit 1
import torchcell
f = torchcell.__file__
assert f.startswith("$PROJECT_ROOT"), f"torchcell imports from {f}, not $PROJECT_ROOT"
print("torchcell:", f)
PY

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
"$DELTA_CONDA_BASE/envs/$DELTA_CONDA_ENV/bin/torchrun" --nnodes=1 --nproc_per_node=$GPUS \
  --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:$MASTER_PORT \
  --rdzv-id=$SLURM_JOB_ID --local-addr=127.0.0.1 \
  "$PROJECT_ROOT/$EXP_REL/scripts/equivariant_cell_graph_transformer.py" \
  --config-name "$CONFIG_NAME" \
  "dataset.root_rel=$LOCAL_STORE" trainer.devices=$GPUS \
  data_module.num_workers=$NUM_WORKERS "${OVERRIDES[@]}"
