#!/bin/bash
# experiments/030-solid-growth-multi/scripts/perlmutter_cgt_030.slurm
# [[experiments.030-solid-growth-multi.scripts.perlmutter_cgt_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/perlmutter_cgt_030.slurm
#
# NERSC Perlmutter one-node 4 x A100 DDP launcher for ONE 030 per-entry CellGraphTransformer
# run from a POOL STORE (make_pool_store_030.py), the sibling of delta_cgt_030.slurm with
# the 025 Perlmutter launcher's account, QOS and filesystem rules:
#
#   POOL_STORE_REL=data/torchcell/experiments/030-solid-growth-multi/001-multi-build-s3r \
#   sbatch -J 030-pm-r-onto_aff-s1 perlmutter_cgt_030.slurm cgt_030_s3_r_onto_affine_010 +seed=1
#
# WHAT THE MACHINE IMPOSES.
#   - Account m5429_g, QOS `regular` on the command line (shows as gpu_regular; 48 h max
#     wall, 400 node-hours per user). 24 h per job as the 025 launcher chose: shorter jobs
#     schedule sooner, so a 200-epoch run is a CHAIN of jobs (perlmutter_chain_030.sh).
#   - Diskless GPU nodes: no stage-in. The pool store (16 GB) is read in place from the
#     all-flash $SCRATCH Lustre, where the LMDB lock file and the filelock locks beside the
#     split indices are allowed. CFS refuses file locks (errno 524), so DATA_ROOT and
#     WANDB_DIR stay on /pscratch; the code tree and the conda env live on CFS.
#   - Compute nodes reach api.wandb.ai, so W&B is ONLINE.
#
# CONTINUATION. `RESUME_FROM_JOB=<slurm job id>` resumes the full training state from that
# job's `<run>-last.ckpt` under $DATA_ROOT/models/checkpoints/*-<jobid>_*/ (the group the
# 030 script names after the host and job). The checkpoint must exist when this job
# STARTS, which --dependency=afterany guarantees. The continuation is a new W&B run with
# `resumed_from` in its config, exactly as on Delta.
#
# WORKERS. Spawned workers re-import the stack from CFS once (minutes); with the store on
# flash the per-record graph processing dominates, so the default is 8 of the node's 128.
#
# ONE-TIME: mkdir -p <PROJECT_ROOT>/experiments/030-solid-growth-multi/slurm/output
#SBATCH -A m5429_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH --time=24:00:00
#SBATCH --job-name=030-CGT
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/global/cfs/cdirs/m5429/mjvolk3/torchcell-030/experiments/030-solid-growth-multi/slurm/output/%x_%j.out
#SBATCH --error=/global/cfs/cdirs/m5429/mjvolk3/torchcell-030/experiments/030-solid-growth-multi/slurm/output/%x_%j.out

CONFIG_NAME="${1:?usage: sbatch perlmutter_cgt_030.slurm <config_name> [hydra overrides...]}"
shift
OVERRIDES=("$@")

TC_PREFIX="${TC_PREFIX:-/global/cfs/cdirs/m5429/mjvolk3}"
PROJECT_ROOT="${PROJECT_ROOT:-$TC_PREFIX/torchcell-030}"
PM_DATA_ROOT="${PM_DATA_ROOT:-/pscratch/sd/m/mjvolk3/torchcell-data}"
PM_PY="$TC_PREFIX/miniforge3/envs/torchcell/bin/python"
PM_TORCHRUN="$TC_PREFIX/miniforge3/envs/torchcell/bin/torchrun"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPUS="${GPUS:-4}"
EXP_REL="experiments/030-solid-growth-multi"
POOL_STORE_REL="${POOL_STORE_REL:?set POOL_STORE_REL=data/torchcell/experiments/030-solid-growth-multi/001-multi-build-s3r (or -s3q)}"
RESUME_FROM_JOB="${RESUME_FROM_JOB:-}"

echo "=========================================================="
echo "job         : $SLURM_JOB_ID  name=$SLURM_JOB_NAME  host=$(hostname)"
echo "config      : $CONFIG_NAME  overrides: ${OVERRIDES[*]:-none}"
echo "project     : $PROJECT_ROOT"
echo "DATA_ROOT   : $PM_DATA_ROOT"
echo "pool store  : $PM_DATA_ROOT/$POOL_STORE_REL"
echo "workers     : $NUM_WORKERS   gpus: $GPUS   resume_from_job: ${RESUME_FROM_JOB:-none}"
echo "start       : $(date)"
echo "=========================================================="

# ---- PREFLIGHT ----
[[ -x "$PM_PY" ]] || { echo "ERROR: interpreter not found: $PM_PY" >&2; exit 1; }
[[ -f "$PROJECT_ROOT/$EXP_REL/conf/$CONFIG_NAME.yaml" ]] || {
  echo "ERROR: config not found: $PROJECT_ROOT/$EXP_REL/conf/$CONFIG_NAME.yaml" >&2; exit 1; }
STORE="$PM_DATA_ROOT/$POOL_STORE_REL"
[[ -f "$STORE/processed/lmdb/STORE.json" ]] || {
  echo "ERROR: pool store missing: $STORE/processed/lmdb/STORE.json" >&2; exit 1; }
N_INDEX=$(find "$STORE/data_module_cache" -maxdepth 1 -name 'index_seed_*.json' 2>/dev/null | wc -l)
[[ "$N_INDEX" -gt 0 ]] || { echo "ERROR: pool store has no split cache (index_seed_*.json)" >&2; exit 1; }
for d in data/sgd/genome data/go/go.obo data/string data/tflink \
         data/scerevisiae/fudt_embedding/processed/species_upstream.pt \
         data/scerevisiae/fudt_embedding/processed/species_downstream.pt \
         data/scerevisiae/calm_embedding/processed/calm.pt \
         data/scerevisiae/protT5_embedding/processed/prot_t5_xl_uniref50_all.pt; do
  [[ -e "$PM_DATA_ROOT/$d" ]] || { echo "ERROR: missing $PM_DATA_ROOT/$d" >&2; exit 1; }
done
# An empty raw/lmdb satisfies PyG's files_exist check so the base class never tries to
# rebuild from a database this node cannot reach.
mkdir -p "$STORE/raw/lmdb"
cat "$STORE/processed/lmdb/STORE.json"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---- CONTINUATION ----
if [[ -n "$RESUME_FROM_JOB" ]]; then
  mapfile -t CKPTS < <(ls -t "$PM_DATA_ROOT"/models/checkpoints/*-"${RESUME_FROM_JOB}"_*/*-last.ckpt 2>/dev/null)
  [[ "${#CKPTS[@]}" -ge 1 ]] || { echo "ERROR: no *-last.ckpt under $PM_DATA_ROOT/models/checkpoints/*-${RESUME_FROM_JOB}_*/" >&2; exit 1; }
  echo "resuming from ${CKPTS[0]}"
  OVERRIDES+=("+resume.ckpt_path=${CKPTS[0]}")
fi

cd "$PROJECT_ROOT/experiments" || exit 1
export DATA_ROOT="$PM_DATA_ROOT"
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
export ASSET_IMAGES_DIR="$PROJECT_ROOT/notes/assets/images"
export WANDB_DIR=/pscratch/sd/m/mjvolk3/wandb
mkdir -p "$WANDB_DIR" "$ASSET_IMAGES_DIR"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=600

"$PM_PY" - <<PY || exit 1
import torchcell
f = torchcell.__file__
assert f.startswith("$PROJECT_ROOT"), f"torchcell imports from {f}, not $PROJECT_ROOT"
print("torchcell:", f)
PY

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
"$PM_TORCHRUN" --nnodes=1 --nproc_per_node=$GPUS \
  --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:$MASTER_PORT \
  --rdzv-id=$SLURM_JOB_ID --local-addr=127.0.0.1 \
  "$PROJECT_ROOT/$EXP_REL/scripts/equivariant_cell_graph_transformer.py" \
  --config-name "$CONFIG_NAME" \
  "dataset.root_rel=$POOL_STORE_REL" trainer.devices=$GPUS \
  data_module.num_workers=$NUM_WORKERS "${OVERRIDES[@]}"
echo "end         : $(date)"
