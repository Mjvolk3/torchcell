#!/bin/bash
# experiments/025-solid-growth/scripts/delta_cgt.slurm
# [[experiments.025-solid-growth.scripts.delta_cgt]]
#
# Delta (NCSA) 4-GPU DDP launcher for ONE 025 CellGraphTransformer arm on the 010 build.
# Parameterized over the Hydra config name and free Hydra overrides, so one script launches
# every arm of the graph-regularization sweep:
#
#   sbatch --account=bfjt-delta-gpu --time=24:00:00 -J 025-kl-1e-3-s1 \
#     experiments/025-solid-growth/scripts/delta_cgt.slurm cgt_010b_r_kl_005 \
#     model.graph_regularization.graph_reg_lambda=1e-3 seed=1
#
#   sbatch --account=bfjt-delta-gpu --time=12:00:00 -J 025-mask-s1 \
#     experiments/025-solid-growth/scripts/delta_cgt.slurm cgt_010b_r_mask_006 seed=1
#
# See delta_submit_sweep.sh for the whole sweep and delta_preflight_025.sh for the checks to
# run on the login node first.
#
# ONE-TIME SETUP before the first sbatch. SLURM opens --output at job start, BEFORE this
# script runs, so the log dir must exist or the job dies at t=0 with NO log anywhere:
#   mkdir -p /scratch/bbub/mjvolk3/torchcell/experiments/025-solid-growth/slurm/output
#
# Every environment fact below was validated on Delta by the 019 campaign
# (experiments/019-simb-multimodal/scripts/delta_grid_common.sh) and is repeated here
# rather than sourced, because that file also exports grid-runner variables this job does
# not want.
#SBATCH --mem=200g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --gpus-per-node=4
#SBATCH --constraint="scratch"
#SBATCH --account=bfjt-delta-gpu
#SBATCH --job-name=025-CGT
#SBATCH --time=24:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell/experiments/025-solid-growth/slurm/output/%x_%j.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell/experiments/025-solid-growth/slurm/output/%x_%j.out

CONFIG_NAME="${1:?usage: sbatch delta_cgt.slurm <config_name> [hydra overrides...]}"
shift
OVERRIDES=("$@")

# The checkout this job runs. Never the shared /projects checkout, which another session may
# `git checkout` under a multi-day job (Delta grid 21796128 died that way). The submit
# directory is the repo, because that is where `sbatch` is run from.
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
# The LARGE space. /projects has a tight quota. Must agree with the Delta .env.
DELTA_DATA_ROOT="${DELTA_DATA_ROOT:-/scratch/bbub/mjvolk3/torchcell}"
DELTA_CONDA_BASE="${DELTA_CONDA_BASE:-/work/hdd/bbub/miniconda3}"
DELTA_CONDA_ENV="${DELTA_CONDA_ENV:-torchcell}"
DELTA_PY="$DELTA_CONDA_BASE/envs/$DELTA_CONDA_ENV/bin/python"
# 0 on Delta, and this is a measurement: workers are spawned interpreters that re-import
# torch + PyG + torchcell from Lustre, which held a sanity check for 38 minutes in July 2026.
NUM_WORKERS="${NUM_WORKERS:-0}"

# Which build the preflight checks. The 010b configs (cgt_010b_*) read the 010 build on
# /scratch; every other 025 config reads the 025 FULL build, which on Delta is a symlink
# /scratch/.../025-solid-growth/001-full-build -> the Taiga copy at
# /taiga/illinois/eng/chbe/zhao5/mjvolk3/projects/torchcell/... (sync_taiga_025_build.sh,
# 2026-09-09; no Duo, GilaHyper -> Radiant -> Taiga -> Delta). The symlink keeps
# DATA_ROOT on /scratch, where genome/GO/STRING/TFLink already live.
if [[ "$CONFIG_NAME" == cgt_010b_* ]]; then
  BUILD_REL="data/torchcell/experiments/010-kuzmin-tmi/001-small-build-schema-v2"
else
  BUILD_REL="data/torchcell/experiments/025-solid-growth/001-full-build"
fi

echo "=========================================================="
echo "job         : $SLURM_JOB_ID  name=$SLURM_JOB_NAME  host=$(hostname)"
echo "config      : $CONFIG_NAME  overrides: ${OVERRIDES[*]:-none}"
echo "project     : $PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"
echo "DATA_ROOT   : $DELTA_DATA_ROOT"
echo "python      : $DELTA_PY"
echo "workers     : $NUM_WORKERS"
echo "=========================================================="

# ---- PREFLIGHT. Every check below has cost a real job on Delta. ----
[[ -x "$DELTA_PY" ]] || { echo "ERROR: interpreter not found: $DELTA_PY" >&2; exit 1; }
[[ -f "$PROJECT_ROOT/experiments/025-solid-growth/conf/$CONFIG_NAME.yaml" ]] || {
  echo "ERROR: config not found: $PROJECT_ROOT/experiments/025-solid-growth/conf/$CONFIG_NAME.yaml" >&2; exit 1; }
for d in "$BUILD_REL/processed/lmdb" data/sgd/genome data/go data/string data/tflink; do
  [[ -e "$DELTA_DATA_ROOT/$d" ]] || {
    echo "ERROR: missing $DELTA_DATA_ROOT/$d" >&2
    echo "       010 build: sync_delta_010_build.sh (Duo). 025 build: sync_taiga_025_build.sh" >&2
    echo "       from GilaHyper, then the /scratch symlink to the Taiga copy (see above)." >&2
    echo "       A missing processed/ makes Neo4jCellDataset try to REBUILD from a database" >&2
    echo "       this node cannot reach." >&2
    exit 1; }
done
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "$PROJECT_ROOT/experiments" || exit 1

export DATA_ROOT="$DELTA_DATA_ROOT"
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
export ASSET_IMAGES_DIR="$PROJECT_ROOT/notes/assets/images"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
# Delta compute nodes have internet, so W&B is ONLINE.
export WANDB__SERVICE_WAIT=600

# The import must resolve to THIS checkout, not a stale editable install.
"$DELTA_PY" - <<PY || exit 1
import torchcell, sys
f = torchcell.__file__
assert f.startswith("$PROJECT_ROOT"), f"torchcell imports from {f}, not $PROJECT_ROOT"
print("torchcell:", f)
PY

# DDP rendezvous pinned to loopback: single node, and torchrun's --local-addr default is the
# hostname, which does not route on every cluster.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

"$DELTA_CONDA_BASE/envs/$DELTA_CONDA_ENV/bin/torchrun" --nnodes=1 --nproc_per_node=4 \
  --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:$MASTER_PORT \
  --rdzv-id=$SLURM_JOB_ID --local-addr=127.0.0.1 \
  "$PROJECT_ROOT/experiments/025-solid-growth/scripts/equivariant_cell_graph_transformer.py" \
  --config-name "$CONFIG_NAME" \
  data_module.num_workers=$NUM_WORKERS "${OVERRIDES[@]}"
