#!/bin/bash
# experiments/030-solid-growth-multi/scripts/gh_cgt_030.slurm
# [[experiments.030-solid-growth-multi.scripts.gh_cgt_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/gh_cgt_030.slurm
#
# GilaHyper launcher for ONE 030 per-entry CellGraphTransformer run from a POOL STORE
# (make_pool_store_030.py) on $DATA_ROOT, one GPU by default. The IGB launcher
# (igb_mmli_cgt_030.slurm) without the RAM-disk staging: GilaHyper's /scratch store is
# read in place. `GPUS` sets torchrun's process count and trainer.devices; `STORE_REL`
# picks the pool store (default the strict S3Q store the Q arms read).
#
#   sbatch -J 030-aff-q4-s1 experiments/030-solid-growth-multi/scripts/gh_cgt_030.slurm \
#       cgt_030_s3_q_tok_affine_005 +seed=1 \
#       subset.indices=subset_S3Q_frac4_indices_030.json.gz transforms.fit_stats_strict=false \
#       trainer.max_epochs=40
#   STORE_REL=data/torchcell/experiments/030-solid-growth-multi/001-multi-build-s3r \
#       sbatch ... cgt_030_s3_r_tok_embfit_003 +seed=1
#
# MEMORY AND CPUS. GilaHyper's MaxMemPerCPU is 4.1 GB, so --mem above cpus x 4.1 GB
# silently raises the CPU request (gh_smoke_dataset_token.slurm, job 2860). One rank of
# the 2-GPU IGB runs peaked near 36 GB RSS; 14 CPUs x 4.1 GB = 57 GB covers one rank and
# its 12 workers.
#SBATCH -p main
#SBATCH --mem=56g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --job-name=030-gh-cgt
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-token-affine/experiments/030-solid-growth-multi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-token-affine/experiments/030-solid-growth-multi/slurm/output/%x_%j.out

CONFIG_NAME="${1:?usage: sbatch gh_cgt_030.slurm <config_name> [hydra overrides...]}"
shift
OVERRIDES=("$@")

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
EXP_REL="experiments/030-solid-growth-multi"
GPUS="${GPUS:-1}"
STORE_REL="${STORE_REL:-data/torchcell/experiments/030-solid-growth-multi/001-multi-build-s3q}"

echo "=========================================================="
echo "job         : $SLURM_JOB_ID  name=$SLURM_JOB_NAME  host=$(hostname)  started=$(date)"
echo "config      : $CONFIG_NAME  overrides: ${OVERRIDES[*]:-none}"
echo "project     : $PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"
echo "gpus        : $GPUS   cpus: $SLURM_CPUS_PER_TASK"
echo "=========================================================="

# conda.sh, not bin/activate (which would consume $1) and not ~/.zshrc (this is bash).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; source "$PROJECT_ROOT/.env"; set +a
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"
export WANDB__SERVICE_WAIT=600
NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 2))
STORE="$DATA_ROOT/$STORE_REL"

[[ -f "$PROJECT_ROOT/$EXP_REL/conf/$CONFIG_NAME.yaml" ]] || {
  echo "ERROR: config not found: $PROJECT_ROOT/$EXP_REL/conf/$CONFIG_NAME.yaml" >&2; exit 1; }
[[ -f "$STORE/processed/lmdb/STORE.json" ]] || {
  echo "ERROR: pool store missing: $STORE/processed/lmdb/STORE.json" >&2; exit 1; }
N_INDEX=$(find "$STORE/data_module_cache" -maxdepth 1 -name 'index_seed_*.json' 2>/dev/null | wc -l)
[[ "$N_INDEX" -ge 1 ]] || {
  echo "ERROR: no split cache under $STORE/data_module_cache (warm_split_cache_030.py first)" >&2; exit 1; }
python - <<PY || exit 1
import torchcell
f = torchcell.__file__
assert f.startswith("$PROJECT_ROOT"), f"torchcell imports from {f}, not $PROJECT_ROOT"
print("torchcell:", f)
PY
echo "store       : $STORE ($N_INDEX split-cache index files)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "$PROJECT_ROOT/experiments" || exit 1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
torchrun --nnodes=1 --nproc_per_node=$GPUS \
  --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:$MASTER_PORT \
  --rdzv-id=$SLURM_JOB_ID --local-addr=127.0.0.1 \
  "$PROJECT_ROOT/$EXP_REL/scripts/equivariant_cell_graph_transformer.py" \
  --config-name "$CONFIG_NAME" \
  "dataset.root_rel=$STORE" trainer.devices=$GPUS \
  data_module.num_workers=$NUM_WORKERS "${OVERRIDES[@]}"
STATUS=$?
echo "finished=$(date) exit=$STATUS"
exit $STATUS
