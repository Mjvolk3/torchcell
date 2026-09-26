#!/bin/bash
# experiments/030-solid-growth-multi/scripts/igb_mmli_cgt_030.slurm
# [[experiments.030-solid-growth-multi.scripts.igb_mmli_cgt_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/igb_mmli_cgt_030.slurm
#
# WHY. IGB Biocluster `mmli` (compute-5-7, 4x A100) 4-GPU DDP launcher for ONE 030
# CellGraphTransformer arm (per-entry dataset token) on the 030 multi-measurement build.
# The 025 launcher (experiments/025-solid-growth/scripts/igb_mmli_cgt.slurm) with the
# branch, worktree, build and conf paths pointed at 030 and a stricter preflight.
# Parameterized over the Hydra config name and free Hydra overrides:
#
#   sbatch --export=ALL,PROJECT_ROOT=<worktree> -J 030-r1w1-s0 \
#       experiments/030-solid-growth-multi/scripts/igb_mmli_cgt_030.slurm cgt_030_s3_r_tok_embfit_001 +seed=0
#
# THE DATA is the 030 build mirrored by sync_igb_030_build.sh: processed/ (955,268,878,336
# bytes of LMDB plus the index JSONs, byte-identical to GilaHyper's), the empty raw/lmdb
# stub that keeps PyG from rebuilding from a Neo4j this node cannot reach, AND
# data_module_cache/ with the split indices. The preflight refuses to start without the
# cache: the datamodule's cache check is a bare osp.exists with no lock, so four DDP
# ranks would race to compute the same split over 13.5M records.
#
# THE CHECKOUT. IGB's main clone (~/projects/torchcell) must not be flipped under a
# running job, so this job runs from its OWN detached worktree, prepared on the LOGIN node
# before sbatch: the compute nodes have no git at all (025 jobs 2390540-43 died at
# `git: command not found`, and rockylinux_9.sif carries none either) and no network. The
# job only verifies the worktree exists and prints its HEAD from the gitdir. One detached
# worktree per running commit: a job reads whatever the worktree holds when it starts
# (`spawn` dataloader workers re-import live code), so advance a worktree only between
# jobs, never under a running one.
#
#   git -C ~/projects/torchcell fetch origin feat/030-per-entry-dataset-token
#   # first time:
#   git -C ~/projects/torchcell worktree add --detach \
#       ~/projects/torchcell.worktrees/030-per-entry-dataset-token origin/feat/030-per-entry-dataset-token
#   # afterwards, to advance a QUEUED job's code (never under a RUNNING job):
#   git -C ~/projects/torchcell.worktrees/030-per-entry-dataset-token checkout --detach origin/feat/030-per-entry-dataset-token
#
# LOG DIR must pre-exist: SLURM opens --output before this script runs.
#   mkdir -p /home/a-m/mjvolk3/scratch/torchcell/experiments/030-solid-growth-multi/slurm/output
#
# WALL CLOCK. mmli has no partition limit; 4 days as in 025. Epoch budget vs the clock
# (plan gotcha 3): LMDB reads stay 1x the 025 build but the readout runs 4.3x rows and
# the metric collections grow, so max_epochs is sized from the MEASURED epoch-1 wall time
# of the first cell and re-sized if epoch 1 exceeds 90 min. Hypothesis (untested): cold
# epoch 0 takes 5 to 8 h, warm epochs 45 to 60 min.
#
# MEMORY (plan gotcha 4). --mem=250g is the 025 value, sized on the 554 GB 025 LMDB; each
# rank opens the build before its dataloader workers fork, and per-rank RSS grows with the
# map. Read MaxRSS after the first cell (`sacct -j <id> --format=MaxRSS,Elapsed`) and
# resize before seeds 1 and 2; never launch three seeds blind.
#SBATCH -p mmli
#SBATCH --mem=250g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=030-r1w1-s0
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/030-solid-growth-multi/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/030-solid-growth-multi/slurm/output/%x_%j.out

CONFIG_NAME="${1:?usage: sbatch igb_mmli_cgt_030.slurm <config_name> [hydra overrides...]}"
shift
OVERRIDES=("$@")

BRANCH="${BRANCH:-feat/030-per-entry-dataset-token}"
MAIN_REPO=/home/a-m/mjvolk3/projects/torchcell
PROJECT_ROOT="${PROJECT_ROOT:-/home/a-m/mjvolk3/projects/torchcell.worktrees/030-per-entry-dataset-token}"
IGB_DATA_ROOT=/home/a-m/mjvolk3/scratch/torchcell
BUILD_REL="data/torchcell/experiments/030-solid-growth-multi/001-multi-build"
EXP_REL="experiments/030-solid-growth-multi"

module purge
# shellcheck disable=SC1090
source ~/.bashrc
module load singularity

echo "=========================================================="
echo "job         : $SLURM_JOB_ID  name=$SLURM_JOB_NAME  host=$(hostname)"
echo "config      : $CONFIG_NAME  overrides: ${OVERRIDES[*]:-none}"
echo "branch      : $BRANCH"
echo "worktree    : $PROJECT_ROOT"
echo "DATA_ROOT   : $IGB_DATA_ROOT"
echo "build       : $IGB_DATA_ROOT/$BUILD_REL"
echo "=========================================================="

# ---- Checkout (prepared on the login node; compute nodes have no git) ----
[[ -d "$PROJECT_ROOT" && -f "$PROJECT_ROOT/.git" ]] || {
  echo "ERROR: worktree $PROJECT_ROOT missing; on the login node run:" >&2
  echo "       git -C $MAIN_REPO fetch origin $BRANCH" >&2
  echo "       git -C $MAIN_REPO worktree add --detach $PROJECT_ROOT origin/$BRANCH" >&2
  exit 1; }
# A worktree's .git is a one-line file `gitdir: <path>`; a detached HEAD there is the SHA.
GITDIR="$(sed -n 's/^gitdir: //p' "$PROJECT_ROOT/.git")"
echo "commit      : $(cat "$GITDIR/HEAD")"
# The worktree has no .env (untracked); the main clone's holds IGB's roots. The exports
# below win anyway, since load_dotenv never overrides a set variable.
[[ -f "$PROJECT_ROOT/.env" ]] || cp "$MAIN_REPO/.env" "$PROJECT_ROOT/.env"

# ---- Preflight ----
[[ -f "$PROJECT_ROOT/$EXP_REL/scripts/equivariant_cell_graph_transformer.py" ]] || {
  echo "ERROR: training script not found: $PROJECT_ROOT/$EXP_REL/scripts/equivariant_cell_graph_transformer.py" >&2; exit 1; }
[[ -f "$PROJECT_ROOT/$EXP_REL/conf/$CONFIG_NAME.yaml" ]] || {
  echo "ERROR: config not found: $PROJECT_ROOT/$EXP_REL/conf/$CONFIG_NAME.yaml" >&2; exit 1; }
# The build: LMDB, the two index JSONs the subset definitions read, the raw stub, and the
# shared genome / GO / STRING / TFLink inputs the cell graph is built from.
for d in "$BUILD_REL/processed/lmdb/data.mdb" \
         "$BUILD_REL/processed/perturbation_count_index.json" \
         "$BUILD_REL/processed/phenotype_label_index.json" \
         "$BUILD_REL/raw/lmdb" \
         data/sgd/genome data/go/go.obo data/string data/tflink; do
  [[ -e "$IGB_DATA_ROOT/$d" ]] || { echo "ERROR: missing $IGB_DATA_ROOT/$d" >&2; exit 1; }
done
# The split cache is REQUIRED (plan gotcha 1). Without it every one of the four DDP ranks
# runs the same 13.5M-record scan to compute the split, with no lock between them. Warm it
# on GilaHyper (the smoke job), rerun sync_igb_030_build.sh, then submit.
CACHE_DIR="$IGB_DATA_ROOT/$BUILD_REL/data_module_cache"
if [[ ! -d "$CACHE_DIR" ]]; then
  echo "ERROR: split cache missing: $CACHE_DIR" >&2
  echo "       Refusing to start: four DDP ranks would race to compute a split over 13.5M records." >&2
  echo "       Warm the cache on GilaHyper (smoke job), rerun sync_igb_030_build.sh, resubmit." >&2
  exit 1
fi
N_INDEX=$(find "$CACHE_DIR" -maxdepth 1 -name 'index_seed_*.json' | wc -l)
if [[ "$N_INDEX" -eq 0 ]]; then
  echo "ERROR: $CACHE_DIR holds no index_seed_*.json" >&2
  echo "       Refusing to start: four DDP ranks would race to compute a split over 13.5M records." >&2
  echo "       Warm the cache on GilaHyper (smoke job), rerun sync_igb_030_build.sh, resubmit." >&2
  exit 1
fi
echo "split cache : $N_INDEX index_seed_*.json in $CACHE_DIR"
find "$CACHE_DIR" -maxdepth 1 -name 'index_seed_*.json' -printf '              %f\n'
# The composite embedding arm reads four pre-computed per-gene vector builds
# (fudt upstream + downstream, CaLM, ProtT5 all); the directory names are the ones the
# 025 launcher checks. They must be on IGB scratch, since the compute node cannot download.
for d in fudt_embedding/processed/species_upstream.pt fudt_embedding/processed/species_downstream.pt \
         calm_embedding/processed/calm.pt protT5_embedding/processed/prot_t5_xl_uniref50_all.pt; do
  [[ -e "$IGB_DATA_ROOT/data/scerevisiae/$d" ]] || { echo "ERROR: missing embedding build $IGB_DATA_ROOT/data/scerevisiae/$d" >&2; exit 1; }
done
mkdir -p "$IGB_DATA_ROOT/$EXP_REL/slurm/output"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "$PROJECT_ROOT/experiments" || exit 1

MASTER_ADDR=$(hostname); export MASTER_ADDR
MASTER_PORT=$(shuf -i 10000-65500 -n 1); export MASTER_PORT
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 2))

singularity exec --nv "$MAIN_REPO/rockylinux_9.sif" bash -c "
source \$HOME/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\"
export DATA_ROOT=$IGB_DATA_ROOT
export EXPERIMENT_ROOT=$PROJECT_ROOT/experiments
export ASSET_IMAGES_DIR=$PROJECT_ROOT/notes/assets/images
# Compute nodes have no outbound network: offline, then igb_login_wandb_sync.sh later.
export WANDB_MODE=offline
export WANDB__SERVICE_WAIT=600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

python - <<PY || exit 1
import torchcell
assert torchcell.__file__.startswith('$PROJECT_ROOT'), torchcell.__file__
print('torchcell:', torchcell.__file__)
PY

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  $PROJECT_ROOT/$EXP_REL/scripts/equivariant_cell_graph_transformer.py \
  --config-name $CONFIG_NAME \
  data_module.num_workers=\$NUM_WORKERS ${OVERRIDES[*]}
"
