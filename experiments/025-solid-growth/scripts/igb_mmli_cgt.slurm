#!/bin/bash
# experiments/025-solid-growth/scripts/igb_mmli_cgt.slurm
# [[experiments.025-solid-growth.scripts.igb_mmli_cgt]]
#
# IGB Biocluster `mmli` (compute-5-7, 4x A100) 4-GPU DDP launcher for ONE 025
# CellGraphTransformer arm on the 025 full build. Parameterized over the Hydra config name
# and free Hydra overrides, like delta_cgt.slurm:
#
#   sbatch -J 025-kl-fit1   experiments/025-solid-growth/scripts/igb_mmli_cgt.slurm cgt_s0_r_kl_fit_008
#   sbatch -J 025-kl-ctrl   experiments/025-solid-growth/scripts/igb_mmli_cgt.slurm cgt_s0_r_kl_000
#   sbatch -J 025-kl-fit0.1 experiments/025-solid-growth/scripts/igb_mmli_cgt.slurm cgt_s0_r_kl_fit_009
#
# THE DATA is the 025 build mirrored by sync_igb_025_build.sh: processed/ (554,075,586,560
# bytes of LMDB, byte-identical to GilaHyper's) plus the empty raw/lmdb stub that keeps
# PyG from trying to rebuild from a Neo4j this node cannot reach.
#
# THE CHECKOUT. IGB's main clone (~/projects/torchcell) sits on whatever branch the 019
# work last needed and must not be flipped under a running job, so this job runs from its
# OWN detached worktree. The worktree is prepared on the LOGIN node before sbatch: the
# compute nodes have no git at all (jobs 2390540-43 died at `git: command not found`, and
# rockylinux_9.sif carries none either, as the 019 wave-5 launcher records), and no
# network. The job only verifies the worktree exists and prints its HEAD from the gitdir.
#
#   git -C ~/projects/torchcell fetch origin feat/025-fitness-joint-head
#   # first time:
#   git -C ~/projects/torchcell worktree add --detach \
#       ~/projects/torchcell.worktrees/025-fitness-joint-head origin/feat/025-fitness-joint-head
#   # afterwards, to advance a queued job's code (never under a RUNNING job):
#   git -C ~/projects/torchcell.worktrees/025-fitness-joint-head checkout --detach origin/feat/025-fitness-joint-head
#
# A job reads whatever the worktree holds when it starts, so advance it only between jobs.
#
# LOG DIR must pre-exist: SLURM opens --output before this script runs.
#   mkdir -p /home/a-m/mjvolk3/scratch/torchcell/experiments/025-solid-growth/slurm/output
#
# WALL CLOCK. mmli has no time limit. The weekly note measured the KL arm at ~58 min per
# epoch on this node against 19.4 on GilaHyper; the replication peaked at epoch 14 and
# plateaued by 20-35, so 4 days (~99 epochs) reads the plateau with room to spare. The
# val-Pearson checkpoint keeps the best epoch either way.
#
# MEMORY matches gh_cgt.slurm: each rank opens the 13.5M-record build (~5.5 GB RSS)
# before its dataloader workers fork.
#SBATCH -p mmli
#SBATCH --mem=250g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=025-CGT
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/025-solid-growth/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/025-solid-growth/slurm/output/%x_%j.out

CONFIG_NAME="${1:?usage: sbatch igb_mmli_cgt.slurm <config_name> [hydra overrides...]}"
shift
OVERRIDES=("$@")

BRANCH="${BRANCH:-feat/025-fitness-joint-head}"
MAIN_REPO=/home/a-m/mjvolk3/projects/torchcell
PROJECT_ROOT="${PROJECT_ROOT:-/home/a-m/mjvolk3/projects/torchcell.worktrees/025-fitness-joint-head}"
IGB_DATA_ROOT=/home/a-m/mjvolk3/scratch/torchcell
BUILD_REL="data/torchcell/experiments/025-solid-growth/001-full-build"

module purge
source ~/.bashrc
module load singularity

echo "=========================================================="
echo "job         : $SLURM_JOB_ID  name=$SLURM_JOB_NAME  host=$(hostname)"
echo "config      : $CONFIG_NAME  overrides: ${OVERRIDES[*]:-none}"
echo "branch      : $BRANCH"
echo "DATA_ROOT   : $IGB_DATA_ROOT"
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
[[ -f "$PROJECT_ROOT/experiments/025-solid-growth/conf/$CONFIG_NAME.yaml" ]] || {
  echo "ERROR: config not found: $CONFIG_NAME" >&2; exit 1; }
for d in "$BUILD_REL/processed/lmdb/data.mdb" "$BUILD_REL/raw/lmdb" data/sgd/genome data/go/go.obo data/string data/tflink; do
  [[ -e "$IGB_DATA_ROOT/$d" ]] || { echo "ERROR: missing $IGB_DATA_ROOT/$d" >&2; exit 1; }
done
# The sequence-embedding arms (cgt_s0_q_kl_emb_017 and after) read pre-computed per-gene
# vectors; the builds must be on IGB scratch, since the compute node cannot download.
for d in fudt_embedding/processed/species_upstream.pt fudt_embedding/processed/species_downstream.pt \
         calm_embedding/processed/calm.pt protT5_embedding/processed/prot_t5_xl_uniref50_all.pt \
         random_embedding/processed/random_1000.pt; do
  [[ -e "$IGB_DATA_ROOT/data/scerevisiae/$d" ]] || { echo "ERROR: missing embedding build $IGB_DATA_ROOT/data/scerevisiae/$d" >&2; exit 1; }
done
mkdir -p "$IGB_DATA_ROOT/experiments/025-solid-growth/slurm/output"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "$PROJECT_ROOT/experiments" || exit 1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
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
  $PROJECT_ROOT/experiments/025-solid-growth/scripts/equivariant_cell_graph_transformer.py \
  --config-name $CONFIG_NAME \
  data_module.num_workers=\$NUM_WORKERS ${OVERRIDES[*]}
"
