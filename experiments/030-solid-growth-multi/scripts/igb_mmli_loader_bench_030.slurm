#!/bin/bash
# experiments/030-solid-growth-multi/scripts/igb_mmli_loader_bench_030.slurm
# [[experiments.030-solid-growth-multi.scripts.igb_mmli_loader_bench_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/igb_mmli_loader_bench_030.slurm
#
# The loader throughput sweep (loader_throughput_030.py) on the mmli node, CPU only, under
# the node's real conditions (whatever else is running on compute-5-7). Same checkout,
# container and environment as igb_mmli_cgt_030.slurm. Two optional parts:
#
#   PROCS=N        run N copies of the sweep at once (each with its own workers), to put
#                  N x workers processes on the GPFS file the way N DDP ranks do; each
#                  copy writes its own results file (label gets a _pN suffix)
#   STAGE_SHM=1    also stage the pool store $POOL_STORE_REL onto /dev/shm and add it to
#                  bench.roots, so the same job compares GPFS full build vs RAM pool store
#
#   mkdir -p /home/a-m/mjvolk3/scratch/torchcell/experiments/030-solid-growth-multi/slurm/output
#   sbatch --export=ALL,PROJECT_ROOT=<worktree>,PROCS=2 -J 030-bench-gpfs \
#       experiments/030-solid-growth-multi/scripts/igb_mmli_loader_bench_030.slurm \
#       cgt_030_s3_q_tok_embfit_004 '+bench.workers=[14,8,4,2]' +bench.batches=60 +bench.warmup=15 +bench.label=gpfs
#SBATCH -p mmli
#SBATCH --mem=120g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=030-bench
#SBATCH --time=08:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/030-solid-growth-multi/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/030-solid-growth-multi/slurm/output/%x_%j.out

CONFIG_NAME="${1:?usage: sbatch igb_mmli_loader_bench_030.slurm <config_name> [hydra overrides...]}"
shift
OVERRIDES=("$@")
PROCS="${PROCS:-1}"
STAGE_SHM="${STAGE_SHM:-0}"

MAIN_REPO=/home/a-m/mjvolk3/projects/torchcell
PROJECT_ROOT="${PROJECT_ROOT:-/home/a-m/mjvolk3/projects/torchcell.worktrees/030-per-entry-dataset-token}"
IGB_DATA_ROOT=/home/a-m/mjvolk3/scratch/torchcell
BUILD_REL="data/torchcell/experiments/030-solid-growth-multi/001-multi-build"
POOL_STORE_REL="${POOL_STORE_REL:-${BUILD_REL}-s3q}"
EXP_REL="experiments/030-solid-growth-multi"

module purge
# shellcheck disable=SC1090
source ~/.bashrc
module load singularity

echo "job=$SLURM_JOB_ID name=$SLURM_JOB_NAME host=$(hostname) cpus=$SLURM_CPUS_PER_TASK procs=$PROCS stage_shm=$STAGE_SHM"
echo "config=$CONFIG_NAME overrides: ${OVERRIDES[*]:-none}"
GITDIR="$(sed -n 's/^gitdir: //p' "$PROJECT_ROOT/.git")"
echo "worktree=$PROJECT_ROOT commit=$(cat "$GITDIR/HEAD")"
[[ -f "$PROJECT_ROOT/.env" ]] || cp "$MAIN_REPO/.env" "$PROJECT_ROOT/.env"
echo "--- node"; squeue -w "$(hostname)" -o "%.10i %.9u %.20j %.3t %.12M %.4C %.8m %b"
top -bn1 | head -4 | tail -2

ROOTS="[config]"
if [[ "$STAGE_SHM" == "1" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/$EXP_REL/scripts/stage_pool_store_shm.sh"
  SHM_ROOT="/dev/shm/$USER/$(basename "$POOL_STORE_REL")"
  stage_pool_store "$IGB_DATA_ROOT/$POOL_STORE_REL" "$SHM_ROOT" || exit 1
  ROOTS="[config,$SHM_ROOT]"
fi

cd "$PROJECT_ROOT/experiments" || exit 1
run_one() {  # run_one <proc index>; with PROCS > 1 the label gets a _p<index> suffix
  local p="$1" suffix="" ovr="${OVERRIDES[*]}"
  if [[ "$PROCS" -gt 1 ]]; then
    suffix="_p$p"
    ovr="${ovr//+bench.label=/+bench.label=p${p}_}"
  fi
  singularity exec --nv "$MAIN_REPO/rockylinux_9.sif" bash -c "
source \$HOME/miniconda3/bin/activate
conda activate torchcell
export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\"
export DATA_ROOT=$IGB_DATA_ROOT
export EXPERIMENT_ROOT=$PROJECT_ROOT/experiments
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
ls -d /dev/shm >/dev/null && echo 'container sees /dev/shm'
python $PROJECT_ROOT/$EXP_REL/scripts/loader_throughput_030.py \
  --config-name $CONFIG_NAME '+bench.roots=$ROOTS' $ovr \
  hydra.run.dir=$IGB_DATA_ROOT/$EXP_REL/hydra/bench_${SLURM_JOB_ID}$suffix 2>&1 | sed 's/^/[p$p] /'
"
}
if [[ "$PROCS" -gt 1 ]]; then
  for ((p = 0; p < PROCS; p++)); do
    run_one "$p" &
    sleep 20
  done
  wait
else
  run_one 0
fi
echo "finished=$(date)"
