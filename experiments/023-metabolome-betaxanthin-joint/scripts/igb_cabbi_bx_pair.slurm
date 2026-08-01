#!/bin/bash
# experiments/023-metabolome-betaxanthin-joint/scripts/igb_cabbi_bx_pair.slurm
# [[experiments.023-metabolome-betaxanthin-joint]]
#
# IGB cabbi -- ONE freed GPU, TWO runs: the metabolome->betaxanthin CONTROLLED PAIR.
#
#   cd <worktree-root>
#   sbatch experiments/023-metabolome-betaxanthin-joint/scripts/igb_cabbi_bx_pair.slurm
#
# WHY THESE TWO RUNS AND NOT ANY OTHER TWO. With a single opportunistic GPU the only question
# worth two runs is the one that is ANSWERED BY A DIFFERENCE rather than by a level:
#
#     bx_ctrl   active_heads=[betaxanthin]                 control
#     bx_m19    active_heads=[betaxanthin, mulleder19]     + the 19-AA metabolome head
#
# Both sit in the SAME grid cell (`s01_L2_maskon_lr0.001`) at the SAME seed, and
# `data_module.split_seed` pins the partition, so the two runs share an identical split AND an
# identical initialization. `bx_m19 - bx_ctrl` is then a PAIRED difference -- which is what
# makes n=1 per arm worth running at all. Two runs from DIFFERENT cells would each be a single
# draw against sigma = 0.030 and would answer nothing.
#
# Both arms are additionally restricted to the 4,432 genotypes carrying BOTH targets
# (`require_head_targets`), so the joint arm cannot win by seeing more rows.
#
# WHICH CELL, and why `s01_L2_maskon_lr0.001`:
#   L=2        metabolism's own measured preference (0.4301 vs 0.2764 at L=4, 0.2396 at L=6)
#   mask on    the architecture the Delta grid commits to
#   lr 1e-3    of the grid's two levels, the closer in log space to 4.06e-4 -- the lr of the
#              best betaxanthin run ever recorded -- and metabolism's own config default
# So this is the cell most likely to be the eventual winner, which is the right one to spend an
# opportunistic GPU on. `GRID_ONLY_SETTINGS` selects it FROM THE GRID rather than hand-writing
# overrides, so these two runs are directly comparable to the Delta results.
#
# IGB SPECIFICS, none of which apply on Delta:
#   * singularity + rockylinux_9.sif -- the IGB convention for this project.
#   * W&B is OFFLINE. IGB compute nodes have NO internet; an online run hangs on init. Sync
#     from the LOGIN node afterwards (command printed at the end of this script).
#   * PROJECT_ROOT is a DETACHED WORKTREE, not the main checkout: IGB's checkout sits on
#     someone else's branch with uncommitted work, and this job must not disturb it.
#
#SBATCH -p cabbi
#SBATCH --mem=80g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=023-bxm19-pair
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/023-metabolome-betaxanthin-joint/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/023-metabolome-betaxanthin-joint/slurm/output/%x_%j.out

# `set -u` is DELIBERATELY OFF across the cluster init. IGB's /etc/bashrc line 12 reads
# $BASHRCSOURCED before assigning it, so `source ~/.bashrc` under `set -u` aborts the job in
# 10 seconds with a single line of output and nothing else -- which is exactly how job 2332340
# died. Commit 93d5a7a9 fixed this for the other IGB launchers; this one reintroduced it.
# pipefail is safe and stays on; -u comes back after the init, where it is ours to enforce.
set -o pipefail
set +u
module purge
source ~/.bashrc
module load singularity
set -u

PROJECT_ROOT="${IGB_PROJECT_ROOT:-/home/a-m/mjvolk3/projects/torchcell-main}"
SCRATCH_ROOT=/home/a-m/mjvolk3/scratch/torchcell
EXP=023-metabolome-betaxanthin-joint
SETTING="${GRID_ONLY_SETTINGS:-s01_L2_maskon_lr0.001}"

OPTUNA_DIR="$SCRATCH_ROOT/experiments/$EXP/optuna"
mkdir -p "$OPTUNA_DIR" "$SCRATCH_ROOT/experiments/$EXP/"{slurm/output,results}

RUNNER="$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/metabolism_grid_runner.py"
CONF="$PROJECT_ROOT/experiments/$EXP/conf"

echo "=========================================================="
echo "host      : $(hostname)   job=${SLURM_JOB_ID:-none}"
echo "project   : $PROJECT_ROOT   ($(git -C "$PROJECT_ROOT" log --oneline -1 2>/dev/null))"
echo "DATA_ROOT : $SCRATCH_ROOT"
echo "setting   : $SETTING   (bx_ctrl vs bx_m19, paired)"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=========================================================="

# TWO WORKERS ON ONE GPU. They are independent processes on the same card, which is fine --
# the pairing is by (setting, seed), not by co-residency. If the log shows CUDA OOM, resubmit
# with IGB_SERIAL=1 and they run back to back instead (each ~2x faster, same total wall).
SERIAL="${IGB_SERIAL:-0}"

singularity exec --nv "$PROJECT_ROOT/rockylinux_9.sif" bash -c "
  source \$HOME/miniconda3/bin/activate
  conda activate torchcell
  export DATA_ROOT='$SCRATCH_ROOT'
  export EXPERIMENT_ROOT='$PROJECT_ROOT/experiments'
  export PYTHONPATH='$PROJECT_ROOT'
  export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # NO INTERNET on IGB compute nodes -- an online W&B run blocks on init until it times out.
  export WANDB_MODE=offline
  export WANDB_DIR='$SCRATCH_ROOT/wandb-experiments'
  export GRID_CONF_DIR='$CONF'
  export GRID_BASE_CONFIG=delta_grid_bx_pair
  export GRID_ONLY_SETTINGS='$SETTING'
  export GRID_ROUNDS=1
  # 0 dataloader workers: the dataset is small and spawn re-imports the whole stack per worker
  # off GPFS, which is the same cost that made Delta's sanity check take 38 minutes.
  export NUM_WORKERS=0
  # No deadline -- the epoch cap (1000) is the stop, and 24 h is ample for two runs.

  for arm in bx_ctrl bx_m19; do
    ARM=\$arm OPTUNA_STORAGE=\"sqlite:///$OPTUNA_DIR/optuna_igb_\${arm}.db\" \
      python '$RUNNER' --create-only
  done

  if [ '$SERIAL' = 1 ]; then
    for arm in bx_ctrl bx_m19; do
      ARM=\$arm OPTUNA_STORAGE=\"sqlite:///$OPTUNA_DIR/optuna_igb_\${arm}.db\" \
        OPTUNA_WORKER_ID=0 python '$RUNNER'
    done
  else
    ARM=bx_ctrl OPTUNA_STORAGE=\"sqlite:///$OPTUNA_DIR/optuna_igb_bx_ctrl.db\" \
      OPTUNA_WORKER_ID=0 python '$RUNNER' &
    ARM=bx_m19  OPTUNA_STORAGE=\"sqlite:///$OPTUNA_DIR/optuna_igb_bx_m19.db\" \
      OPTUNA_WORKER_ID=1 python '$RUNNER' &
    wait
  fi
"

echo "=========================================================="
echo "DONE. W&B ran OFFLINE -- sync from a LOGIN node (compute nodes have no internet):"
echo "  wandb sync $SCRATCH_ROOT/wandb-experiments/*/wandb/offline-run-*"
echo "=========================================================="
