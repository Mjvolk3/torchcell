#!/bin/bash
# experiments/023-metabolome-betaxanthin-joint/scripts/igb_cabbi_bx_aa.slurm
# [[experiments.023-metabolome-betaxanthin-joint]]
#
# IGB cabbi -- TWO freed GPUs, SIX concurrent workers: replicate the amino-acid effect.
#
#   cd <worktree-root>
#   sbatch experiments/023-metabolome-betaxanthin-joint/scripts/igb_cabbi_bx_aa.slurm
#
# WHAT THIS ROUND IS. `bx_aa` freezes the whole grid at Delta cell `s04_L6_maskon_lr0.0001`
# -- the one cell where the auxiliary metabolome head was measured to HELP betaxanthin
# (+0.0203 paired, at a matched budget of 333 epochs) -- and varies only the auxiliary panel
# and the seed. Rationale in full: `conf/igb_bx_aa.yaml`.
#
#   3 settings (none / m19 / tyr) x 8 seeds = 24 cells enqueued
#
# ROUND-MAJOR ENQUEUE IS LOAD-BEARING. `_enqueue_all` walks rounds outer / settings inner, so
# the queue is (seed 0: none,m19,tyr), (seed 1: none,m19,tyr), ... If the wall clock runs out
# we lose WHOLE SEEDS from the tail, never a control without its treatments. A truncated
# seed-major queue would leave paired differences with no partner, which is not a smaller
# result -- it is an unreadable one.
#
# SIX WORKERS ON TWO GPUS -- three per card, and the arithmetic:
#   * measured ~11.4 GB per worker (22,705 MB total for 2 co-resident workers on Delta
#     gpub035; 22,824 MB for 2 on this very node, compute-3-3). 3 x 11.4 = 34 GB of 46 GB.
#   * measured 27-31 s/epoch at TWO workers/GPU, with GPU utilization only 50-65% -- i.e.
#     there is compute headroom, not just memory headroom.
#   * epoch time at THREE workers/GPU is NOT measured. ~40 s is an extrapolation. At 40 s a
#     600-epoch cell costs 6.7 h and ~18 cells fit in 20 h (6 seeds); at a pessimistic 50 s,
#     ~14 cells (4-5 seeds). Either way the queue is drained in seed order, so the result is
#     fewer complete seeds rather than a broken design.
#   * a CUDA OOM fails ONE trial (optuna catches it) rather than the job, and FAILED cells
#     are re-enqueued by `_enqueue_all` on a resubmit. Set IGB_WORKERS_PER_GPU=2 to back off.
#
# THE DEADLINE GATES CLAIMING, NOT TRAINING. `GRID_NO_TIME_CAP=1` removes the per-trial time
# cap so every cell that starts runs the full 600 epochs; `GRID_MIN_TRIAL_S` then stops a
# worker from claiming a cell it cannot finish. Together: identical epoch budgets within
# every pair, and no half-run cell at the end. See `_trial_time_budget`.
#
# IGB SPECIFICS, none of which apply on Delta:
#   * singularity + rockylinux_9.sif -- the IGB convention for this project.
#   * W&B is OFFLINE. IGB compute nodes have NO internet; an online run hangs on init. Sync
#     from the LOGIN node afterwards (command printed at the end of this script).
#   * PROJECT_ROOT is a DETACHED WORKTREE, not the main checkout: IGB's checkout sits on
#     someone else's branch with uncommitted work, and this job must not disturb it.
#
#SBATCH -p cabbi
#SBATCH --mem=160g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --job-name=023-bxaa
#SBATCH --time=22:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/023-metabolome-betaxanthin-joint/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/023-metabolome-betaxanthin-joint/slurm/output/%x_%j.out

# `set -u` is DELIBERATELY OFF across the cluster init. IGB's /etc/bashrc line 12 reads
# $BASHRCSOURCED before assigning it, so `source ~/.bashrc` under `set -u` aborts the job in
# 10 seconds with a single line of output and nothing else -- which is exactly how job 2332340
# died. pipefail is safe and stays on; -u comes back after the init.
set -o pipefail
set +u
module purge
source ~/.bashrc
module load singularity
set -u

PROJECT_ROOT="${IGB_PROJECT_ROOT:-/home/a-m/mjvolk3/projects/torchcell-main}"
SCRATCH_ROOT=/home/a-m/mjvolk3/scratch/torchcell
EXP=023-metabolome-betaxanthin-joint
WORKERS_PER_GPU="${IGB_WORKERS_PER_GPU:-3}"
N_GPUS="${IGB_N_GPUS:-2}"
ROUNDS="${GRID_ROUNDS:-8}"

# Stop claiming a cell we cannot finish: a 600-epoch cell at the pessimistic 50 s/epoch is
# 8.3 h, so a worker with less than that left idles instead of starting a run that slurm
# would kill mid-training. This is the ONLY thing DEADLINE does here -- GRID_NO_TIME_CAP
# below removes its effect on the training length itself.
MIN_TRIAL_S="${GRID_MIN_TRIAL_S:-30600}"
# 21.5 h from now, half an hour inside the 22 h wall so the last teardown lands in-allocation.
DEADLINE_EPOCH=$(( $(date +%s) + 77400 ))

OPTUNA_DIR="$SCRATCH_ROOT/experiments/$EXP/optuna"
mkdir -p "$OPTUNA_DIR" "$SCRATCH_ROOT/experiments/$EXP/"{slurm/output,results}

RUNNER="$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/metabolism_grid_runner.py"
CONF="$PROJECT_ROOT/experiments/$EXP/conf"

echo "=========================================================="
echo "host       : $(hostname)   job=${SLURM_JOB_ID:-none}"
echo "project    : $PROJECT_ROOT   ($(git -C "$PROJECT_ROOT" log --oneline -1 2>/dev/null))"
echo "DATA_ROOT  : $SCRATCH_ROOT"
echo "arm        : bx_aa   3 settings (none/m19/tyr) x $ROUNDS seeds, round-major"
echo "workers    : $N_GPUS GPUs x $WORKERS_PER_GPU = $(( N_GPUS * WORKERS_PER_GPU ))"
echo "min trial  : $(( MIN_TRIAL_S / 3600 )) h    deadline: $(date -d "@$DEADLINE_EPOCH")"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=========================================================="

singularity exec --nv "$PROJECT_ROOT/rockylinux_9.sif" bash -c "
  set -o pipefail
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
  export ARM=bx_aa
  export GRID_CONF_DIR='$CONF'
  export GRID_BASE_CONFIG=igb_bx_aa
  export GRID_ROUNDS='$ROUNDS'
  export GRID_MANIFEST_DIR='$SCRATCH_ROOT/experiments/$EXP/results'
  # SHARDED: each worker owns a disjoint slice in its OWN study and its OWN SQLite file.
  # A SHARED queue does not work here -- optuna's WAITING-trial pop races across processes on
  # SQLite. Measured on job 2332400 with this exact launcher: w0 and w1 both claimed trial 0,
  # w2 and w3 both claimed trial 1, so six workers ran three distinct cells. The same race is
  # in the Delta grids (one setting ran four times). Per-worker files remove the shared
  # object entirely; see GRID_SHARD_COUNT / _owns_cell in the runner.
  export GRID_SHARD_COUNT=$(( N_GPUS * WORKERS_PER_GPU ))
  # 0 dataloader workers: the dataset is small and spawn re-imports the whole stack per worker
  # off GPFS, which is the same cost that made Delta's sanity check take 38 minutes.
  export NUM_WORKERS=0
  export GRID_DEADLINE_EPOCH='$DEADLINE_EPOCH'
  export GRID_MIN_TRIAL_S='$MIN_TRIAL_S'
  # Fixed 600 epochs for EVERY cell that starts -- see the header and _trial_time_budget.
  export GRID_NO_TIME_CAP=1

  # Build each worker's queue ONCE, SEQUENTIALLY, before any GPU work. Sequential because
  # each --create-only writes its own fresh SQLite file and a fresh file's DDL should not be
  # raced. This is also the import-time gate: a mis-specified arm (missing REQUIRED_KEYS,
  # drifted amino-acid list, unknown panel, shard count not dividing the settings) exits here
  # in seconds rather than inside the first trial of a 22 h allocation.
  for w in \$(seq 0 \$(( GRID_SHARD_COUNT - 1 ))); do
    OPTUNA_WORKER_ID=\$w \
      OPTUNA_STORAGE=\"sqlite:///$OPTUNA_DIR/optuna_igb_bx_aa_w\${w}.db\" \
      python '$RUNNER' --create-only || exit 1
  done

  w=0
  for g in \$(seq 0 \$(( $N_GPUS - 1 ))); do
    for k in \$(seq 1 $WORKERS_PER_GPU); do
      CUDA_VISIBLE_DEVICES=\$g OPTUNA_WORKER_ID=\$w \
        OPTUNA_STORAGE=\"sqlite:///$OPTUNA_DIR/optuna_igb_bx_aa_w\${w}.db\" \
        python '$RUNNER' &
      w=\$(( w + 1 ))
      # Stagger starts: six processes opening the same LMDB at once is a needless contention
      # spike at t=0.
      sleep 20
    done
  done
  wait
"

echo "=========================================================="
echo "DONE. W&B ran OFFLINE -- sync from a LOGIN node (compute nodes have no internet):"
echo "  wandb sync $SCRATCH_ROOT/wandb-experiments/*/wandb/offline-run-*"
echo "=========================================================="
