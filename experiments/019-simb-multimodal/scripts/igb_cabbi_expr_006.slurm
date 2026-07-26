#!/bin/bash
# experiments/019-simb-multimodal/scripts/igb_cabbi_expr_006.slurm
# IGB cabbi half (2x RTX6000) of the POOLED expression-only _006 sweep.
#
# POOLED WITH CABBI. This job and igb_cabbi_expr_006.slurm share ONE Optuna study: same
# storage path, same study name. Both IGB partitions mount the same GPFS filesystem
# (`df -T ~/scratch` -> igbdata gpfs), and GPFS -- unlike NFS -- provides coherent POSIX
# byte-range locking across nodes, which is what concurrent SQLite needs. Coordination
# overhead is negligible: a handful of transactions per trial against trials that run
# tens of minutes to hours. Total 6 GPU-pinned workers (4 here + 2 on cabbi).
#
# The study therefore MIXES HARDWARE (A100 here, RTX6000 on cabbi). Optuna does not care,
# but trial durations differ, so `gpu_type` is logged to W&B config for every run.
#
#   cd <worktree-root> && sbatch experiments/019-simb-multimodal/scripts/igb_cabbi_expr_006.slurm
#
#SBATCH -p cabbi
#SBATCH --mem=80g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --job-name=019-expr006-cabbi
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out

module purge
source ~/.bashrc
module load singularity

PROJECT_ROOT=/home/a-m/mjvolk3/projects/torchcell
SCRATCH_ROOT=/home/a-m/mjvolk3/scratch/torchcell
OPTUNA_DIR=$SCRATCH_ROOT/experiments/019-simb-multimodal/optuna
mkdir -p "$OPTUNA_DIR"
mkdir -p "$SCRATCH_ROOT/experiments/019-simb-multimodal/slurm/output"

# Up to 3 days, resubmitting across walltime boundaries.
export DEADLINE="2026-07-29T18:00:00"
source "$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/requeue_until.sh"
requeue_arm_trap "$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/igb_cabbi_expr_006.slurm"

cd "$PROJECT_ROOT/experiments" || exit 1
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 2))

singularity exec --nv "$PROJECT_ROOT/rockylinux_9.sif" bash -c "
source \$HOME/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH_ROOT/experiments/019-simb-multimodal
export WANDB__SERVICE_WAIT=600

export CONDITION=expr
export JOINT_BASE_CONFIG=cgt_expr_006
export OPTUNA_STORAGE='sqlite:///$OPTUNA_DIR/optuna_019_expr_006.db'
export OPTUNA_STUDY_NAME=expr_006
export OPTUNA_N_TRIALS=\${OPTUNA_N_TRIALS:-40}
export WANDB_PROJECT_SUFFIX=v6
export NUM_WORKERS=$NUM_WORKERS

SWEEP=$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py

# The mmli half pre-creates the study; --create-only here too is idempotent and
# covers the case where cabbi is scheduled first.
python \$SWEEP --create-only

CUDA_VISIBLE_DEVICES=0 OPTUNA_WORKER_ID=4 python \$SWEEP &
CUDA_VISIBLE_DEVICES=1 OPTUNA_WORKER_ID=5 python \$SWEEP &
wait
" &
# Background the singularity job so the batch shell stays responsive to SIGUSR1.
wait

requeue_if_before_deadline "$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/igb_cabbi_expr_006.slurm"
