#!/bin/bash
# experiments/019-simb-multimodal/scripts/cabbi_expr_embed_005.slurm
# IGB `cabbi` EXPRESSION arm of the _005 NODE-EMBEDDING (cold-start) study.
#
# NOT a control arm. It sweeps `node_embeddings` and `learnable_embedding` exactly like the
# morph and joint arms -- the axis is sampled unconditionally by the driver, not gated on
# CONDITION. (The "control" label belonged to the _003/_004 DECODER study, where expression
# had no `decoder` lever and was predicted to stay flat at its noise ceiling. That framing
# does not carry over: the cold-start diagnosis applies to expression too, which shows TRAIN
# per-feature Pearson 0.46 vs VAL 0.067 -- memorizing as well, just less than morphology's
# 0.66/0.023.)
#
# Its secondary value is as a SATURATION REFERENCE: expression's measured noise ceiling is
# ~0.09-0.11 and its best observed val is 0.0905, so it is essentially at ceiling. If content
# embeddings push it ABOVE ~0.11, the ceiling estimate is too low and the "expression is
# saturated" premise that anchors the morphology comparison needs revisiting.
#
# 2 GPUs -> 2 GPU-pinned Optuna workers on one shared SQLite study; cabbi allocation stays on
# one node (partition rule). IGB compute is OFFLINE -> W&B offline, `wandb sync` from a LOGIN
# node later. Self-resubmits until the deadline.
#
#   ssh mjvolk3@biologin.igb.illinois.edu \
#   'sbatch /home/a-m/mjvolk3/projects/torchcell/experiments/019-simb-multimodal/scripts/cabbi_expr_embed_005.slurm'
#
#SBATCH -p cabbi
#SBATCH --mem=120g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --job-name=019-expr-embed005
#SBATCH --time=5-00:00:00
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
mkdir -p "$SCRATCH_ROOT/experiments/019-simb-multimodal/wandb"

# Run until this wall-clock deadline, resubmitting across walltime boundaries.
export DEADLINE="2026-07-29T10:00:00"
source "$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/requeue_until.sh"
REQUEUE_SELF="$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/cabbi_expr_embed_005.slurm"
requeue_arm_trap "$REQUEUE_SELF"

cd "$PROJECT_ROOT/experiments" || exit 1
# 8 cpus / 2 workers = 4 dataloader workers per trial.
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 2))

singularity exec --nv "$PROJECT_ROOT/rockylinux_9.sif" bash -c "
source \$HOME/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

# IGB compute is OFFLINE -> W&B offline, synced from a login node later.
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH_ROOT/experiments/019-simb-multimodal
export WANDB__SERVICE_WAIT=600

export CONDITION=expr
export JOINT_BASE_CONFIG=cgt_embed_005
export OPTUNA_STORAGE='sqlite:///$OPTUNA_DIR/optuna_019_expr_005.db'
export OPTUNA_STUDY_NAME=expr_005
export OPTUNA_N_TRIALS=\${OPTUNA_N_TRIALS:-30}
export WANDB_PROJECT_SUFFIX=v5
export NUM_WORKERS=$NUM_WORKERS

SWEEP=$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py

# Pre-create the study in ONE process so the workers only load_if_exists (fresh-DB DDL race).
python \$SWEEP --create-only

CUDA_VISIBLE_DEVICES=0 OPTUNA_WORKER_ID=0 python \$SWEEP &
CUDA_VISIBLE_DEVICES=1 OPTUNA_WORKER_ID=1 python \$SWEEP &
wait
" &
# `wait` on the singularity job in the BACKGROUND so the batch shell stays responsive to
# SIGUSR1 (a foreground child would otherwise swallow the signal until it exits).
wait

# Clean exit before the deadline (workers finished their trial budget): keep going.
requeue_if_before_deadline "$REQUEUE_SELF"
