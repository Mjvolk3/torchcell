#!/bin/bash
# experiments/019-simb-multimodal/scripts/igb_mmli_optuna_morph_6h.slurm
# IGB mmli MORPHOLOGY Optuna sweep, PEAK objective (fresh study morph_peak_000), 6h cap.
# Drives optuna_joint_sweep.py CONDITION=morph on mmli_morph_optuna_000 (full Ohya, global head).
# Singularity + W&B offline (IGB compute is offline; `wandb sync` from a login node later).
#
#   sbatch experiments/019-simb-multimodal/scripts/igb_mmli_optuna_morph_6h.slurm
#
#SBATCH -p mmli
#SBATCH --mem=120g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=019-morph-peak
#SBATCH --time=24:00:00
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

cd "$PROJECT_ROOT/experiments" || exit 1
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 4))

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

export CONDITION=morph
export JOINT_BASE_CONFIG=mmli_morph_optuna_000
export OPTUNA_STORAGE=\"sqlite:///$OPTUNA_DIR/optuna_019_morph_peak.db\"
export OPTUNA_STUDY_NAME=morph_peak_000
export OPTUNA_N_TRIALS=30
export NUM_WORKERS=$NUM_WORKERS

SWEEP=$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py
python \$SWEEP --create-only

CUDA_VISIBLE_DEVICES=0 OPTUNA_WORKER_ID=0 python \$SWEEP &
CUDA_VISIBLE_DEVICES=1 OPTUNA_WORKER_ID=1 python \$SWEEP &
CUDA_VISIBLE_DEVICES=2 OPTUNA_WORKER_ID=2 python \$SWEEP &
CUDA_VISIBLE_DEVICES=3 OPTUNA_WORKER_ID=3 python \$SWEEP &
wait
"
