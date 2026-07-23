#!/bin/bash
# experiments/019-simb-multimodal/scripts/igb_mmli_optuna_morph.slurm
# IGB BioCluster `mmli` MORPHOLOGY Optuna sniff sweep launcher (SIMB 2026, Fig-6 strand).
#
# ONE mmli node (compute-5-7, 4 GPUs). Instead of Hydra submitit (which submits many child
# SLURM jobs — wasteful on a single-node partition), this runs Optuna's canonical multi-worker
# pattern: 4 worker processes, each PINNED to one GPU via CUDA_VISIBLE_DEVICES, all pulling
# trials from ONE shared SQLite study on scratch. IGB compute nodes are OFFLINE, so W&B runs
# in `offline` mode and is `wandb sync`'d later from an internet machine.
#
#   sbatch experiments/019-simb-multimodal/scripts/igb_mmli_optuna_morph.slurm
#
#SBATCH -p mmli
#SBATCH --mem=488g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=019-morph-optuna
#SBATCH --time=3-00:00:00
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

cd "$PROJECT_ROOT/experiments" || exit 1

# 16 cpus / 4 workers = 4 dataloader workers per trial.
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 4))

singularity exec --nv "$PROJECT_ROOT/rockylinux_9.sif" bash -c "
source \$HOME/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

# --- OFFLINE: no internet on IGB compute nodes ---
export WANDB_MODE=offline
export WANDB__SERVICE_WAIT=600

# --- Optuna shared study (SQLite on scratch) ---
export OPTUNA_STORAGE='sqlite:////home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/optuna/optuna_019_morph.db'
export OPTUNA_STUDY_NAME='morph_000'
export OPTUNA_N_TRIALS=20            # per worker -> 4*20 = 80 trials over 72 combos
export MORPH_BASE_CONFIG='igb_mmli_optuna_morph_000'
export NUM_WORKERS=$NUM_WORKERS

SWEEP=$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/optuna_morph_sweep.py

# Worker 0 first — it creates the study + tables; the others load_if_exists after a short
# stagger to avoid a fresh-DB table-creation race.
CUDA_VISIBLE_DEVICES=0 OPTUNA_WORKER_ID=0 python \$SWEEP &
sleep 15
CUDA_VISIBLE_DEVICES=1 OPTUNA_WORKER_ID=1 python \$SWEEP &
CUDA_VISIBLE_DEVICES=2 OPTUNA_WORKER_ID=2 python \$SWEEP &
CUDA_VISIBLE_DEVICES=3 OPTUNA_WORKER_ID=3 python \$SWEEP &
wait
"
