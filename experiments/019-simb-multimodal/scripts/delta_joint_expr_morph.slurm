#!/bin/bash
# experiments/019-simb-multimodal/scripts/delta_joint_expr_morph.slurm
# Delta (NCSA) CONTROLLED expr<->morph auxiliary-task Optuna sweep launcher.
# One 4-GPU A40 node, 4 single-GPU Optuna workers pinned per GPU -> one shared SQLite study
# for a single CONDITION. Delta compute nodes have internet -> W&B ONLINE. 48h walltime cap.
#
# Launch all THREE conditions (each its own study) to get the controlled comparison:
#   sbatch --export=ALL,CONDITION=morph experiments/019-simb-multimodal/scripts/delta_joint_expr_morph.slurm
#   sbatch --export=ALL,CONDITION=expr  experiments/019-simb-multimodal/scripts/delta_joint_expr_morph.slurm
#   sbatch --export=ALL,CONDITION=joint experiments/019-simb-multimodal/scripts/delta_joint_expr_morph.slurm
#
#SBATCH --mem=200g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbub-delta-gpu
#SBATCH --gpus-per-node=4
#SBATCH --job-name=019-joint-optuna
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch"
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out

module reset
source ~/.bashrc

PROJECT_ROOT=/scratch/bbub/mjvolk3/torchcell
CONDITION="${CONDITION:-joint}"
OPTUNA_DIR=$PROJECT_ROOT/experiments/019-simb-multimodal/optuna
mkdir -p "$OPTUNA_DIR"
mkdir -p "$PROJECT_ROOT/experiments/019-simb-multimodal/slurm/output"

cd "$PROJECT_ROOT" || exit 1
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 4))

srun apptainer exec --nv \
    --bind /projects/bbub:/projects/bbub \
    --bind /scratch/bbub/mjvolk3:/scratch/bbub/mjvolk3 \
    --bind /work:/work \
    "$PROJECT_ROOT/rockylinux_9.sif" bash -c "
source /projects/bbub/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

# Delta compute has internet -> W&B ONLINE (no offline dance).
export WANDB__SERVICE_WAIT=600

export CONDITION='$CONDITION'
export OPTUNA_STORAGE=\"sqlite:////scratch/bbub/mjvolk3/torchcell/experiments/019-simb-multimodal/optuna/optuna_019_joint_${CONDITION}.db\"
export OPTUNA_STUDY_NAME='joint_${CONDITION}_000'
export OPTUNA_N_TRIALS=20
export JOINT_BASE_CONFIG='delta_joint_expr_morph_000'
export NUM_WORKERS=$NUM_WORKERS

SWEEP=$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py

# Worker 0 creates the study + tables; stagger the rest to dodge a fresh-DB race.
CUDA_VISIBLE_DEVICES=0 OPTUNA_WORKER_ID=0 python \$SWEEP &
sleep 15
CUDA_VISIBLE_DEVICES=1 OPTUNA_WORKER_ID=1 python \$SWEEP &
CUDA_VISIBLE_DEVICES=2 OPTUNA_WORKER_ID=2 python \$SWEEP &
CUDA_VISIBLE_DEVICES=3 OPTUNA_WORKER_ID=3 python \$SWEEP &
wait
"
