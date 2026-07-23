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
# ONE-TIME SETUP before the first sbatch — SLURM opens --output at job start, BEFORE this
# script's own mkdir runs, so the log dir must already exist or the job FAILS instantly with
# no output:
#   mkdir -p /scratch/bbub/mjvolk3/torchcell/experiments/019-simb-multimodal/slurm/output
#
#SBATCH --mem=200g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
# CHM230022 (ODE param-estimation) alloc — code rotates: TRY BOTH. Parameter_Estimation notes
# use bbhh (live /projects/bbhh/mjvolk3); its workspace + PDE4 use bbtp. An expired account is
# rejected instantly ("Invalid account"). Override at submit: sbatch --account=bbtp-delta-gpu ...
#SBATCH --account=bbhh-delta-gpu
#SBATCH --gpus-per-node=4
#SBATCH --job-name=019-joint-optuna
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch"
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
# CONFIRM ON DELTA before first submit (Delta is NOT provisioned for torchcell yet):
#   * PROJECT_ROOT below = where you clone torchcell (default /projects/bbhh/mjvolk3, where
#     Parameter_Estimation lives). --output path MUST be a real dir (pre-create it) or the
#     job fails instantly with no log.
#   * conda env `torchcell` must exist on Delta + `optuna` installed (see runbook note).
#   * DATA_ROOT in the cloned repo's .env must point where sync_delta_fig3_core.sh landed the data.
#SBATCH --output=/projects/bbhh/mjvolk3/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbhh/mjvolk3/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out

module reset
source ~/.bashrc

# Single storage-root knob: change here (and the --output path above) to switch bbhh<->bbtp.
PROJECT_ROOT="${DELTA_TC_ROOT:-/projects/bbhh/mjvolk3/torchcell}"
CONDITION="${CONDITION:-joint}"
OPTUNA_DIR=$PROJECT_ROOT/experiments/019-simb-multimodal/optuna
mkdir -p "$OPTUNA_DIR"
mkdir -p "$PROJECT_ROOT/experiments/019-simb-multimodal/slurm/output"

cd "$PROJECT_ROOT" || exit 1
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 4))

# Provisioning knobs — set once for your Delta allocation (defaults = bbhh, where
# Parameter_Estimation lives). CONFIRM the conda base + whether the apptainer container is
# needed (Parameter_Estimation runs bare conda on Delta; torchcell's heavier env may too).
ACCT_PROJECTS="${DELTA_ACCT_PROJECTS:-/projects/bbhh}"
CONDA_BASE="${DELTA_CONDA_BASE:-$ACCT_PROJECTS/miniconda3}"

srun apptainer exec --nv \
    --bind "$ACCT_PROJECTS:$ACCT_PROJECTS" \
    --bind /work:/work \
    "$PROJECT_ROOT/rockylinux_9.sif" bash -c "
source $CONDA_BASE/bin/activate
conda activate torchcell

export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

# Delta compute has internet -> W&B ONLINE (no offline dance).
export WANDB__SERVICE_WAIT=600

export CONDITION='$CONDITION'
export OPTUNA_STORAGE=\"sqlite:///$OPTUNA_DIR/optuna_019_joint_${CONDITION}.db\"
export OPTUNA_STUDY_NAME='joint_${CONDITION}_000'
export OPTUNA_N_TRIALS=20
export JOINT_BASE_CONFIG='delta_joint_expr_morph_000'
export NUM_WORKERS=$NUM_WORKERS

SWEEP=$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py

# Pre-create the study in ONE process so the 4 workers only load_if_exists — a stagger is not
# enough (each worker imports torch for >15s before create_study, so all four race the fresh
# DB's DDL: sqlite 'table studies already exists'). Serializing creation kills the race.
python -c \"import os, optuna; optuna.create_study(study_name=os.environ['OPTUNA_STUDY_NAME'], storage=os.environ['OPTUNA_STORAGE'], direction='maximize', load_if_exists=True)\"

CUDA_VISIBLE_DEVICES=0 OPTUNA_WORKER_ID=0 python \$SWEEP &
CUDA_VISIBLE_DEVICES=1 OPTUNA_WORKER_ID=1 python \$SWEEP &
CUDA_VISIBLE_DEVICES=2 OPTUNA_WORKER_ID=2 python \$SWEEP &
CUDA_VISIBLE_DEVICES=3 OPTUNA_WORKER_ID=3 python \$SWEEP &
wait
"
