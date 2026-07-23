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
# CHM230022 = bbtp-delta-gpu (`accounts`: 10,418 hrs, "generative ml for ODE param est").
# NOT bbhh (that's overdrawn -6447, a different project). bbub-delta-gpu (3851 hrs, torchcell
# multimodal) also valid. Override at submit: sbatch --account=bbub-delta-gpu ...
#SBATCH --account=bbtp-delta-gpu
#SBATCH --gpus-per-node=4
#SBATCH --job-name=019-joint-optuna
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch"
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
# torchcell IS provisioned on Delta at /projects/bbub/mjvolk3/torchcell (repo + rockylinux_9.sif
# + conda). STORAGE stays on bbub; only the CHARGE account points at CHM230022 (bbhh/bbtp) above.
# Pre-create the --output dir once (SLURM opens it before the script's mkdir), and confirm the
# `torchcell` conda env has `optuna` (pip install if not — same stale-env lesson as IGB).
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out

module reset
source ~/.bashrc

# Storage root = the existing bbub clone (override with DELTA_TC_ROOT if you relocate).
PROJECT_ROOT="${DELTA_TC_ROOT:-/projects/bbub/mjvolk3/torchcell}"
CONDITION="${CONDITION:-joint}"
OPTUNA_DIR=$PROJECT_ROOT/experiments/019-simb-multimodal/optuna
mkdir -p "$OPTUNA_DIR"
mkdir -p "$PROJECT_ROOT/experiments/019-simb-multimodal/slurm/output"

cd "$PROJECT_ROOT" || exit 1
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 4))

# Call the env's python DIRECTLY — NO conda activate. The py3.13 env (rebuilt 2026-07-23,
# torch 2.5.1+cu124/PyG 2.4.0/optuna) is a PREFIX env at /work/hdd/bbub/miniconda3/envs/torchcell,
# but /work/hdd/bbub/miniconda3 is NOT a conda install (no bin/activate — base conda is at
# /projects/bbub). `source .../bin/activate` there fails silently -> job runs system python3.9.
# The env python resolves its own libs via rpath, so bare invocation is the zero-ambiguity path.
ENV_PY="${DELTA_ENV_PY:-/work/hdd/bbub/miniconda3/envs/torchcell/bin/python}"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

# W&B: Delta compute has internet -> ONLINE (live dashboards). If a compute node ever blocks
# egress, flip to the IGB-proven offline path — uncomment these two, then 'wandb sync' from a
# Delta LOGIN node afterward:
#   export WANDB_MODE=offline
#   export WANDB_DIR=$PROJECT_ROOT/experiments/019-simb-multimodal
export WANDB__SERVICE_WAIT=600

export OPTUNA_STORAGE="sqlite:///$OPTUNA_DIR/optuna_019_joint_${CONDITION}.db"
export OPTUNA_STUDY_NAME="joint_${CONDITION}_000"
export OPTUNA_N_TRIALS=20
export JOINT_BASE_CONFIG="delta_joint_expr_morph_000"

SWEEP=$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py

# Pre-create the study in ONE process so the 4 workers only load_if_exists — avoids the fresh-DB
# DDL race (sqlite 'table studies already exists'). --create-only makes the driver create the
# study with the RIGHT directions (joint = multi-objective, expr/morph = single) from one place.
$ENV_PY $SWEEP --create-only

CUDA_VISIBLE_DEVICES=0 OPTUNA_WORKER_ID=0 $ENV_PY $SWEEP &
CUDA_VISIBLE_DEVICES=1 OPTUNA_WORKER_ID=1 $ENV_PY $SWEEP &
CUDA_VISIBLE_DEVICES=2 OPTUNA_WORKER_ID=2 $ENV_PY $SWEEP &
CUDA_VISIBLE_DEVICES=3 OPTUNA_WORKER_ID=3 $ENV_PY $SWEEP &
wait
