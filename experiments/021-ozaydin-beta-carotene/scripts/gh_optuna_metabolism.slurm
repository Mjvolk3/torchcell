#!/bin/bash
# experiments/021-ozaydin-beta-carotene/scripts/gh_optuna_metabolism.slurm
# [[experiments.021-ozaydin-beta-carotene]]
#
# ONE Optuna study on ONE GPU. Three of these run concurrently for ~2 days -- betaxanthin,
# beta-carotene, and the 19-AA metabolome.
#
#   ARM=betaxanthin   sbatch --job-name=020-sweep-bx experiments/.../gh_optuna_metabolism.slurm
#   ARM=beta_carotene sbatch --job-name=020-sweep-bc ...
#   ARM=mulleder19    sbatch --job-name=020-sweep-m19 ...
#
# WHY ONE WORKER PER JOB, unlike the 019 expression launcher that forks four workers across
# four GPUs inside a single job: there the four workers share ONE study, so they are four
# samplers cooperating on one search. Here the three jobs are three DIFFERENT studies with
# THREE DIFFERENT objective metrics (beta-carotene ranks on Spearman, the others on Pearson),
# so they must not share a storage file or a study name. Separate jobs also mean one arm
# crashing does not take the other two down.
#
# `--create-only` still runs first: on a fresh SQLite file, concurrent DDL races. It is cheap
# insurance even with a single worker, because a requeue re-enters the same path.
#
# W&B is ONLINE. These are keeper runs and the sweep view is the artifact.
#
#SBATCH -p main
#SBATCH --mem=96g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/021-ozaydin-beta-carotene/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/021-ozaydin-beta-carotene/slurm/output/%x_%j.out

set -euo pipefail

# Resolve the worktree root from the submit dir so the BRANCH's code is used, not the primary
# checkout's -- the editable install otherwise wins (memory: worktree-build-needs-pythonpath).
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell}"
SCRATCH_EXP=/scratch/projects/torchcell-scratch/experiments/021-ozaydin-beta-carotene
OPTUNA_DIR="$SCRATCH_EXP/optuna"
# Pre-create BOTH dirs: slurm will not create the --output path itself, and a missing optuna
# dir surfaces as an opaque SQLite "unable to open database file".
mkdir -p "$OPTUNA_DIR" "$SCRATCH_EXP/slurm/output"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB__SERVICE_WAIT=600

export ARM="${ARM:?set ARM to one of betaxanthin | beta_carotene | mulleder19}"
export OPTUNA_STORAGE="sqlite:///$OPTUNA_DIR/optuna_020_${ARM}.db"
export OPTUNA_STUDY_NAME="${OPTUNA_STUDY_NAME:-${ARM}_000}"
export OPTUNA_N_TRIALS="${OPTUNA_N_TRIALS:-40}"
export OPTUNA_WORKER_ID="${OPTUNA_WORKER_ID:-0}"
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 2))

SWEEP="$PROJECT_ROOT/experiments/021-ozaydin-beta-carotene/scripts/optuna_metabolism_sweep.py"

echo "=========================================================="
echo "host          : $(hostname)"
echo "job           : ${SLURM_JOB_ID:-none}  gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "project root  : $PROJECT_ROOT"
echo "torchcell from: $(python -c 'import torchcell; print(torchcell.__file__)')"
echo "arm           : $ARM"
echo "study         : $OPTUNA_STUDY_NAME"
echo "storage       : $OPTUNA_STORAGE"
echo "n_trials      : $OPTUNA_N_TRIALS"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo "=========================================================="

python "$SWEEP" --create-only
python "$SWEEP"

echo "=========================================================="
echo "DONE sweep: $ARM"
echo "=========================================================="
