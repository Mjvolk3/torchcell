#!/bin/bash
# experiments/021-ozaydin-beta-carotene/scripts/gh_optuna_metabolism_004.slurm
# [[experiments.021-ozaydin-beta-carotene]]
#
# ONE arm, ONE GPU, round _004 -- the metabolism port of expression round _007. Three of
# these run concurrently on GilaHyper: betaxanthin here, beta-carotene in 021, the 19-AA
# metabolome in 022. Each arm keeps its own GPU because each has its own objective metric
# (beta-carotene ranks on Spearman, the other two on Pearson) and its own downstream use --
# betaxanthin feeds the Merzbacher comparison, beta-carotene feeds the in-house CIT2
# double-mutant panel.
#
#   cd <worktree-root>
#   ARM=betaxanthin sbatch --job-name=020-bx-003 experiments/021-ozaydin-beta-carotene/scripts/gh_optuna_metabolism_004.slurm
#
# TWO STAGES, RESOLVED AT JOB START, NOT BAKED IN. The driver reports which stage to run from
# the number of completed screen trials (`--resolve-stage`), so a requeue after a walltime
# kill continues rather than restarting:
#
#   screen   ~SCREEN_TARGET QMC trials, one seed each -> which axes matter
#   confirm  the top CONFIRM_TOP_K screen configs re-run across 5 seeds, ranked on the MEAN
#            -> which config is actually best
#
# The confirm stage is not optional bookkeeping on this arm. `_002` measured a replicate sigma
# of 0.030 at IDENTICAL config and seed, against a top-five spread of 0.021 -- so a
# single-seed sweep here cannot separate its best config from its fifth. See the driver's
# module docstring for the full derivation.
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
#SBATCH --signal=B:USR1@180
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/021-ozaydin-beta-carotene/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/021-ozaydin-beta-carotene/slurm/output/%x_%j.out

set -euo pipefail

# Resolve the worktree root from the submit dir so the BRANCH's code is used, not the primary
# checkout's -- the editable install otherwise wins (memory: worktree-build-needs-pythonpath).
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell}"
EXP_DIR=experiments/021-ozaydin-beta-carotene
SCRATCH_EXP=/scratch/projects/torchcell-scratch/$EXP_DIR
OPTUNA_DIR="$SCRATCH_EXP/optuna"
# Pre-create BOTH: slurm will not create the --output path itself, and a missing optuna dir
# surfaces as an opaque SQLite "unable to open database file".
mkdir -p "$OPTUNA_DIR" "$SCRATCH_EXP/slurm/output"

# A FIXED date, never a relative one. `date -d '+3 days'` would be recomputed by every
# requeue and the chain would extend forever. Overridable at submit time (sbatch propagates
# the environment, and so does the requeue's own sbatch).
export DEADLINE="${DEADLINE:-2026-07-30T22:00:00}"
source "$PROJECT_ROOT/experiments/019-simb-multimodal/scripts/requeue_until.sh"
requeue_arm_trap "$PROJECT_ROOT/$EXP_DIR/scripts/gh_optuna_metabolism_004.slurm"

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
export METAB_BASE_CONFIG="${METAB_BASE_CONFIG:-${ARM}_004}"
export OPTUNA_N_TRIALS="${OPTUNA_N_TRIALS:-20}"
export OPTUNA_WORKER_ID="${OPTUNA_WORKER_ID:-0}"
export SCREEN_TARGET="${SCREEN_TARGET:-60}"
export CONFIRM_TOP_K="${CONFIRM_TOP_K:-8}"
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 2))

SWEEP="$PROJECT_ROOT/$EXP_DIR/scripts/optuna_metabolism_sweep.py"

# `tail -n1`: the driver imports the trainer module, so stdout may carry other lines; only the
# final one is the stage word.
STAGE_RESOLVED=$(python "$SWEEP" --resolve-stage | tail -n1)
export STAGE="$STAGE_RESOLVED"

echo "=========================================================="
echo "host          : $(hostname)"
echo "job           : ${SLURM_JOB_ID:-none}  gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "project root  : $PROJECT_ROOT"
echo "torchcell from: $(python -c 'import torchcell; print(torchcell.__file__)')"
echo "arm           : $ARM   stage=$STAGE   base=$METAB_BASE_CONFIG"
echo "storage       : $OPTUNA_STORAGE"
echo "screen target : $SCREEN_TARGET   confirm top-k: $CONFIRM_TOP_K"
echo "deadline      : $DEADLINE"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo "=========================================================="

# Serialized pre-create so a fresh SQLite file's DDL is never raced by a concurrent worker --
# cheap insurance even at one worker, because a requeue re-enters this same path.
python "$SWEEP" --create-only
# Backgrounded + `wait` so the batch shell stays responsive to the SIGUSR1 requeue trap.
python "$SWEEP" &
wait

echo "=========================================================="
echo "DONE $ARM stage=$STAGE"
echo "=========================================================="

# Also resubmit on a CLEAN exit before the deadline: `optimize(n_trials=N)` returns after N
# trials, which is well short of the window. This is what carries the arm from screen into
# confirm without anyone watching it.
requeue_if_before_deadline "$PROJECT_ROOT/$EXP_DIR/scripts/gh_optuna_metabolism_004.slurm"
