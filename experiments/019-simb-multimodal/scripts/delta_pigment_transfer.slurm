#!/bin/bash
# experiments/019-simb-multimodal/scripts/delta_pigment_transfer.slurm
# [[plan.cgt-metabolism.2026.07.25]]
# Delta (NCSA) single-GPU run of the pigment metabolome-transfer contrast (Track A).
#
# Companion to `gh_pigment_transfer.slurm`; same four conditions, same driver, same shared
# base config via `delta_pigment_base` (which inherits `gh_pigment_base` and overrides only
# infrastructure). Cross-machine numbers stay comparable because the science block has ONE
# definition.
#
# PREREQUISITES -- all four, in order. Delta charges Service Units, so a job that dies on a
# missing prerequisite costs real allocation.
#
#   1. CODE on Delta. This work lives on branch `plan/cgt-metabolism`, which Delta does not
#      have. From GilaHyper:      git push -u origin plan/cgt-metabolism
#      then on Delta:             cd /projects/bbub/mjvolk3/torchcell
#                                 git fetch origin && git checkout plan/cgt-metabolism
#   2. DATA on Delta (291 MB, ONE Duo prompt). From GilaHyper:
#      bash experiments/019-simb-multimodal/scripts/sync_delta_dataset.sh fig6_pigment_transfer
#   3. DRIVER must honour PIGMENT_BASE_CONFIG. `run_pigment_conditions.py` hardcodes
#      BASE_CONFIG = "gh_pigment_base"; until that reads the env var, this job silently runs
#      the GilaHyper config (same science, wrong num_workers) -- harmless but not what the
#      header claims. Verify before the first Delta submit.
#   4. ENV check on a login node (cheap, allowed):
#      /work/hdd/bbub/miniconda3/envs/torchcell/bin/python -c "import torchcell, torch; print(torch.__version__)"
#
# CONVENTIONS taken from the WORKING interactive run (verified 2026-07-25) and from
# `delta_joint_expr_morph.slurm`. Three of them are the opposite of what the older
# `delta_train_cgt_multitask_000.slurm` does, which is stale:
#   1. account is `bbtp-delta-gpu` (NOT bbub-delta-gpu),
#   2. NO apptainer/singularity and NO `conda activate` -- call the env python DIRECTLY,
#   3. repo lives on /projects/bbub, DATA_ROOT on /work/hdd/bbub (quota).
#
#   ssh dt-login02.delta.ncsa.illinois.edu
#   cd /projects/bbub/mjvolk3/torchcell
#   sbatch experiments/019-simb-multimodal/scripts/delta_pigment_transfer.slurm
#
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbtp-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=019-pigment
#SBATCH --time=08:00:00
#SBATCH --constraint="scratch"
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/projects/bbub/mjvolk3/torchcell}"
mkdir -p "$PROJECT_ROOT/experiments/019-simb-multimodal/slurm/output"

# Call the env's python DIRECTLY -- NO conda activate. /work/hdd/bbub/miniconda3 is a PREFIX
# location, not a conda install (no bin/activate); `source .../bin/activate` there fails
# silently and the job then runs system python3.9. The env python resolves its own libs via
# rpath, so bare invocation is the zero-ambiguity path.
ENV_PY="${DELTA_ENV_PY:-/work/hdd/bbub/miniconda3/envs/torchcell/bin/python}"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Delta compute nodes have no outbound network; wandb MUST be offline or the run stalls.
export WANDB_MODE=offline

export PIGMENT_BASE_CONFIG="${PIGMENT_BASE_CONFIG:-delta_pigment_base}"
export PIGMENT_CONDITIONS="${PIGMENT_CONDITIONS:-A1,A2,A3,A4}"
export PIGMENT_MAX_EPOCHS="${PIGMENT_MAX_EPOCHS:-150}"

echo "=========================================================="
echo "host          : $(hostname)"
echo "job           : ${SLURM_JOB_ID:-none}  gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "project root  : $PROJECT_ROOT"
echo "env python    : $ENV_PY"
echo "torchcell from: $($ENV_PY -c 'import torchcell; print(torchcell.__file__)')"
echo "base config   : $PIGMENT_BASE_CONFIG"
echo "conditions    : $PIGMENT_CONDITIONS   max_epochs=$PIGMENT_MAX_EPOCHS"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

$ENV_PY -u experiments/019-simb-multimodal/scripts/run_pigment_conditions.py

RESULTS=experiments/019-simb-multimodal/results
cp "$RESULTS/pigment_transfer_runs.json" \
   "$RESULTS/pigment_transfer_runs_delta_${SLURM_JOB_ID:-manual}.json"

echo "=========================================================="
echo "DONE. Snapshot: $RESULTS/pigment_transfer_runs_delta_${SLURM_JOB_ID:-manual}.json"
echo "Sync wandb from a login node afterwards (compute nodes are offline)."
echo "=========================================================="
