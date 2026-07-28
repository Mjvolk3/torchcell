#!/bin/bash
# experiments/020-cachera-betaxanthin/scripts/delta_metabolism_000.slurm
# [[experiments.020-cachera-betaxanthin]]
#
# DELTA (NCSA) metabolism sweep -- ONE arm per job, 4 GPUs, 4 Optuna workers on ONE study.
#
# Four jobs are launched from this single script, selected by $ARM:
#
#   ARM=betaxanthin    base=delta_betaxanthin_000     mirrors the GilaHyper arm
#   ARM=beta_carotene  base=delta_beta_carotene_000   mirrors the GilaHyper arm
#   ARM=mulleder19     base=delta_mulleder19_000      mirrors the GilaHyper arm
#   ARM=bx_ctrl        base=delta_bx_m19_000          CONTROL  -- betaxanthin head only
#   ARM=bx_m19         base=delta_bx_m19_000          JOINT    -- + the metabolome head
#
# The last two are the CONTROLLED PAIR and answer "does the rest of the metabolism signal
# improve betaxanthin prediction". They share a base config, run on the same 4,432
# both-target genotypes (`require_head_targets`), and differ ONLY in `active_heads`. Run them
# as ONE job with 2 workers each so they cannot drift apart in queue position, node, or
# software state -- a paired comparison split across two jobs days apart is a worse
# comparison for no benefit. Set ARM=bx_pair for that mode.
#
# WHY 4 WORKERS ON 4 GPUs SHARING ONE STUDY, unlike the 1-GPU GilaHyper launcher: the
# workers here sit on ONE node and one filesystem, so concurrent SQLite is safe (the same
# reason the IGB pooled study works across mmli/cabbi on shared GPFS, and the reason the
# GilaHyper and Delta studies are NOT pooled -- those are different filesystems).
#
# Delta compute nodes HAVE internet, so W&B is ONLINE -- no offline/sync dance.
#
#   cd /projects/bbub/mjvolk3/torchcell
#   ARM=betaxanthin sbatch --account=bbtp-delta-gpu experiments/020-cachera-betaxanthin/scripts/delta_metabolism_000.slurm
#
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=180g
#SBATCH --time=2-00:00:00
#SBATCH --job-name=020-metab-delta
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/020-cachera-betaxanthin/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/020-cachera-betaxanthin/slurm/output/%x_%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/projects/bbub/mjvolk3/torchcell}"
# DATA_ROOT is the LARGE /work/hdd space, not /projects (tight quota). This must agree with
# the Delta .env, because run_training resolves the dataset from $DATA_ROOT.
DELTA_DATA_ROOT="${DELTA_DATA_ROOT:-/work/hdd/bbub/mjvolk3/torchcell}"
# THE WORKING ENV, as validated interactively on Delta 2026-07-28: `torchcell` on /work/hdd
# (NOT `torchcell313`, an earlier name from the July notes, and NOT Delta's stock envs, which
# are Python 3.11 and hit a SyntaxError on the repo's PEP 695 generics).
#   torch 2.8.0+cu128 / PyG 2.7.0 / torch_scatter 2.1.2+pt28cu128 / torchcell 1.2.0
#
# GOTCHA WORTH KEEPING: install into it with the env's EXPLICIT pip
# (`$DELTA_CONDA_BASE/envs/$DELTA_CONDA_ENV/bin/pip`). A bare `pip` on the Delta login node
# is system python3.9 and silently drops packages into ~/.local, where this env never sees
# them -- which is how a "successful" install can still leave a ModuleNotFoundError at run
# time.
DELTA_CONDA_BASE="${DELTA_CONDA_BASE:-/work/hdd/bbub/miniconda3}"
DELTA_CONDA_ENV="${DELTA_CONDA_ENV:-torchcell}"
DELTA_PY="$DELTA_CONDA_BASE/envs/$DELTA_CONDA_ENV/bin/python"

export ARM="${ARM:?set ARM to betaxanthin|beta_carotene|mulleder19|bx_ctrl|bx_m19|bx_pair}"

OPTUNA_DIR="$PROJECT_ROOT/experiments/020-cachera-betaxanthin/optuna"
# slurm does NOT create the --output directory; a missing one makes the job die in 0s with
# no log at all, which is the single most common way a Delta submission fails silently.
mkdir -p "$OPTUNA_DIR" "$PROJECT_ROOT/experiments/020-cachera-betaxanthin/slurm/output"

# Base config per arm; the controlled pair shares one.
case "$ARM" in
  bx_ctrl|bx_m19|bx_pair) DEFAULT_BASE=delta_bx_m19_000 ;;
  *)                      DEFAULT_BASE="delta_${ARM}_000" ;;
esac
export METAB_BASE_CONFIG="${METAB_BASE_CONFIG:-$DEFAULT_BASE}"

export DATA_ROOT="$DELTA_DATA_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB__SERVICE_WAIT=600
export SCREEN_TARGET="${SCREEN_TARGET:-80}"
export CONFIRM_TOP_K="${CONFIRM_TOP_K:-8}"
export OPTUNA_N_TRIALS="${OPTUNA_N_TRIALS:-30}"
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK / 8))

SWEEP="$PROJECT_ROOT/experiments/020-cachera-betaxanthin/scripts/optuna_metabolism_sweep.py"

# NO SINGULARITY. An earlier draft wrapped every call in
# `singularity exec --nv rockylinux_9.sif`, inherited from the 019 launchers. The env that
# was actually validated on Delta runs NATIVELY on the host -- its torch is built against
# the host CUDA (2.8.0+cu128) and the interactive smoke test that got through the whole
# import chain called `$DELTA_PY` directly. Running a host-built CUDA wheel inside a
# container that carries its own driver stack is an untested second variable, and there is
# nothing to gain from it here. Kept out rather than "probably fine".

# One worker: $1 = ARM, $2 = CUDA device, $3 = optuna worker id.
launch_worker() {
  local arm="$1" dev="$2" wid="$3"
  bash -c "
    export ARM=$arm
    export OPTUNA_STORAGE='sqlite:///$OPTUNA_DIR/optuna_020_${arm}_delta.db'
    export STAGE=\$('$DELTA_PY' '$SWEEP' --resolve-stage | tail -n1)
    [ \"\$STAGE\" = done ] && { echo '[$arm] screen+confirm complete'; exit 0; }
    CUDA_VISIBLE_DEVICES=$dev OPTUNA_WORKER_ID=$wid '$DELTA_PY' '$SWEEP'
  " &
}

echo "=========================================================="
echo "host        : $(hostname)"
echo "job         : ${SLURM_JOB_ID:-none}   arm=$ARM   base=$METAB_BASE_CONFIG"
echo "project     : $PROJECT_ROOT"
echo "DATA_ROOT   : $DATA_ROOT"
echo "python      : $DELTA_PY"
echo "screen tgt  : $SCREEN_TARGET   confirm top-k: $CONFIRM_TOP_K"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=========================================================="

if [ "$ARM" = bx_pair ]; then
  # The controlled pair, 2 GPUs each, in ONE job -- same node, same software, same queue
  # slot, so the only difference between the arms is the one under test.
  for a in bx_ctrl bx_m19; do
    ARM=$a OPTUNA_STORAGE="sqlite:///$OPTUNA_DIR/optuna_020_${a}_delta.db" \
      "$DELTA_PY" "$SWEEP" --create-only
  done
  launch_worker bx_ctrl 0 0
  launch_worker bx_ctrl 1 1
  launch_worker bx_m19  2 0
  launch_worker bx_m19  3 1
else
  # Serialized pre-create so the four workers never race a fresh SQLite file's DDL -- each
  # spends >15s importing torch before it would reach create_study.
  ARM=$ARM OPTUNA_STORAGE="sqlite:///$OPTUNA_DIR/optuna_020_${ARM}_delta.db" \
    "$DELTA_PY" "$SWEEP" --create-only
  for i in 0 1 2 3; do launch_worker "$ARM" "$i" "$i"; done
fi
wait

echo "=========================================================="
echo "DONE $ARM"
echo "=========================================================="
