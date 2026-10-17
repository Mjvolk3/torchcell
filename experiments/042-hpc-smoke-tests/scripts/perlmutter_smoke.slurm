#!/bin/bash
#SBATCH -A m5429_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -t 00:20:00
#SBATCH -J 042-smoke-perlmutter
#SBATCH -o /global/cfs/cdirs/m5429/mjvolk3/torchcell/experiments/042-hpc-smoke-tests/slurm/output/%x_%j.out
# experiments/042-hpc-smoke-tests/scripts/perlmutter_smoke.slurm
# [[experiments.042-hpc-smoke-tests.scripts.perlmutter_smoke]]
# Submit from a Perlmutter login node:  sbatch <this file>   (paths match machines.sh).
set -euo pipefail

TC_PREFIX=/global/cfs/cdirs/m5429/mjvolk3
TC_TREE=$TC_PREFIX/torchcell
PY=$TC_PREFIX/miniforge3/envs/torchcell/bin/python

cd "$TC_TREE"
set -a; source .env; set +a
export PYTHONUNBUFFERED=1 PYTHONPATH="$TC_TREE"
# CFS refuses file locks (wandb.init died with errno 524 on the .wandb.syncstate lock,
# job 59896049), so W&B's run directory lives on scratch; results stay on CFS.
export WANDB_DIR=/pscratch/sd/m/mjvolk3/wandb
mkdir -p "$WANDB_DIR"
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

"$PY" experiments/042-hpc-smoke-tests/scripts/smoke_train.py \
  --machine perlmutter \
  --out "experiments/042-hpc-smoke-tests/results/smoke_perlmutter_${SLURM_JOB_ID}.json"
