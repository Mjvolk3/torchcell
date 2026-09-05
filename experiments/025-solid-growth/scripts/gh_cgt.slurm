#!/bin/bash
# experiments/025-solid-growth/scripts/gh_cgt.slurm
# GilaHyper 4-GPU DDP launcher for the 025 CellGraphTransformer arms.
# Parameterized over the Hydra config name, so ONE script launches every arm:
#
#   sbatch -J 025-s0-r-kl   gh_cgt.slurm cgt_s0_r_kl_000
#   sbatch -J 025-s0-r-mask gh_cgt.slurm cgt_s0_r_mask_001
#   sbatch -J 025-s0-q-mask gh_cgt.slurm cgt_s0_q_mask_002
#
# TIME is 12 h, not the 7 d the 010 launchers used. These arms are read at a checkpoint
# rather than at convergence, and the val-Pearson checkpoint callback keeps the best
# epoch, so a truncated run still answers the question it was launched to answer.
#
# MEMORY is 250 g. Each rank opens the 13.5M-record build (5.5 GB RSS measured, mostly
# the three split-index JSONs) before any dataloader worker forks, so four ranks start at
# ~22 GB.
#SBATCH -p main
#SBATCH --mem=250g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=025-CGT
#SBATCH --time=12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/025-solid-growth/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/025-solid-growth/slurm/output/%x_%j.out

CONFIG_NAME="${1:-cgt_s0_r_kl_000}"
echo "job=$SLURM_JOB_ID name=$SLURM_JOB_NAME config=$CONFIG_NAME host=$(hostname)"

cd /home/michaelvolk/Documents/projects/torchcell/experiments || exit 1

# DDP rendezvous -- pin to loopback. GilaHyper's bare hostname does not route (errno
# 113), and torchrun's --local-addr defaults to that hostname, so set it explicitly along
# with an explicit c10d endpoint. MASTER_ADDR is kept for Lightning's benefit.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Source conda.sh, NOT bin/activate, which consumes the script's $1 = config name and
# errors "env not found: <config>".
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# CUDA optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"  # Suppress SWIG DeprecationWarnings
export WANDB__SERVICE_WAIT=600

# No fd-sharing setting is needed here: CellDataModule sets file_system sharing in both
# the parent and a worker_init_fn (f1bfa95b), which is where it has to happen since the
# strategy is read in the process doing the pickling.

# num_workers from the SLURM CPU allocation (reserve 2 for overhead)
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 2))

torchrun --nnodes=1 --nproc_per_node=4 \
  --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:$MASTER_PORT \
  --rdzv-id=$SLURM_JOB_ID --local-addr=127.0.0.1 \
  /home/michaelvolk/Documents/projects/torchcell/experiments/025-solid-growth/scripts/equivariant_cell_graph_transformer.py \
  --config-name "$CONFIG_NAME" \
  data_module.num_workers=$NUM_WORKERS
