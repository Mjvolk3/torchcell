#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_cgt_multitask.slurm
# GilaHyper 4-GPU DDP launcher for the Fig-3 multitask CGT (SIMB 2026 WS-RUN).
# Parameterized over the Hydra config name: pass it as the first argument so ONE script
# launches every main config AND every edge-of-config-space sweep config.
#
#   sbatch -J <config-name> gh_cgt_multitask.slurm <config-name>
#
# e.g.  sbatch -J gh_cgt_multitask_expr_000  gh_cgt_multitask.slurm gh_cgt_multitask_expr_000
# Defaults to the full triple-head joint config when no argument is given.
#SBATCH -p main
#SBATCH --mem=250g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=019-MTCGT
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out

CONFIG_NAME="${1:-gh_cgt_multitask_joint_000}"
echo "job=$SLURM_JOB_ID name=$SLURM_JOB_NAME config=$CONFIG_NAME host=$(hostname)"

# Change to the experiments directory (matches 010 GilaHyper convention)
cd /home/michaelvolk/Documents/projects/torchcell/experiments || exit 1

# DDP rendezvous — pin to loopback. GilaHyper's bare hostname does not route
# (errno 113), and torchrun's --local-addr defaults to that hostname, so we set it
# explicitly to 127.0.0.1 along with an explicit c10d rendezvous endpoint. MASTER_ADDR
# is kept for Lightning's benefit.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Activate conda (no container on GilaHyper). Source conda.sh (NOT bin/activate, which
# consumes the script's $1 = config name and errors "env not found: <config>").
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

# Add project to PYTHONPATH
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# CUDA optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"  # Suppress SWIG DeprecationWarnings
export WANDB__SERVICE_WAIT=600

# Calculate num_workers from SLURM CPU allocation (reserve 2 for overhead)
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 2))

torchrun --nnodes=1 --nproc_per_node=4 \
  --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:$MASTER_PORT \
  --rdzv-id=$SLURM_JOB_ID --local-addr=127.0.0.1 \
  /home/michaelvolk/Documents/projects/torchcell/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py \
  --config-name "$CONFIG_NAME" \
  data_module.num_workers=$NUM_WORKERS
