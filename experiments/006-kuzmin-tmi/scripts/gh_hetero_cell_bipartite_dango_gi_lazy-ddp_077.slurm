#!/bin/bash
#SBATCH -p main
#SBATCH --mem=500g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=006-077-FULL-MASKS
#SBATCH --time=300-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

# Training with FULL MASKS preprocessed dataset - Maximum speed approach
# Prerequisites: Run preprocess_lazy_dataset_full_masks.py first (creates ~847.7GB LMDB)
#
# Expected Performance (verified):
#   - Loading: <0.1ms per sample (direct deserialization)
#   - Training: 0.38+ it/s (matching or exceeding on-the-fly)
#   - Storage: UINT8 format (optimal memory bandwidth)
#
# Usage (from project root /home/michaelvolk/Documents/projects/torchcell):
#   1. First run preprocessing (one-time, ~50 minutes):
#      python experiments/006-kuzmin-tmi/scripts/preprocess_lazy_dataset_full_masks.py
#   2. Then submit this job:
#      sbatch experiments/006-kuzmin-tmi/scripts/gh_hetero_cell_bipartite_dango_gi_lazy-ddp_074_full_masks.slurm

# Change to the experiments directory
cd /home/michaelvolk/Documents/projects/torchcell/experiments || exit 1

# Set environment variables needed for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Launch with torchrun for multi-GPU training with FULL MASKS dataset
apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
  --env PYTHONUNBUFFERED=1 \
  /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc '
# Add the project root to Python path
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# Enable expandable memory segments to reduce GPU memory fragmentation

# ONLY DIFFERENCE IN THIS EXPERIMENT in NO expandable_segments!
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CUDA debugging flags (set to 0 for maximum performance, 1 for debugging)
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

echo "=========================================="
echo "USING FULL MASKS PREPROCESSED DATASET"
echo "=========================================="
echo "Expected performance: 0.38+ it/s"
echo "Storage method: Full masks UINT8 (~847.7GB LMDB)"
echo "=========================================="

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_lazy_preprocessed.py \
  --config-name hetero_cell_bipartite_dango_gi_gh_077
'