#!/bin/bash
#SBATCH -p main
#SBATCH --mem=250g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=010-metrics-1gpu-005
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# Change to the experiments directory
cd /home/michaelvolk/Documents/projects/torchcell/experiments || exit 1

# Activate conda environment directly
source ~/miniconda3/bin/activate
conda activate torchcell

# Add project to PYTHONPATH
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# CUDA optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"  # Suppress all warnings including SWIG DeprecationWarnings
# Calculate num_workers from SLURM CPU allocation (reserve 2 for overhead)
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 2))

# Run 1-GPU training for metrics consistency test (no torchrun needed)
~/miniconda3/envs/torchcell/bin/python \
  /home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/scripts/equivariant_cell_graph_transformer.py \
  --config-name equivariant_cell_graph_transformer_cabbi_metrics_1gpu_test_gh_005 \
  data_module.num_workers=$NUM_WORKERS
