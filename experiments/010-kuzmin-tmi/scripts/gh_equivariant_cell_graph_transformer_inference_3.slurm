#!/bin/bash
#SBATCH -p main
#SBATCH --mem=250g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=010-inference-3
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# INFERENCE_3: Multi-GPU model inference on triples with relaxed thresholds
# Thresholds: max(SMF) > 1.04, max(DMF) > 1.08
# Baselines: all(SMF) > 0.90, all(DMF) > 0.90
# Uses best model checkpoint (Pearson=0.4619)
# 4 GPUs via torchrun: each GPU processes 1/4 of dataset

cd /home/michaelvolk/Documents/projects/torchcell || exit 1

source ~/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# CUDA optimizations for inference
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

# NCCL settings for multi-GPU
export NCCL_DEBUG=WARN

# Launch 4-GPU inference with torchrun
~/miniconda3/envs/torchcell/bin/torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  experiments/010-kuzmin-tmi/scripts/equivariant_cell_graph_transformer_inference_3.py \
  --config-name equivariant_cell_graph_transformer_cabbi_002_inference_m02 \
  data_module.batch_size=4096
