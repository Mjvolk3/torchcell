#!/bin/bash
#SBATCH -p main
#SBATCH --mem=250g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=010-eval-m01
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# Full evaluation of CellGraphTransformer on val and test sets
# Collects ALL predictions, computes metrics, saves to parquet
# Logs to same wandb group as training run

# Change to the project root directory (required for relative paths in Python script)
cd /home/michaelvolk/Documents/projects/torchcell || exit 1

# Activate conda environment directly
source ~/miniconda3/bin/activate
conda activate torchcell

# Add project to PYTHONPATH
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# CUDA optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

# Run evaluation
~/miniconda3/envs/torchcell/bin/python \
  experiments/010-kuzmin-tmi/scripts/equivariant_cell_graph_transformer_eval.py \
  --config-name equivariant_cell_graph_transformer_cabbi_000_eval_m01
