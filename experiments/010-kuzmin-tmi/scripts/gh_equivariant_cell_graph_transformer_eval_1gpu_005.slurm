#!/bin/bash
#SBATCH -p main
#SBATCH --mem=250g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=010-eval-1gpu-trained
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# Eval of 1-GPU trained model (job 747 - pre-hook, no on_fit_start) on 1 GPU
# Part of metrics consistency test - WITHOUT the hook fix

cd /home/michaelvolk/Documents/projects/torchcell || exit 1

source ~/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

~/miniconda3/envs/torchcell/bin/python \
  experiments/010-kuzmin-tmi/scripts/equivariant_cell_graph_transformer_eval.py \
  --config-name equivariant_cell_graph_transformer_cabbi_metrics_1gpu_test_gh_005_eval \
  data_module.batch_size=512
