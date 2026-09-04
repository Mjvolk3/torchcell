#!/bin/bash
#SBATCH -p main
#SBATCH --mem=110g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=010-inf4-m00-s0
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi/slurm/output/%x_%j.out

# One contiguous quarter of the inference_4 space on one GPU. Twelve of these cover
# three checkpoints; SLURM runs four at a time on the four GilaHyper GPUs.

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
  experiments/010-kuzmin-tmi/scripts/equivariant_cell_graph_transformer_inference_4.py \
  --config-name equivariant_cell_graph_transformer_inference_4_m00 \
  data_module.batch_size=2048 \
  inference.n_shards=4 \
  inference.shard_index=0
