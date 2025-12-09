#!/bin/bash
#SBATCH -p main
#SBATCH --mem=500g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=006-069-HCPD-LAZY
#SBATCH --time=300-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

# Change to the experiments directory
cd /home/michaelvolk/Documents/projects/torchcell/experiments || exit 1

# Set environment variables needed for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Launch with torchrun for multi-GPU training with lazy model
apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
  --env PYTHONUNBUFFERED=1 \
  /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc '
# Add the project root to Python path
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# Enable expandable memory segments to reduce GPU memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_lazy.py \
  --config-name hetero_cell_bipartite_dango_gi_gh_069
'