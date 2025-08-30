#!/bin/bash
#SBATCH --job-name=opt_DCell
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/opt_DCell_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/error/opt_DCell_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=200G

# Change to the experiments directory
cd /home/michaelvolk/Documents/projects/torchcell/experiments || exit 1

# Set environment variables needed for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Launch with torchrun for multi-GPU training
apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
  --env PYTHONUNBUFFERED=1 \
  /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc '
# Add the project root to Python path
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dcell.py \
  --config-name dcell_kuzmin2018_tmi
'