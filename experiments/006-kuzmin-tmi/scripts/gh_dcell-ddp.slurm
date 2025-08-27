#!/bin/bash
#SBATCH -p main
#SBATCH --mem=480g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --job-name=opt_DCell
#SBATCH --time=300-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

module purge
source ~/.bashrc
# Apptainer is installed system-wide, no module needed
# module load singularity

cd /home/michaelvolk/Documents/projects/torchcell

# Set environment variables needed for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Create scratch directory if it doesn't exist
mkdir -p /home/michaelvolk/scratch/projects/torchcell

# Launch with torchrun for multi-GPU training
apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
  --env PYTHONUNBUFFERED=1 \
  /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -c "
source $HOME/miniconda3/bin/activate
conda activate torchcell
export PYTHONUNBUFFERED=1
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dcell.py \
  --config-name dcell_kuzmin2018_tmi
"

#/home/a-m/mjvolk3/projects/torchcell/experiments/005-kuzmin2018-tmi/conf/dcell_kuzmin2018_tmi_maxing_resources.yaml
# experiments/005-kuzmin2018-tmi/conf/dcell_kuzmin2018_tmi_cpu.yaml


