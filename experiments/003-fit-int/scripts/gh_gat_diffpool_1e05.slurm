#!/bin/bash
#SBATCH --job-name=001-GD_1e05-codon_frequency
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=243G
#SBATCH --gres=gpu:4
#SBATCH --time=365-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/scratch/projects/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out

echo "Running on $(hostname)"
echo "Checking resource allocation"

source ~/.zshrc
conda activate ~/miniconda3/envs/torchcell

# Check if wandb is available
echo "Checking if wandb is available..."
which wandb
if [ $? -ne 0 ]; then
    echo "wandb command not found. Exiting."
    exit 1
fi

# Display the allocated resources
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"

# Check system information
nproc
free -h
lscpu
cat /proc/meminfo
nvidia-smi

python experiments/003-fit-int/scripts/gat_diffpool.py +config_name=001-gat_diffpool-1e05-codon_frequency



