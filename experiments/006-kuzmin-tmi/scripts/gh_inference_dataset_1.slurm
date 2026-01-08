#!/bin/bash
#SBATCH -p main
#SBATCH --mem=400g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=006-inference-dataset
#SBATCH --time=365-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

# Change to the project root directory (required for relative paths in Python script)
cd /home/michaelvolk/Documents/projects/torchcell || exit 1

# Activate conda environment directly
source ~/miniconda3/bin/activate
conda activate torchcell

# Add project to PYTHONPATH
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# Suppress warnings for cleaner output
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

# Calculate num_workers from SLURM CPU allocation (reserve 4 for main thread + overhead)
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 4))

# Run the inference dataset creation script
~/miniconda3/envs/torchcell/bin/python \
  experiments/006-kuzmin-tmi/scripts/inference_dataset_1.py

