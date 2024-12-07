#!/bin/bash
#SBATCH --job-name=CPU_PERF
#SBATCH --ntasks=1               # Set to the desired number of tasks (agents)
#SBATCH --cpus-per-task=128           # Adjust the number of CPUs per task based on your requirements
#SBATCH --mem=995G                  # Total memory for the job
#SBATCH --gres=gpu:0                # No GPUs
#SBATCH --time=365-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/scratch/projects/torchcell/experiments/002-dmi-tmi/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell/experiments/002-dmi-tmi/slurm/output/%x_%j.out

echo "Running on $(hostname)"

echo "Checking resource allocation..."
source ~/.zshrc  # Source a minimal environment, adjust to your shell if needed
conda activate ~/miniconda3/envs/torchcell

# Display the allocated resources
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE"

# Check system information
nproc
free -h
lscpu
cat /proc/meminfo

# Run Script
srun python /home/michaelvolk/Documents/projects/torchcell/torchcell/scratch/cpu_performance_benchmark.py
