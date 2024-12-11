#!/bin/bash
#SBATCH --job-name=CPU_PERF
#SBATCH --ntasks=4         # One per NUMA node
#SBATCH --cpus-per-task=32 # Full CPUs per NUMA node
#SBATCH --mem=240G
#SBATCH --time=365-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/scratch/projects/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out

export OMP_NUM_THREADS=32
export OMP_PLACES=cores   
export OMP_PROC_BIND=close

echo "Running on $(hostname)"
source ~/.zshrc
conda activate ~/miniconda3/envs/torchcell

numactl --hardware              # Display NUMA config
python /home/michaelvolk/Documents/projects/torchcell/experiments/003-fit-int/scripts/cpu_performance_benchmark_numa_aware.py