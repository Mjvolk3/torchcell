#!/bin/bash
#SBATCH --job-name=CPU_PERF
#SBATCH --ntasks=1         # Single task since we're using ProcessPoolExecutor
#SBATCH --cpus-per-task=128 # Use all available cores
#SBATCH --mem=240G
#SBATCH --time=365-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/scratch/projects/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out

# Let numpy handle thread allocation since we're using ProcessPoolExecutor
export OMP_NUM_THREADS=1    # Limit threads since we're using processes
export OMP_PLACES=cores   
export OMP_PROC_BIND=close

echo "Running on $(hostname)"
source ~/.zshrc
conda activate ~/miniconda3/envs/torchcell

numactl --hardware
python /home/michaelvolk/Documents/projects/torchcell/experiments/003-fit-int/scripts/cpu_performance_benchmark_multiprocess.py