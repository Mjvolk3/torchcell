#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # Increase the number of tasks per node to the desired number of agents
#SBATCH --cpus-per-task=128  # Adjust the number of CPUs per task based on your requirements
#SBATCH --partition=cpu
#SBATCH --account=bbub-delta-cpu
#SBATCH --job-name=CPU_PERF
#SBATCH --time=2:00:00
#SBATCH --constraint="scratch"
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell/experiments/002-dmi-tmi/slurm/output/%x_%j.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell/experiments/002-dmi-tmi/slurm/output/%x_%j.out

module resets
source ~/.bashrc
cd /scratch/bbub/mjvolk3/torchcell
pwd
lscpu
cat /proc/meminfo
module list

conda activate /scratch/bbub/miniconda3/envs/torchcell
echo "job is starting on `hostname`"

##

srun python /scratch/bbub/mjvolk3/torchcell/experiments/002-dmi-tmi/scripts/cpu_performance_benchmark.py