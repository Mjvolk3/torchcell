#!/bin/bash
#SBATCH --mem=8g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # Increase the number of tasks per node to the desired number of agents
#SBATCH --cpus-per-task=1  # Adjust the number of CPUs per task based on your requirements
#SBATCH --partition=cpu
#SBATCH --account=bbub-delta-cpu
#SBATCH --job-name=wandb_test
#SBATCH --time=4:00:00
#SBATCH --constraint="scratch"
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/slurm/output/%x_%j.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/slurm/output/%x_%j.out

module reset
source ~/.bashrc
cd /scratch/bbub/mjvolk3/torchcell
pwd
lscpu
cat /proc/meminfo
module list

conda activate /projects/bbub/miniconda3/envs/torchcell

python experiments/smf-dmf-tmf-001/wandb_test.py