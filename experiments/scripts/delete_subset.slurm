#!/bin/bash
#SBATCH --mem=2g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1  # Changed to 1 as deleting files usually doesn't require much CPU
#SBATCH --array=1-100 # creates 100 array tasks
#SBATCH --partition=cpu
#SBATCH --account=bbub-delta-cpu
#SBATCH --job-name=delete_files
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch&projects"
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/slurm/output/%x_%j.out
##SBATCH --error=/projects/bbub/mjvolk3/torchcell/slurm/output/%x_%j.out

module reset # drop modules and explicitly load the ones needed
source ~/.bashrc
cd /projects/bbub/mjvolk3/torchcell  # Change to directory where delete_subset.py is located if different
pwd
lscpu
module list  # job documentation and metadata
conda activate /projects/bbub/miniconda3/envs/torchcell  # Activate the necessary environment

python src/torchcell/delete_subset.py /scratch/bbub/mjvolk3/torchcell/data/scerevisiae/costanzo2016_1 --task-id ${SLURM_ARRAY_TASK_ID} --total-tasks 10  # Replace '/path/to/directory' with your directory

