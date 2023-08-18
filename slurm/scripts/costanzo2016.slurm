#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu
#SBATCH --account=bbub-delta-cpu
#SBATCH --job-name=costanzo
#SBATCH --time=2:00:00
#SBATCH --constraint="projects"
#SBATCH --mail-user=mjvolk3@illinois.edus
#SBATCH --mail-type="END"
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/slurm/error/%x_%j.out

module reset
source ~/.bashrc
cd /projects/bbub/mjvolk3/torchcell
pwd
lscpu
cat /proc/meminfo
module list
conda activate /projects/bbub/miniconda3/envs/torchcell

echo "job is starting on `hostname`"
wandb artifact cache cleanup 1GB
####
python src/torchcell/datasets/scerevisiae/costanzo2016.py