#!/bin/bash
#SBATCH -p mmli
#SBATCH --mem=480g
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=HCB_DDP
#SBATCH --time=300-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out

module purge
source ~/.bashrc
conda activate torchcell

cd /home/a-m/mjvolk3/projects/torchcell

#python /home/a-m/mjvolk3/projects/torchcell/experiments/003-fit-int/scripts/hetero_cell_nsa.py

# Replace the direct Python call with torchrun
#torchrun --nproc_per_node=4 --nnodes=1 /home/a-m/mjvolk3/projects/torchcell/experiments/003-fit-int/scripts/hetero_cell_nsa.py

srun --ntasks=4 --ntasks-per-node=4 python /home/a-m/mjvolk3/scratch/torchcell/experiments/003-fit-int/scripts/hetero_cell_bipartite.py