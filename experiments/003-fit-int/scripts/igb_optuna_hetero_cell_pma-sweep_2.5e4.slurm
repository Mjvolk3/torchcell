#!/bin/bash
#SBATCH -p mmli
#SBATCH --mem=4g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=HCP_launcher
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out

module purge
source ~/.bashrc
conda activate torchcell

cd /home/a-m/mjvolk3/projects/torchcell

python experiments/003-fit-int/scripts/hetero_cell_pma.py \
  --config-dir experiments/003-fit-int/conf \
  --config-name igb_optuna-hetero_cell_pma-sweep_2.5e4 \
  --multirun