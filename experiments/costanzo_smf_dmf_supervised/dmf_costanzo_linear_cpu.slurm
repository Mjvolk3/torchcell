#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu
#SBATCH --account=bbub-delta-cpu
#SBATCH --job-name=dmf_costanzo_linear
#SBATCH --time=48:00:00
#SBATCH --constraint="projects&scratch"
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/slurm/error/%x_%j.out

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
source ~/.bashrc
cd /projects/bbub/mjvolk3/torchcell
pwd
lscpu
#module load anaconda3_cpu
module list  # job documentation and metadata
#conda activate torchcell
conda activate /projects/bbub/miniconda3/envs/torchcell
python experiments/costanzo_smf_dmf_supervised/dmf_costanzo_linear.py