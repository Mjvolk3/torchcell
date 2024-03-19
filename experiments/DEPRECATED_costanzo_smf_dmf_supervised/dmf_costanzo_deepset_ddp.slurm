#!/bin/bash
#SBATCH --mem=243g # up to 256 gb gpu... max is only 243gb... 256 doesn't work. 128 for cpu.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbub-delta-gpu
#SBATCH --job-name=TC
#SBATCH --time=48:00:00      # hh:mm:ss for the job
#SBATCH --constraint="projects&scratch"
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/slurm/output/%x_%j.out
# #SBATCH --output=/dev/null
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/slurm/error/%x_%j.out

module reset # drop modules and explicitly load the ones needed
source ~/.bashrc
cd /projects/bbub/mjvolk3/torchcell
pwd
#module list  # job documentation
conda activate /projects/bbub/miniconda3/envs/torchcell
#echo "job is starting on `hostname`"
#wandb artifact cache cleanup 1GB
srun python experiments/costanzo_smf_dmf_supervised/dmf_costanzo_deepset.py

