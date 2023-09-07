#!/bin/bash
#SBATCH --mem=16g # up to 256 gb cpu... 256 doesn't work, 128 does... for cpu but not for gpu? 64 works for gpu.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbub-delta-gpu
#SBATCH --job-name=1e5
#SBATCH --time=48:00:00      # hh:mm:ss for the job, Couldn't do 72 hrs.
#SBATCH --constraint="projects&scratch"
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/slurm/output/%x_%j.out
# #SBATCH --output=/dev/null
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/slurm/error/%x_%j.out

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
source ~/.bashrc
cd /projects/bbub/mjvolk3/torchcell
pwd
lscpu
nvidia-smi
#module load anaconda3_cpu
module list  # job documentation and metadata
conda activate /projects/bbub/miniconda3/envs/torchcell
echo "job is starting on `hostname`"
wandb artifact cache cleanup 1GB
srun python experiments/dmf_costanzo_deepset.py

