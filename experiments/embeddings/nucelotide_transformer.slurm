#!/bin/bash
#SBATCH --mem=64g # up to 256 gb cpu... 256 doesn't work, 128 does... for cpu but not for gpu? 64 works for gpu.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbub-delta-gpu
#SBATCH --job-name=embed
#SBATCH --time=18:00:00      # hh:mm:ss for the job, 48 hr max.
#SBATCH --constraint="projects&scratch"
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
# #SBATCH --output=/dev/null
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/embeddings/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/embeddings/slurm/output/%x_%j.out


module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
source ~/.bashrc
cd /projects/bbub/mjvolk3/torchcell
pwd
lscpu
nvidia-smi 
cat /proc/meminfo
#module load anaconda3_cpu
module list  # job documentation and metadata
conda activate /projects/bbub/miniconda3/envs/torchcell
echo "Starting Embedding Computation..."
python experiments/embeddings/compute_nucleotide_transformer_embeddings.py