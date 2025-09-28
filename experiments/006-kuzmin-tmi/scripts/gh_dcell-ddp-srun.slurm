#!/bin/bash
#SBATCH --job-name=opt_DCell_srun
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/opt_DCell_srun_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/opt_DCell_srun_%j.out
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=500G

# Change to the torchcell directory
cd /home/michaelvolk/Documents/projects/torchcell || exit 1

# Set environment variables needed for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Add the project root to Python path
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# Enable NCCL debugging to diagnose issues
export NCCL_DEBUG=INFO
#export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Keep CUDA_LAUNCH_BLOCKING=1 for now to help debug GPU issues
# TODO: Set to 0 in production for better performance (10-30% speedup)
export CUDA_LAUNCH_BLOCKING=1

# Verify GPU access before starting
echo "Checking GPU availability before training..."
nvidia-smi

# Launch with srun to ensure proper process distribution
# Match the working pattern from other clusters - no --gpus-per-task
srun --ntasks-per-node=4 bash -c "
source ~/miniconda3/bin/activate
conda activate torchcell
python /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dcell.py --config-name dcell_kuzmin2018_tmi
"