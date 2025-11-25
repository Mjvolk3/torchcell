#!/bin/bash
#SBATCH -p main
#SBATCH --mem=500g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=006-DANGO-PROFILE-DDP
#SBATCH --time=4:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

# Change to the experiments directory
cd /home/michaelvolk/Documents/projects/torchcell/experiments || exit 1

# Set environment variables needed for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Profile DANGO with DDP for fair comparison with lazy hetero model
# This will help identify why DANGO achieves ~10 it/s vs 0.42 it/s for lazy
# Uses PyTorch profiler with same protocol as lazy hetero:
#  - Wait 600 steps for steady-state
#  - Profile 25 steps (601-625)
#  - 4 GPUs with DDP
#  - Categorized analysis of bottlenecks

# Launch with torchrun for multi-GPU training
apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
  --env PYTHONUNBUFFERED=1 \
  /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc '
# Add the project root to Python path
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# CUDA debugging flags (set to 0 for maximum performance)
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dango.py \
  --config-name dango_kuzmin2018_tmi_string12_0_profile
'

echo "Profiling complete. To analyze results:"
echo "1. Find the latest profile in: /scratch/projects/torchcell-scratch/data/torchcell/experiments/006-kuzmin-tmi/profiler_output/dango_*/"
echo "2. Run analysis script:"
echo "   python experiments/006-kuzmin-tmi/scripts/analyze_profile_detailed.py <trace_file.pt.trace.json>"
