#!/bin/bash
#SBATCH -p main
#SBATCH --mem=200g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=006-DANGO-PROFILE
#SBATCH --time=2:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

# Change to the experiments directory
cd /home/michaelvolk/Documents/projects/torchcell/experiments || exit 1

# Profile DANGO baseline for comparison with lazy hetero model
# This will help identify why DANGO achieves 10 it/s vs 0.42 it/s for lazy
# Uses PyTorch profiler to analyze:
#  - Data loading time
#  - Model forward/backward passes
#  - DDP communication overhead
#  - GPU/CPU utilization

# Single GPU for clean profiling
export CUDA_VISIBLE_DEVICES=0

# Launch with apptainer
apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
  --env PYTHONUNBUFFERED=1 \
  /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc '
# Add the project root to Python path
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

# Run DANGO with profiling configuration
python /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dango.py \
  --config-name dango_kuzmin2018_tmi_string12_0_profile
'

echo "Profiling complete. To analyze results:"
echo "1. Find the latest profile in: /scratch/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/profiler_output/dango_*/"
echo "2. Run analysis script:"
echo "   python experiments/006-kuzmin-tmi/scripts/analyze_profile_detailed.py <trace_file.pt.trace.json>"