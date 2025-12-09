#!/bin/bash
#SBATCH -p cabbi
#SBATCH --mem=400g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=028-006-ECGT
#SBATCH --time=300-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

module purge
source ~/.bashrc
module load singularity

# Change to the experiments directory
cd /home/a-m/mjvolk3/projects/torchcell/experiments || exit 1

# Set environment variables needed for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Calculate num_workers from SLURM CPU allocation (reserve 2 for overhead)
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 2))

# Run training with singularity container
singularity exec --nv /home/a-m/mjvolk3/projects/torchcell/rockylinux_9.sif bash -c "
source \$HOME/miniconda3/bin/activate
conda activate torchcell

# Add project to PYTHONPATH
export PYTHONPATH=\"/home/a-m/mjvolk3/projects/torchcell:\$PYTHONPATH\"

# CUDA optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export PYTHONUNBUFFERED=1

# Run training with torchrun
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  /home/a-m/mjvolk3/projects/torchcell/experiments/006-kuzmin-tmi/scripts/equivariant_cell_graph_transformer.py \
  --config-name equivariant_cell_graph_transformer_cabbi_028 \
  data_module.num_workers=\$NUM_WORKERS
"

