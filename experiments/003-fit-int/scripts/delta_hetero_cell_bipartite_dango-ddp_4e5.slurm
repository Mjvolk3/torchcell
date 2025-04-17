#!/bin/bash
#SBATCH --mem=128g #243g
#SBATCH --nodes=1
#SBATCH --ntasks=2 # Changed from ntasks-per-node
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbub-delta-gpu
#SBATCH --job-name=HCBD_DDP
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-task=1  # Explicit GPU allocation
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out

module reset
source ~/.bashrc
cd /scratch/bbub/mjvolk3/torchcell

# Set distributed training environment variables
export MASTER_ADDR=$(hostname -f)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Rest of your existing setup
conda activate /scratch/bbub/miniconda3/envs/torchcell
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
echo "job is starting on `hostname`"
wandb artifact cache cleanup 1GB

echo "-----------------"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "-----------------"

mkdir -p /scratch/bbub/mjvolk3/torchcell/experiments/003-fit-int/agent_log/$SLURM_JOB_ID

# Launch with srun to ensure proper process distribution
srun --ntasks-per-node=2 apptainer exec \
    --nv \
    --bind /projects/bbub:/projects/bbub \
    --bind /scratch/bbub/mjvolk3:/scratch/bbub/mjvolk3 \
    --bind /work:/work \
    rockylinux_9.sif \
    bash -c "source /projects/bbub/miniconda3/bin/activate && \
    conda activate torchcell && \
    python /scratch/bbub/mjvolk3/torchcell/experiments/003-fit-int/scripts/hetero_cell_bipartite_dango.py"