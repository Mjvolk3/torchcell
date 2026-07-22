#!/bin/bash
#SBATCH --mem=243g
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbub-delta-gpu
#SBATCH --job-name=019-MTCGT
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch"
#SBATCH --gres=gpu:4
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out

module reset
source ~/.bashrc
cd /scratch/bbub/mjvolk3/torchcell

export MASTER_ADDR=$(hostname -f)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
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

srun --ntasks-per-node=4 apptainer exec \
    --nv \
    --bind /projects/bbub:/projects/bbub \
    --bind /scratch/bbub/mjvolk3:/scratch/bbub/mjvolk3 \
    rockylinux_9.sif \
    bash -c "source /projects/bbub/miniconda3/bin/activate && \
    conda activate torchcell && \
    export PYTHONPATH=/scratch/bbub/mjvolk3/torchcell:\$PYTHONPATH && \
    export PYTHONUNBUFFERED=1 && \
    python /scratch/bbub/mjvolk3/torchcell/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py \
    --config-name delta_train_cgt_multitask_000"
