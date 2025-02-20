#!/bin/bash
#SBATCH -p mmli
#SBATCH --mem=100g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=ICA_grid
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/003-fit-int/slurm/output/%x_%j.err

module purge
source ~/.bashrc

cd /home/a-m/mjvolk3/projects/torchcell
pwd
lscpu
nvidia-smi
cat /proc/meminfo

conda activate torchcell

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "Job starting on $(hostname)"
echo "-----------------"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "-----------------"

mkdir -p ~/scratch/torchcell/experiments/003-fit-int/agent_log/$SLURM_JOB_ID

NUM=3

# Run different configs in parallel
(CUDA_VISIBLE_DEVICES=0 python experiments/003-fit-int/scripts/isomorphic_cell_attentional.py --config-name isomorphic_cell_attentional-igb_$NUM > ~/scratch/torchcell/experiments/003-fit-int/agent_log/$SLURM_JOB_ID/gpu_$NUM.log 2>&1) &

wait

echo "All jobs completed"