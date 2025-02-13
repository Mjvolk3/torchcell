#!/bin/bash
#SBATCH -p mmli
#SBATCH --mem=256g
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=ICA_2.5e4
#SBATCH --time=48:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=~/scratch/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out
#SBATCH --error=~/scratch/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out


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
wandb artifact cache cleanup 1GB

SWEEP_ID=
PROJECT_NAME=torchcell_003-fit-int_isomorphic_cell_attentional_2.5e4

echo "-----------------"
echo "SWEEP_ID: $SWEEP_ID"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "PROJECT_NAME: $PROJECT_NAME"
echo "-----------------"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "-----------------"

mkdir -p /home/a-m/mjvolk3/projects/torchcell/experiments/003-fit-int/agent_log/$SLURM_JOB_ID


for ((i=0; i<4; i++)); do
    (CUDA_VISIBLE_DEVICES=$i wandb agent zhao-group/$PROJECT_NAME/$SWEEP_ID > /home/a-m/mjvolk3/projects/torchcell/experiments/003-fit-int/agent_log/$SLURM_JOB_ID/agent-$PROJECT_NAME-$SWEEP_ID$i.log 2>&1) &
done
wait