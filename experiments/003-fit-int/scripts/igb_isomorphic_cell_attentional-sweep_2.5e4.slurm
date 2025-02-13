#!/bin/bash
#SBATCH -p mmli
#SBATCH --mem=256g
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=ICA_grid
#SBATCH --time=48:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=~/scratch/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out
#SBATCH --error=~/scratch/torchcell/experiments/003-fit-int/slurm/output/%x_%j.err

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

mkdir -p ~/scratch/torchcell/experiments/003-fit-int/slurm/logs/$SLURM_JOB_ID

# Run different configs in parallel
(CUDA_VISIBLE_DEVICES=0 python experiments/003-fit-int/scripts/isomorphic_cell_attentional.py --config-name isomorphic_cell_attentional-igb_0 > ~/scratch/torchcell/experiments/003-fit-int/slurm/logs/$SLURM_JOB_ID/gpu_0.log 2>&1) &

(CUDA_VISIBLE_DEVICES=1 python experiments/003-fit-int/scripts/isomorphic_cell_attentional.py --config-name isomorphic_cell_attentional-igb_1 > ~/scratch/torchcell/experiments/003-fit-int/slurm/logs/$SLURM_JOB_ID/gpu_1.log 2>&1) &

(CUDA_VISIBLE_DEVICES=2 python experiments/003-fit-int/scripts/isomorphic_cell_attentional.py --config-name isomorphic_cell_attentional-igb_2 > ~/scratch/torchcell/experiments/003-fit-int/slurm/logs/$SLURM_JOB_ID/gpu_2.log 2>&1) &

(CUDA_VISIBLE_DEVICES=3 python experiments/003-fit-int/scripts/isomorphic_cell_attentional.py --config-name isomorphic_cell_attentional-igb_3 > ~/scratch/torchcell/experiments/003-fit-int/slurm/logs/$SLURM_JOB_ID/gpu_3.log 2>&1) &

wait

echo "All jobs completed"