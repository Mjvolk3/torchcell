#!/bin/bash
#SBATCH --mem=243g # up to 256 gb cpu... 256 doesn't work, 128 does... for cpu but not for gpu? 64 works for gpu.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbub-delta-gpu
#SBATCH --job-name=GD_5e4
#SBATCH --time=48:00:00      # hh:mm:ss for the job, 48 hr max.
#SBATCH --constraint="scratch"
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
# #SBATCH --output=/dev/null
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out


module reset # drop modules and explicitly loado the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
source ~/.bashrc

cd /scratch/bbub/mjvolk3/torchcell
pwd
lscpu
nvidia-smi 
cat /proc/meminfo
#module load anaconda3_cpu
module list  # job documentation and metadata
conda activate /scratch/bbub/miniconda3/envs/torchcell

# TODO Needed for sweep... need to look into this more.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "job is starting on `hostname`"
wandb artifact cache cleanup 1GB

SWEEP_ID=ug52ua9d
PROJECT_NAME=torchcell_003-fit-int_gat_diffpool_5e4

echo "-----------------"
echo "SWEEP_ID: $SWEEP_ID"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "PROJECT_NAME: $PROJECT_NAME"
echo "-----------------"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "-----------------"


mkdir /scratch/bbub/mjvolk3/torchcell/experiments/003-fit-int/agent_log/$SLURM_JOB_ID

# Use for Agent logging. Only use in testing.
# Not sure if visivles devices scales with num nodes or resets to 0
for ((i=0; i<$SLURM_GPUS_ON_NODE*$SLURM_JOB_NUM_NODES; i++)); do
    (CUDA_VISIBLE_DEVICES=$i wandb agent zhao-group/$PROJECT_NAME/$SWEEP_ID > /scratch/bbub/mjvolk3/torchcell/experiments/003-fit-int/agent_log/$SLURM_JOB_ID/agent-$PROJECT_NAME-$SWEEP_ID$i.log 2>&1) &
done
wait
