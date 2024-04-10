#!/bin/bash
#SBATCH --mem=60g # up to 256 gb cpu... 256 doesn't work, 128 does... for cpu but not for gpu? 64 works for gpu.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbub-delta-gpu
#SBATCH --job-name=sweep
#SBATCH --time=48:00:00      # hh:mm:ss for the job, 48 hr max.
#SBATCH --constraint="projects&scratch"
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
# #SBATCH --output=/dev/null
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/slurm/output/%x_%j.out


module reset # drop modules and explicitly loado the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
source ~/.bashrc
cd /projects/bbub/mjvolk3/torchcell
pwd
lscpu
nvidia-smi 
cat /proc/meminfo
#module load anaconda3_cpu
module list  # job documentation and metadata
conda activate /projects/bbub/miniconda3/envs/torchcell

# TODO Needed for sweep... need to look into this more.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "job is starting on `hostname`"
wandb artifact cache cleanup 1GB
# 2>&1 stderr to stdout, extract sweep id.
#SWEEP_FILE=conf/sweep_test.yaml
SWEEP_FILE=experiments/smf-dmf-tmf-001/conf/deep_set-sweep_09.yaml
SWEEP_ID=$(wandb sweep $SWEEP_FILE 2>&1 | grep "Creating sweep with ID:" | awk -F': ' '{print $3}')
PROJECT_NAME=$(python torchcell/config.py $SWEEP_FILE)

echo "-----------------"
echo "SWEEP_ID: $SWEEP_ID"
echo "PROJECT_NAME: $PROJECT_NAME"
echo "-----------------"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "-----------------"


mkdir projects/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/agent_log/$SWEEP_ID

# Use for Agent logging. Only use in testing.
# Not sure if visivles devices scales with num nodes or resets to 0
for ((i=0; i<$SLURM_GPUS_ON_NODE*$SLURM_JOB_NUM_NODES; i++)); do
    (CUDA_VISIBLE_DEVICES=$i wandb agent zhao-group/$PROJECT_NAME/$SWEEP_ID > /projects/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/agent_log/$SWEEP_ID/agent-$PROJECT_NAME-$SWEEP_ID$i.log 2>&1) &
done
wait
