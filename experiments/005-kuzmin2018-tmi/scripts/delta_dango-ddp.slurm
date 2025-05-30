#!/bin/bash
#SBATCH --mem=64g #128g #243g # up to 256 gb cpu... 256 doesn't work, 128 does... for cpu but not for gpu? 64 works for gpu.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 #2
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbub-delta-gpu
#SBATCH --job-name=HCB_DDP
#SBATCH --time=2:00:00      # hh:mm:ss for the job, 48 hr max.
#SBATCH --constraint="scratch"
### GPU options ###
#SBATCH --gpus-per-node=1 #2
#SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
# #SBATCH --output=/dev/null
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell/experiments/005-kuzmin2018-tmi/slurm/output/%x_%j.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell/experiments/005-kuzmin2018-tmi/slurm/output/%x_%j.out


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

echo "-----------------"
echo "SWEEP_ID: $SWEEP_ID"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "PROJECT_NAME: $PROJECT_NAME"
echo "-----------------"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "-----------------"


mkdir /scratch/bbub/mjvolk3/torchcell/experiments/005-kuzmin2018-tmi/agent_log/$SLURM_JOB_ID

apptainer shell \
    --nv \
    --bind /projects/bbub:/projects/bbub \
    --bind /scratch/bbub/mjvolk3:/scratch/bbub/mjvolk3 \
    --bind /work:/work \
    rockylinux_9.sif

source /projects/bbub/miniconda3/bin/activate
conda activate torchcell

python /scratch/bbub/mjvolk3/torchcell/experiments/005-kuzmin2018-tmi/scripts/dango.py > /scratch/bbub/mjvolk3/torchcell/experiments/005-kuzmin2018-tmi/agent_log/$SLURM_JOB_ID/agent.log 2>&1

