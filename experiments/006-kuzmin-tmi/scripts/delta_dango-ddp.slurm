#!/bin/bash
#SBATCH --mem=64g #128g #243g # up to 256 gb cpu... 256 doesn't work, 128 does... for cpu but not for gpu? 64 works for gpu.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4  # One task per GPU for DDP
#SBATCH --cpus-per-task=8    # CPUs per task/GPU
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbub-delta-gpu
#SBATCH --job-name=DANGO_DDP_006
#SBATCH --time=48:00:00      # hh:mm:ss for the job, 48 hr max.
#SBATCH --constraint="scratch"
### GPU options ###
#SBATCH --gpus-per-node=4  # Total GPUs, will be distributed among tasks
#SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
# #SBATCH --output=/dev/null
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out


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


mkdir -p /scratch/bbub/mjvolk3/torchcell/experiments/006-kuzmin-tmi/agent_log/$SLURM_JOB_ID

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Launch with srun to ensure proper process distribution
srun --ntasks-per-node=4 --gpus-per-task=1 apptainer exec --nv \
    --bind /projects/bbub:/projects/bbub \
    --bind /scratch/bbub/mjvolk3:/scratch/bbub/mjvolk3 \
    --bind /work:/work \
    rockylinux_9.sif bash -c "
source /projects/bbub/miniconda3/bin/activate
conda activate torchcell
python /scratch/bbub/mjvolk3/torchcell/experiments/006-kuzmin-tmi/scripts/dango.py trainer.devices=4 > /scratch/bbub/mjvolk3/torchcell/experiments/006-kuzmin-tmi/agent_log/$SLURM_JOB_ID/agent_${SLURM_PROCID}.log 2>&1
"

