#!/bin/bash
#SBATCH --mem=64g # up to 256 gb cpu... 256 doesn't work, 128 does... for cpu but not for gpu? 64 works for gpu.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbtp-delta-gpu
#SBATCH --job-name=sweep-cont
#SBATCH --time=48:00:00      # hh:mm:ss for the job, Couldn't do 72 hrs.
#SBATCH --constraint="projects"
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
# #SBATCH --output=/projects/bbtp/mjvolk3/Parameter_Estimation/slurm/output/%x_%j.out
### #SBATCH --error=/projects/bbtp/mjvolk3/Parameter_Estimation/slurm/output/%x_%j.out
# #SBATCH --output=/dev/null


module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
source ~/.bashrc
cd /projects/bbtp/mjvolk3/Parameter_Estimation
pwd
lscpu
nvidia-smi
#module load anaconda3_cpu
module list  # job documentation and metadata
conda activate env-param-delta

# TODO Needed for sweep... need to look into this more.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "job is starting on `hostname`"
wandb artifact cache cleanup 1GB
# 2>&1 stderr to stdout, extract sweep id.
SWEEP_ID=24qn6580
PROJECT_NAME=torchcell

echo $SLURM_GPUS_ON_NODE
echo $SLURM_JOB_NUM_NODES
echo $"-----------------"

#for ((i=0; i<$SLURM_GPUS_ON_NODE*$SLURM_JOB_NUM_NODES; i++)); do
#    (CUDA_VISIBLE_DEVICES=$i wandb agent zhao-group/$PROJECT_NAME/$SWEEP_ID) &
#done
#wait

#Use for Agent logging. Only use in testing.
for ((i=0; i<$SLURM_GPUS_ON_NODE*$SLURM_JOB_NUM_NODES; i++)); do
    (CUDA_VISIBLE_DEVICES=$i wandb agent zhao-group/$PROJECT_NAME/$SWEEP_ID > agent-$SLURM_JOBID-$PROJECT_NAME-$SWEEP_ID$i.log 2>&1) &
done
wait