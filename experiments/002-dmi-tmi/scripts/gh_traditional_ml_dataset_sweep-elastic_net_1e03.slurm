#!/bin/bash
#SBATCH --job-name=EN_1e03
#SBATCH --ntasks=24               # Set to the desired number of tasks (agents)
#SBATCH --cpus-per-task=4           # Adjust the number of CPUs per task based on your requirements
#SBATCH --mem=930G                  # Total memory for the job
#SBATCH --gres=gpu:0                # No GPUs
#SBATCH --time=365-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/scratch/projects/torchcell/experiments/002-dmi-tmi/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell/experiments/002-dmi-tmi/slurm/output/%x_%j.out

echo "Running on $(hostname)"

echo "Checking resource allocation..."
source ~/.zshrc  # Source a minimal environment, adjust to your shell if needed
conda activate ~/miniconda3/envs/torchcell

# Check if wandb is available
echo "Checking if wandb is available..."
which wandb
if [ $? -ne 0 ]; then
    echo "wandb command not found. Exiting."
    exit 1
fi

# Display the allocated resources
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE"

# Check system information
nproc
free -h
lscpu
cat /proc/meminfo

# Job specific settings
echo "job is starting on $(hostname)"
wandb artifact cache cleanup 1GB

# This one is offline
SWEEP_ID=ljre8axa
PROJECT_NAME=torchcell_002-dmi-tmi_trad-ml_elastic-net_1e03

echo "-----------------"
echo "SWEEP_ID: $SWEEP_ID"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "PROJECT_NAME: $PROJECT_NAME"
echo "-----------------"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "-----------------"

mkdir -p /scratch/projects/torchcell/experiments/002-dmi-tmi/agent_log/$SLURM_JOB_ID

for ((i=0; i<$SLURM_NTASKS; i++)); do
    echo "Starting wandb agent $i"
    (wandb agent zhao-group/$PROJECT_NAME/$SWEEP_ID > /scratch/projects/torchcell/experiments/002-dmi-tmi/agent_log/$SLURM_JOB_ID/agent-$PROJECT_NAME-$SWEEP_ID-$i.log 2>&1) &
    if [ $? -ne 0 ]; then
        echo "Failed to start wandb agent $i"
        exit 1
    fi
done

wait
