#!/bin/bash
#SBATCH --job-name=DS_1e03_codon_frequency
#SBATCH --ntasks-per-node=3 # Set to the desired number of tasks per node
#SBATCH --cpus-per-task=16 # Adjust the number of CPUs per task based on your requirements
#SBATCH --mem=192G # Total memory for the job (adjusted based on your available memory)
#SBATCH --gres=gpu:3 # Request 3 GPUs for the job
#SBATCH --time=365-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/scratch/projects/torchcell/experiments/smf-dmf-tmf-001/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell/experiments/smf-dmf-tmf-001/slurm/output/%x_%j.out

echo "Running on $(hostname)"
echo "Checking resource allocation..."

source ~/.zshrc # Source a minimal environment, adjust to your shell if needed
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
echo "Allocated GPUs: $SLURM_GPUS" # Display the number of allocated GPUs

# Check system information
nproc
free -h
lscpu
cat /proc/meminfo
nvidia-smi # Display GPU information

# Job specific settings
echo "job is starting on $(hostname)"
wandb artifact cache cleanup 1GB

SWEEP_ID=mhbbftax
PROJECT_NAME=torchcell_smf-dmf-tmf-001_deep_set_1e03

echo "-----------------"
echo "SWEEP_ID: $SWEEP_ID"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "PROJECT_NAME: $PROJECT_NAME"
echo "-----------------"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "-----------------"

mkdir -p /scratch/projects/torchcell/experiments/smf-dmf-tmf-001/agent_log/$SLURM_JOB_ID

for ((i=0; i<$SLURM_NTASKS_PER_NODE; i++)); do
    echo "Starting wandb agent $i"
    (
        export CUDA_VISIBLE_DEVICES=$i # Set the CUDA_VISIBLE_DEVICES environment variable for each agent
        wandb agent zhao-group/$PROJECT_NAME/$SWEEP_ID > /scratch/projects/torchcell/experiments/smf-dmf-tmf-001/agent_log/$SLURM_JOB_ID/agent-$PROJECT_NAME-$SWEEP_ID-$i.log 2>&1
    ) &
    if [ $? -ne 0 ]; then
        echo "Failed to start wandb agent $i"
        exit 1
    fi
done

wait