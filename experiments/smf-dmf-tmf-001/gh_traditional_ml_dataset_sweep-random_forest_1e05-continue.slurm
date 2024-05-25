#!/bin/bash
#SBATCH --job-name=RF_1e05
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/slurm/output/%x_%j.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/slurm/output/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # Half of the total 128 CPUs
#SBATCH --mem=257G                    # Half of the total 514G memory
#SBATCH --gres=gpu:0                  # No GPUs
#SBATCH --time=36:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END

echo "Running on $(hostname)"

echo "Checking resource allocation..."
source ~/.bashrc  # Source a minimal environment, adjust to your shell if needed
conda activate /scratch/bbub/miniconda3/envs/torchcell

# Display the allocated resources
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE"

# Check system information
nproc
free -h
lscpu
cat /proc/meminfo

# Test writing to /scratch
SCRATCH_FILE="/scratch/test_file_${SLURM_JOB_ID}.txt"
echo "Writing to $SCRATCH_FILE"
echo "This is a test file for job ${SLURM_JOB_ID}" > $SCRATCH_FILE
ls -l $SCRATCH_FILE
cat $SCRATCH_FILE

echo "Sleeping for 30 seconds..."
sleep 30
echo "Finished sleeping."

# Check the content of the test file again
echo "Contents of $SCRATCH_FILE after sleep:"
cat $SCRATCH_FILE

# Job specific settings
echo "job is starting on $(hostname)"
wandb artifact cache cleanup 1GB

SWEEP_ID=p3afkc7b
PROJECT_NAME=torchcell_smf-dmf-tmf-001_trad-ml_random-forest_1e05

echo "-----------------"
echo "SWEEP_ID: $SWEEP_ID"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "PROJECT_NAME: $PROJECT_NAME"
echo "-----------------"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "-----------------"

mkdir -p /scratch/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/agent_log/$SLURM_JOB_ID

for ((i=0; i<$SLURM_NTASKS; i++)); do
    (wandb agent zhao-group/$PROJECT_NAME/$SWEEP_ID > /scratch/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/agent_log/$SLURM_JOB_ID/agent-$PROJECT_NAME-$SWEEP_ID-$i.log 2>&1) &
done

wait
