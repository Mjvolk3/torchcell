#!/bin/bash
#!/bin/bash
#SBATCH --job-name=CGDDB_5e4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --time=365-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/scratch/projects/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out

# FLAG - trying for ddp [trbl](https://github.com/Lightning-AI/pytorch-lightning/issues/18221)
#export NCCL_P2P_DISABLE=1

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
mkdir /scratch/projects/torchcell/experiments/003-fit-int/agent_log/$SLURM_JOB_ID

# Use for Agent logging. Only use in testing.
# Not sure if visivles devices scales with num nodes or resets to 0
#for ((i=0; i<$SLURM_GPUS_ON_NODE*$SLURM_JOB_NUM_NODES; i++)); do
#    (CUDA_VISIBLE_DEVICES=$i wandb agent zhao-group/$PROJECT_NAME/$SWEEP_ID > /scratch/bbub/mjvolk3/torchcell/experiments/003-fit-int/agent_log/$SLURM_JOB_ID/agent-$PROJECT_NAME-$SWEEP_ID$i.log 2>&1) &
#done
#wait

#wandb agent zhao-group/$PROJECT_NAME/$SWEEP_ID > /scratch/bbub/mjvolk3/torchcell/experiments/003-fit-int/agent_log/$SLURM_JOB_ID/agent-$PROJECT_NAME-$SWEEP_ID$i.log 2>&1
srun python /home/michaelvolk/Documents/projects/torchcell/experiments/003-fit-int/scripts/cell_gin_diffpool_dense_binary.py > /scratch/projects/torchcell/experiments/003-fit-int/agent_log/$SLURM_JOB_ID/agent.log 2>&1

