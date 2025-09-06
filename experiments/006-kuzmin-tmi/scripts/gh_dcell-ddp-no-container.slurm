#!/bin/bash
#SBATCH --job-name=opt_DCell
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/opt_DCell_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/opt_DCell_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=500G

# Change to the torchcell directory
cd /home/michaelvolk/Documents/projects/torchcell || exit 1

# Set environment variables needed for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Enable NCCL debugging to diagnose issues
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# Add the project root to Python path
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

# Clean up any zombie GPU processes from current user BEFORE starting
echo "Cleaning up any orphaned GPU processes for user $USER..."
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    # Check if this PID belongs to current user
    if [ -n "$pid" ]; then
        pid_owner=$(ps -o user= -p $pid 2>/dev/null | tr -d ' ')
        if [ "$pid_owner" = "$USER" ]; then
            echo "Killing orphaned GPU process $pid owned by $USER"
            kill -9 $pid 2>/dev/null
        else
            echo "Skipping GPU process $pid owned by $pid_owner"
        fi
    fi
done
sleep 2

# Reset GPUs if they are in a bad state or have zombie contexts
echo "Checking and resetting GPUs if needed..."
for gpu_id in 0 1 2 3; do
    # Check if GPU is in ERR state
    gpu_check=$(nvidia-smi -i $gpu_id 2>&1)
    if echo "$gpu_check" | grep -q "ERR"; then
        echo "GPU $gpu_id is in ERR state, attempting reset..."
        # Try to reset the GPU with sudo if available
        if command -v sudo &> /dev/null && sudo -n true 2>/dev/null; then
            echo "Using sudo to reset GPU $gpu_id..."
            sudo nvidia-smi -i $gpu_id -r 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "GPU $gpu_id reset successfully"
                sleep 3
            else
                echo "Failed to reset GPU $gpu_id even with sudo"
            fi
        else
            # Try without sudo
            nvidia-smi -i $gpu_id -r 2>/dev/null || echo "Could not reset GPU $gpu_id (needs sudo privileges)"
        fi
    else
        echo "GPU $gpu_id is healthy"
    fi
done
sleep 2

# Verify GPU access
echo "Checking GPU availability..."
nvidia-smi
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Run torchrun directly (no container)
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dcell.py \
  --config-name dcell_kuzmin2018_tmi