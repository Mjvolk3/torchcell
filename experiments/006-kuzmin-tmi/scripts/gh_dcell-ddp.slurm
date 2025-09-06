#!/bin/bash
#SBATCH --job-name=opt_DCell
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/opt_DCell_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/opt_DCell_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=500G
#SBATCH --signal=B:TERM@60  # Send SIGTERM 60 seconds before SIGKILL

# Cleanup function to handle job cancellation gracefully
cleanup() {
    echo "Received termination signal, initiating graceful shutdown..."
    
    # Send SIGTERM to our background process (the one we started with &)
    if [ -n "$PID" ]; then
        echo "Sending SIGTERM to job process $PID"
        kill -TERM $PID 2>/dev/null
        
        # Wait up to 50 seconds for our process to terminate
        for i in {1..50}; do
            if ! kill -0 $PID 2>/dev/null; then
                echo "Job process terminated gracefully"
                break
            fi
            echo "Waiting for process to terminate... ($i/50)"
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 $PID 2>/dev/null; then
            echo "Force killing job process $PID"
            kill -KILL $PID 2>/dev/null
        fi
    fi
    
    # Kill only child processes of this script
    pkill -KILL -P $$ 2>/dev/null
    
    # Clean shared memory (only for this job)
    rm -f /dev/shm/*torch*${SLURM_JOB_ID}* 2>/dev/null
    rm -f /dev/shm/*nccl*${SLURM_JOB_ID}* 2>/dev/null
    
    echo "Cleanup completed"
    exit 0
}

# Register the cleanup function to run on SIGTERM and SIGINT
trap cleanup SIGTERM SIGINT

# Change to the experiments directory
cd /home/michaelvolk/Documents/projects/torchcell/experiments || exit 1

# Set environment variables needed for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Enable NCCL debugging to diagnose issues
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1  # Enable async error handling

# Launch with torchrun for multi-GPU training (run in background to catch signals)
apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
  --env PYTHONUNBUFFERED=1 \
  --env NCCL_DEBUG=$NCCL_DEBUG \
  --env NCCL_ASYNC_ERROR_HANDLING=$NCCL_ASYNC_ERROR_HANDLING \
  /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc '
# Add the project root to Python path
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/dcell.py \
  --config-name dcell_kuzmin2018_tmi
' &

# Store the PID
PID=$!

# Wait for the process to complete
wait $PID
EXIT_CODE=$?

echo "Job completed with exit code: $EXIT_CODE"
exit $EXIT_CODE