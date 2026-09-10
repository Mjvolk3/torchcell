#!/bin/bash
#SBATCH --job-name=006-dcell-gpu-profile
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00

# One DCell training step under torch.profiler on one gilahyper GPU, at the cluster's
# per-GPU batch of 600 in bf16-mixed: panel e of FigS-dcell-training. The checkpoint is
# the cluster run's epoch-142 save (the one the training script kept, min val MSE),
# copied from IGB; it is loaded strictly as an architecture check before profiling.
# REPO is the checkout whose code runs (a worktree while the change is unlanded).

REPO="${REPO:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/dcell-gpu-profile}"
CKPT="${CKPT:-/scratch/projects/torchcell-scratch/models/checkpoints/igb/compute-5-7-1922684_944a22e8933d111d5d8539f9e3ef5983a5ea173e9f3a6a58619fa08980e49b59/eni948by-best-epoch=142-val}"

cd "$REPO" || exit 1
export PYTHONPATH="$REPO:$PYTHONPATH"
source ~/miniconda3/bin/activate
conda activate torchcell

echo "host $(hostname)  date $(date)  repo $REPO  commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
CKPT_FILE=$(find "$CKPT" -name '*.ckpt' | head -1)
echo "checkpoint: $CKPT_FILE"
[ -n "$CKPT_FILE" ] || { echo "no .ckpt under $CKPT"; exit 1; }

python experiments/006-kuzmin-tmi/scripts/dcell_training_gpu_profile.py --checkpoint "$CKPT_FILE"
echo "exit $?  date $(date)"
