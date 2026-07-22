#!/bin/bash
#SBATCH -p mmli
#SBATCH --mem=488g
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=000-019-MTCGT
#SBATCH --time=300-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%j.out

module purge
source ~/.bashrc
module load singularity

cd /home/a-m/mjvolk3/projects/torchcell || exit 1

# DDP rendezvous (srun launches one task per GPU; Lightning reads SLURM_* vars)
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 2))

srun singularity exec --nv /home/a-m/mjvolk3/projects/torchcell/rockylinux_9.sif bash -c "
source \$HOME/miniconda3/bin/activate
conda activate torchcell

export PYTHONPATH=\"/home/a-m/mjvolk3/projects/torchcell:\$PYTHONPATH\"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export PYTHONUNBUFFERED=1

python /home/a-m/mjvolk3/projects/torchcell/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py \
  --config-name igb_mmli_train_cgt_multitask_000 \
  data_module.num_workers=\$NUM_WORKERS
"
