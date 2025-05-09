#!/bin/bash
#SBATCH -p cabbi
#SBATCH --mem=300g
#SBATCH -N 1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:3
#SBATCH --job-name=HCBD_DDP
#SBATCH --time=300-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/003-fit-int/slurm/output/%x_%j.out

module purge
source ~/.bashrc
module load singularity

cd /home/a-m/mjvolk3/projects/torchcell

# Set environment variables needed for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)


# Launch with srun to ensure proper process distribution
srun --ntasks-per-node=3 --gpus-per-task=1 singularity exec --nv rockylinux_9.sif bash -c "
source $HOME/miniconda3/bin/activate
conda activate torchcell
python /home/a-m/mjvolk3/scratch/torchcell/experiments/003-fit-int/scripts/hetero_cell_bipartite_dango.py
"