#!/bin/bash
#SBATCH -p mmli
#SBATCH --mem=480g
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=opt_DCell
#SBATCH --time=300-00:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/005-kuzmin2018-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/005-kuzmin2018-tmi/slurm/output/%x_%j.out

module purge
source ~/.bashrc
module load singularity

cd /home/a-m/mjvolk3/projects/torchcell

# Set environment variables needed for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Launch with srun to ensure proper process distribution
srun --ntasks-per-node=4 --gpus-per-task=1 singularity exec --nv rockylinux_9.sif bash -c "
source $HOME/miniconda3/bin/activate
conda activate torchcell
python /home/a-m/mjvolk3/scratch/torchcell/experiments/005-kuzmin2018-tmi/scripts/dcell.py --config-name dcell_kuzmin2018_tmi
"

#/home/a-m/mjvolk3/projects/torchcell/experiments/005-kuzmin2018-tmi/conf/dcell_kuzmin2018_tmi_maxing_resources.yaml
# experiments/005-kuzmin2018-tmi/conf/dcell_kuzmin2018_tmi_cpu.yaml


