#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu
#SBATCH --account=bbub-delta-cpu
#SBATCH--job-name=traditional_ml_dataset_sweep_1e05-normalized_chrom_pathways
#SBATCH --time=48:00:00
#SBATCH --constraint="projects"
#SBATCH --mail-user=mjvolk3@illinois.edus
#SBATCH --mail-type="END"
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/slurm/output/%x_%j.out


module reset
source ~/.bashrc
cd /projects/bbub/mjvolk3/torchcell
pwd
lscpu
cat /proc/meminfo
module list
conda activate /projects/bbub/miniconda3/envs/torchcell

echo "job is starting on `hostname`"
wandb artifact cache cleanup 1GB
####

SWEEP_FILE=experiments/smf-dmf-tmf-001/conf/traditional_ml_dataset_sweep_1e05-normalized_chrom_pathways.yaml
SWEEP_ID=$(wandb sweep $SWEEP_FILE 2>&1 | grep "Creating sweep with ID:" | awk -F': ' '{print $3}')
PROJECT_NAME=$(python torchcell/config.py $SWEEP_FILE)

wandb agent zhao-group/$PROJECT_NAME/$SWEEP_ID