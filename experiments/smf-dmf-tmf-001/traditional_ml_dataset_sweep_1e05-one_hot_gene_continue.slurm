#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu
#SBATCH --account=bbub-delta-cpu
#SBATCH--job-name=traditional_ml_dataset_sweep_1e05-one_hot
#SBATCH --time=48:00:00
#SBATCH --constraint="projects"
#SBATCH --mail-user=mjvolk3@illinois.edu
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

SWEEP_ID=bjncrm5x
PROJECT_NAME=torchcell_smf-dmf-tmf-traditional_ml_dataset

####

mkdir /scratch/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/agent_log/$SLURM_JOB_ID

for ((i=0; i<$SLURM_NTASKS; i++)); do
    (wandb agent zhao-group/$PROJECT_NAME/$SWEEP_ID > /scratch/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/agent_log/$SLURM_JOB_ID/agent-$PROJECT_NAME-$SWEEP_ID-$i.log 2>&1) &
done

wait