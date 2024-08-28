#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16  # Increase the number of tasks per node to the desired number of agents
#SBATCH --cpus-per-task=8  # Adjust the number of CPUs per task based on your requirements
#SBATCH --partition=cpu
#SBATCH --account=bbub-delta-cpu
#SBATCH --job-name=SVR_1e05
#SBATCH --time=48:00:00
#SBATCH --constraint="projects"
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type="END"
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/002-dmi-tmi/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/002-dmi-tmi/slurm/output/%x_%j.out

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

SWEEP_ID=kkpe1uet
PROJECT_NAME=torchcell_002-dmi-tmi_trad-ml_svr_1e05

echo "-----------------"
echo "SWEEP_ID: $SWEEP_ID"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "PROJECT_NAME: $PROJECT_NAME"
echo "-----------------"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "-----------------"

mkdir /projects/bbub/mjvolk3/torchcell/experiments/002-dmi-tmi/agent_log/$SLURM_JOB_ID

for ((i=0; i<$SLURM_NTASKS; i++)); do
    (wandb agent zhao-group/$PROJECT_NAME/$SWEEP_ID > /projects/bbub/mjvolk3/torchcell/experiments/002-dmi-tmi/agent_log/$SLURM_JOB_ID/agent-$PROJECT_NAME-$SWEEP_ID-$i.log 2>&1) &
done

wait