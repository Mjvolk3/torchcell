#!/bin/bash
#SBATCH --job-name=007-fba-screen-medium
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/slurm/output/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=160G
#SBATCH --time=08:00:00

# The Yeast9 FBA baseline rerun on the trigenic screens' selection medium, with two
# controls (SM at glucose 3.3; the model default that reproduces the frozen run).
# CPU only: one GLPK linear program per deletion set, 987,530 per arm, in a process pool.
# REPO is the checkout whose code runs (a worktree while the change is unlanded).

REPO="${REPO:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/fba-screen-medium}"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:$PYTHONPATH"
source ~/miniconda3/bin/activate
conda activate torchcell
mkdir -p /home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/slurm/output

echo "host $(hostname)  date $(date)  repo $REPO  commit $(git rev-parse --short HEAD)  cpus $SLURM_CPUS_PER_TASK"
python experiments/007-kuzmin-tm/scripts/fba_screen_medium.py
echo "exit $?  date $(date)"
