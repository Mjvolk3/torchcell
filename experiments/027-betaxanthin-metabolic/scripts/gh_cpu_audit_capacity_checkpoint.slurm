#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_cpu_audit_capacity_checkpoint.slurm
# [[experiments.026-metabolism-flux.kcat-uncertainty-2026-09-14]]
#
# E2. Take the enzyme capacity bound apart at a trained checkpoint, so the 1.5e-7 median box
# width that the wave-8 sampling note quotes is either reproduced and attributed or shown to
# be something else. One forward pass on a fixed batch, the genotype chain recomputed, the
# use_enzyme_capacity=False control, and the promiscuity overcount per gene.
#
# CPU ONLY, NICED BEHIND THE DATABASE REBUILD: no GPU, 4 cores, 24G, --nice=1000.
#
# PYTHONHASHSEED=0 pins the resolver's tie-breaking, so the reference k_cat table this audit
# compares the checkpoint's own buffer against is the same table on every run.
#SBATCH -p main
#SBATCH --nice=1000
#SBATCH --mem=16g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=027-capacity-audit
#SBATCH --time=03:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export PYTHONHASHSEED=0
export NUM_WORKERS=0
export CUDA_VISIBLE_DEVICES=""
echo "host $(hostname); DATA_ROOT=$DATA_ROOT; cpus $SLURM_CPUS_PER_TASK"

python experiments/027-betaxanthin-metabolic/scripts/audit_capacity_checkpoint.py \
    --out capacity_checkpoint_audit.json
