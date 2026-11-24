#!/bin/bash
#SBATCH --job-name=050-ecoli-probes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/050-ecoli-benchmark/slurm/output/%j_%x.out
#SBATCH --error=experiments/050-ecoli-benchmark/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Fit the linear probes of one benchmark task on CPU.
#
#   sbatch --export=ALL,TASK=ecoli-growth-rate-campos2018,FEATURES='random one_hot',TORCHCELL_SRC=<checkout> \
#       experiments/050-ecoli-benchmark/scripts/gh_linear_probes.slurm

echo "Running on $(hostname), job $SLURM_JOB_ID, task $TASK"
TORCHCELL_SRC="${TORCHCELL_SRC:-/home/michaelvolk/Documents/projects/torchcell}"
cd "$TORCHCELL_SRC" || exit 1
echo "checkout: $TORCHCELL_SRC ($(git rev-parse --short HEAD))"
export PYTHONPATH="$TORCHCELL_SRC"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

~/miniconda3/envs/torchcell/bin/python experiments/050-ecoli-benchmark/scripts/linear_probes.py --task "$TASK" --features $FEATURES
