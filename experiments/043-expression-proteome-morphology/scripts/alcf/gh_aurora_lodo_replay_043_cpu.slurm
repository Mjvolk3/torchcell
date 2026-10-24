#!/bin/bash
#SBATCH -p main
#SBATCH --mem=96g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=043-lodo-replay-cpu
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
# experiments/043-expression-proteome-morphology/scripts/alcf/gh_aurora_lodo_replay_043_cpu.slurm
# [[experiments.043-expression-proteome-morphology.scripts.alcf.gh_aurora_lodo_replay_043_cpu]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/gh_aurora_lodo_replay_043_cpu.slurm
#
# CPU only (no GPU). Bounds audit of every epoch-1 micro-batch and CPU forward/backward of
# the window an Aurora lodo cell faulted in (aurora_lodo_replay_043.py):
#   sbatch --export=ALL,LODO=hu,LO=11780,HI=12010 gh_aurora_lodo_replay_043_cpu.slurm
#   sbatch --export=ALL,LODO=zelezniak,LO=9580,HI=9810 gh_aurora_lodo_replay_043_cpu.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD" DATA_ROOT=/scratch/projects/torchcell-scratch
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
echo "host $(hostname) commit $(git rev-parse --short HEAD) lodo $LODO micro $LO..$HI $(date)"
~/miniconda3/envs/torchcell/bin/python \
  experiments/043-expression-proteome-morphology/scripts/alcf/aurora_lodo_replay_043.py \
  --lo "$LO" --hi "$HI" -- arm=all "arm.exclude_from_training=[$LODO]" \
  data.store_version=v2 data.prep_version=prep_v4 model.conditioning.source=ontology \
  loss=hlgauss weighting=equal seed=0 data.batch_size=2 data.mark_cap=null
echo "done $(date)"
