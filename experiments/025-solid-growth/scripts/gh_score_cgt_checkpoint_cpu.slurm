#!/bin/bash
# experiments/025-solid-growth/scripts/gh_score_cgt_checkpoint_cpu.slurm
# [[experiments.025-solid-growth.scripts.score_cgt_checkpoint_cpu]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/gh_score_cgt_checkpoint_cpu.slurm
#
# CPU-only scoring of job 1640's epoch 7 checkpoint (W&B 327csnlk, cgt_s0_q_kl_004, arm Q
# query-pair-disjoint) on the arm Q validation and test parts. No GPU is requested: the
# encoder runs once and the per-record work is the perturbation transform plus readout,
# about 8 s per 1,000 records on 64 threads, so both parts finish in about 15 minutes.
#
#   sbatch experiments/025-solid-growth/scripts/gh_score_cgt_checkpoint_cpu.slurm
#
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=025-score-1640-cpu
#SBATCH --time=03:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/025-solid-growth/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/025-solid-growth/slurm/output/%x_%j.out

set -euo pipefail

cd /home/michaelvolk/Documents/projects/torchcell || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell

echo "job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK cpus, $(date)"

CKPT_ROOT=/scratch/projects/torchcell-scratch/models/checkpoints
CKPT="$CKPT_ROOT/gilahyper-1640_254c63640a44d1cf47eacc3e8ac2b684899d400f9c834e4b2bddf687afd7ca0a/327csnlk-best-pearson-epoch=07-val/gene_interaction/Pearson=0.1993.ckpt"
[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }

# No GPU on this job; make torch see none even on a GPU node.
export CUDA_VISIBLE_DEVICES=""

python experiments/025-solid-growth/scripts/score_cgt_checkpoint_cpu.py \
  --checkpoint "$CKPT" \
  --run-id 327csnlk \
  --arm Q \
  --parts val,test \
  --pooling sum \
  --batch-size 256

echo "done $(date)"
