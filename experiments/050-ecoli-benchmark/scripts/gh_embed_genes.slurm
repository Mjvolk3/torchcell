#!/bin/bash
#SBATCH --job-name=050-ecoli-embed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/050-ecoli-benchmark/slurm/output/%j_%x.out
#SBATCH --error=experiments/050-ecoli-benchmark/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Embed the protein-coding genes of the two E. coli genomes with one sequence model, on
# CPU (the four GPUs were allocated when this was written; add --gres=gpu:1 to use one).
#
#   TORCHCELL_SRC=<checkout> sbatch --export=ALL,MODEL=esm2 \
#       experiments/050-ecoli-benchmark/scripts/gh_embed_genes.slurm

echo "Running on $(hostname), job $SLURM_JOB_ID, model $MODEL"
TORCHCELL_SRC="${TORCHCELL_SRC:-/home/michaelvolk/Documents/projects/torchcell}"
cd "$TORCHCELL_SRC" || exit 1
echo "checkout: $TORCHCELL_SRC ($(git rev-parse --short HEAD))"
export PYTHONPATH="$TORCHCELL_SRC"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1

~/miniconda3/envs/torchcell/bin/python experiments/050-ecoli-benchmark/scripts/embed_genes.py --model "$MODEL"
