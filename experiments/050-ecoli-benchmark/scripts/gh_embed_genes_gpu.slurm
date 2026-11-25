#!/bin/bash
#SBATCH --job-name=050-ecoli-embed-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=experiments/050-ecoli-benchmark/slurm/output/%j_%x.out
#SBATCH --error=experiments/050-ecoli-benchmark/slurm/output/%j_%x.out
#SBATCH --no-requeue

# Embed the protein-coding genes of the two E. coli genomes with the models named in
# MODELS, one after the other, on one GPU. Minutes of work; the limit is 20 minutes.
#
#   sbatch --export=ALL,MODELS="esm2 prot_t5",TORCHCELL_SRC=<checkout> \
#       experiments/050-ecoli-benchmark/scripts/gh_embed_genes_gpu.slurm

echo "Running on $(hostname), job $SLURM_JOB_ID, models $MODELS"
TORCHCELL_SRC="${TORCHCELL_SRC:-/home/michaelvolk/Documents/projects/torchcell}"
cd "$TORCHCELL_SRC" || exit 1
echo "checkout: $TORCHCELL_SRC ($(git rev-parse --short HEAD))"
export PYTHONPATH="$TORCHCELL_SRC"
export HF_HUB_OFFLINE=1

for model in $MODELS; do
    ~/miniconda3/envs/torchcell/bin/python experiments/050-ecoli-benchmark/scripts/embed_genes.py --model "$model" || exit 1
done
