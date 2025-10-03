#!/bin/bash
#SBATCH --job-name=cobra_fba_growth
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/cobra_fba_growth_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/cobra_fba_growth_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --mem=500G

# Change to the torchcell directory
cd /home/michaelvolk/Documents/projects/torchcell || exit 1

# Add the project root to Python path
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

# Log system info
echo "Starting COBRA FBA growth analysis"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo ""

# Run the analysis for ALL genes
# Warning: This will compute ~1.56 billion triple deletions!
# WE CONLCUDED THIS IS INFEASIBLE ON GH WOULD TAKE 100 DAYS!!
#python /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/cobra-fba-growth.py \
#   --n-genes all \
#   --batch-size 18000

# Alternative options:
# For testing with 20 genes (default):
python /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/cobra-fba-growth.py

# For a specific number of genes:
# python /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/cobra-fba-growth.py --n-genes 100

# From a gene list file:
# python /home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/scripts/cobra-fba-growth.py --gene-list my_genes.txt

echo ""
echo "Analysis complete"
echo "End time: $(date)"