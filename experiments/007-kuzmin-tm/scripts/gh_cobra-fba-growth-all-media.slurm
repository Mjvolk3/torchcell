#!/bin/bash
#SBATCH --job-name=fba_all_media
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/slurm/output/fba_all_media_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/slurm/output/fba_all_media_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --mem=500G

# Change to the torchcell directory
cd /home/michaelvolk/Documents/projects/torchcell || exit 1

# Add the project root to Python path
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

# Create output directories
mkdir -p experiments/007-kuzmin-tm/slurm/output
mkdir -p experiments/007-kuzmin-tm/results/cobra-fba-growth

# Log system info
echo "======================================================================"
echo "COBRA FBA Analysis with Multiple Media Conditions"
echo "======================================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo ""
echo "Media conditions to test:"
echo "  1. Minimal: Glucose + NH4 + O2 + inorganics"
echo "  2. YNB: Minimal + vitamins/cofactors (Suthers et al. 2020)"
echo "  3. YPD: YNB + 20 amino acids (Suthers et al. 2020)"
echo ""

# Step 1: Run FBA data generation for all media
echo "=== Step 1: Running FBA Data Generation ==="
echo "Start time: $(date)"
python experiments/007-kuzmin-tm/scripts/run_fba_all_media.py

if [ $? -ne 0 ]; then
    echo "Error: FBA data generation failed"
    exit 1
fi

echo ""

# Step 2: Post-process and match to experimental data
echo "=== Step 2: Post-Processing and Matching ==="
echo "Start time: $(date)"
python experiments/007-kuzmin-tm/scripts/postprocess_fba_all_media.py

if [ $? -ne 0 ]; then
    echo "Error: Post-processing failed"
    exit 1
fi

echo ""

# Step 3: Generate plots
echo "=== Step 3: Generating Plots ==="
echo "Start time: $(date)"
python experiments/007-kuzmin-tm/scripts/plot_fba_all_media.py

if [ $? -ne 0 ]; then
    echo "Error: Plotting failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Pipeline Complete!"
echo "Results directory: experiments/007-kuzmin-tm/results/cobra-fba-growth"
echo ""
echo "KEY OUTPUTS:"
echo "  - FBA results per media: *_deletions_{minimal,YNB,YPD}.parquet"
echo "  - Matched data: matched_fba_experimental_{minimal,YNB,YPD}.parquet"
echo "  - Comparison plots: ASSET_IMAGES_DIR/experiments.007.fba_comparison_all_media_*.png"
echo ""
echo "INTERPRETATION:"
echo "  - Check if correlation improves with richer media"
echo "  - Look for shifts in fitness distributions"
echo "  - Compare number of affected mutants across conditions"
echo "======================================================================"