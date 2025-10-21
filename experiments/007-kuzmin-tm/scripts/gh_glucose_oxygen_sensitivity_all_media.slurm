#!/bin/bash
#SBATCH --job-name=glucose_oxygen_sensitivity
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/slurm/output/glucose_oxygen_sensitivity_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/slurm/output/glucose_oxygen_sensitivity_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=365-00:00:00
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
echo "Glucose/O2 Sensitivity Analysis - Vikas Upadhyay's Recommendation"
echo "======================================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo ""
echo "Testing 48 conditions:"
echo "  - 3 media types: minimal, YNB, YPD"
echo "  - 4 glucose levels: 2, 5, 10, 20 mmol/gDW/h"
echo "  - 4 O2 levels: unlimited (1000), 20, 10, 5 mmol/gDW/h"
echo ""
echo "Goal: Determine if discrete fitness bands are:"
echo "  1. CONSTRAINT-DRIVEN (bands shift with glucose/O2 changes)"
echo "  2. MODEL-INTRINSIC (bands persist across all conditions)"
echo ""

# Step 1: Run glucose/oxygen sweep (data generation)
echo "=== Step 1: Running Glucose/O2 Sweep ==="
echo "Start time: $(date)"
python experiments/007-kuzmin-tm/scripts/run_glucose_oxygen_sweep.py

if [ $? -ne 0 ]; then
    echo "Error: Glucose/O2 sweep failed"
    exit 1
fi

echo ""

# Step 2: Post-process and match to experimental data
echo "=== Step 2: Post-Processing and Matching ==="
echo "Start time: $(date)"
python experiments/007-kuzmin-tm/scripts/postprocess_glucose_oxygen_sweep.py

if [ $? -ne 0 ]; then
    echo "Error: Post-processing failed"
    exit 1
fi

echo ""

# Step 3: Generate plots
echo "=== Step 3: Generating Plots ==="
echo "Start time: $(date)"
python experiments/007-kuzmin-tm/scripts/plot_glucose_oxygen_sweep.py

if [ $? -ne 0 ]; then
    echo "Error: Plotting failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Sensitivity Analysis Complete!"
echo "======================================================================"
echo ""
echo "KEY OUTPUTS:"
echo "  - FBA results: *_deletions_*_glc*_o2*.parquet"
echo "  - Matched data: *_glc*_o2*_matched.parquet"
echo "  - Plots: ASSET_IMAGES_DIR/experiments.007.glucose_oxygen_*.png"
echo ""
echo "INTERPRETATION GUIDE:"
echo ""
echo "IF correlations improve with richer media or higher glucose/O2:"
echo "  → Issue is CONSTRAINT-DRIVEN (media composition matters)"
echo "  → Recommendation: Use nutrient-rich conditions (YPD, high glucose)"
echo ""
echo "IF correlations remain poor across all conditions:"
echo "  → Issue is MODEL-INTRINSIC"
echo "  → Check: GPR logic, biomass composition, reaction bottlenecks"
echo ""
echo "IF discrete bands shift with glucose/O2 changes:"
echo "  → Bands are CONSTRAINT-DRIVEN"
echo "  → Adjust uptake bounds to improve resolution"
echo ""
echo "IF discrete bands persist across all conditions:"
echo "  → Bands are MODEL-INTRINSIC"
echo "  → Review model structure and reaction stoichiometry"
echo ""
echo "======================================================================"
echo "Results directory: experiments/007-kuzmin-tm/results/cobra-fba-growth"
echo "Plots saved in: ASSET_IMAGES_DIR (from .env)"
echo "======================================================================"
