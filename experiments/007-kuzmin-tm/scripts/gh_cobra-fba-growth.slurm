#!/bin/bash
#SBATCH --job-name=targeted_fba_complete
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/slurm/output/targeted_fba_complete_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/slurm/output/targeted_fba_complete_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --mem=500G

# Change to the torchcell directory
cd /home/michaelvolk/Documents/projects/torchcell || exit 1

# Add the project root to Python path
export PYTHONPATH="/home/michaelvolk/Documents/projects/torchcell:$PYTHONPATH"

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

# Create output directories
mkdir -p /home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/slurm/output
mkdir -p /home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/results/cobra-fba-growth

# Log system info
echo "======================================================================"
echo "Starting Complete COBRA FBA Pipeline for Kuzmin Trigenic Dataset"
echo "======================================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo ""

# Step 0: Run verification checks as recommended by Vikas Upadhyay
echo "=== Step 0: Running Verification Checks ==="
echo "Start time: $(date)"
RESULTS_DIR="/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/results/cobra-fba-growth"

# Run GPR knockout logic verification
echo "\n--- Verifying GPR Knockout Logic (AND/OR rules) ---"
python /home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/scripts/verify_gpr_knockout_logic.py
if [ $? -ne 0 ]; then
    echo "Warning: GPR verification encountered issues"
fi

# Run biomass and maintenance verification
echo "\n--- Verifying Biomass and Maintenance Parameters ---"
python /home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/scripts/verify_biomass_maintenance.py
if [ $? -ne 0 ]; then
    echo "Warning: Biomass/maintenance verification encountered issues"
fi

# Run glucose/oxygen sensitivity analysis
echo "\n--- Running Glucose/Oxygen Sensitivity Analysis ---"
echo "This checks if discrete growth bands are due to media constraints"
python /home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/scripts/glucose_oxygen_sensitivity.py
if [ $? -ne 0 ]; then
    echo "Warning: Sensitivity analysis encountered issues"
fi

echo "Verification checks complete. Results saved in $RESULTS_DIR"
echo ""

# Step 0.5: Clean up old results for fresh start
echo "=== Step 0.5: Cleaning Up Old Results ==="
echo "Start time: $(date)"

# Backup old FBA results if they exist (but keep verification results)
if [ -d "$RESULTS_DIR" ] && [ "$(ls -A $RESULTS_DIR)" ]; then
    # Move verification results temporarily
    TEMP_VERIFY="/tmp/verify_results_$$"
    mkdir -p "$TEMP_VERIFY"
    mv "$RESULTS_DIR"/gpr_verification_results.json "$TEMP_VERIFY"/ 2>/dev/null
    mv "$RESULTS_DIR"/biomass_maintenance_verification.json "$TEMP_VERIFY"/ 2>/dev/null
    mv "$RESULTS_DIR"/glucose_oxygen_sensitivity_results.parquet "$TEMP_VERIFY"/ 2>/dev/null
    mv "$RESULTS_DIR"/sensitivity_analysis_summary.json "$TEMP_VERIFY"/ 2>/dev/null
    mv "$RESULTS_DIR"/glucose_oxygen_sensitivity_*.png "$TEMP_VERIFY"/ 2>/dev/null
    mv "$RESULTS_DIR"/fitness_bands_heatmap_*.png "$TEMP_VERIFY"/ 2>/dev/null

    # Backup remaining old results
    if [ "$(ls -A $RESULTS_DIR)" ]; then
        BACKUP_DIR="${RESULTS_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
        echo "Backing up existing FBA results to: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
        mv "$RESULTS_DIR"/* "$BACKUP_DIR"/ 2>/dev/null
    fi

    # Restore verification results
    mv "$TEMP_VERIFY"/* "$RESULTS_DIR"/ 2>/dev/null
    rm -rf "$TEMP_VERIFY"
fi

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"
echo "Results directory ready"
echo ""

# Step 1: Extract unique perturbations from the dataset
echo "=== Step 1: Extracting Perturbations from Neo4j Dataset ==="
echo "Start time: $(date)"
python /home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/scripts/extract_perturbations.py

if [ $? -ne 0 ]; then
    echo "Error: extract_perturbations.py failed"
    exit 1
fi

echo "Perturbation extraction complete"
echo "Files created:"
ls -lh "$RESULTS_DIR"/perturbations_*.json 2>/dev/null || echo "  No perturbation files found"
echo ""

# Step 2: Run targeted FBA analysis with 60s solver timeout
echo "=== Step 2: Running Targeted FBA Analysis ==="
echo "Configuration:"
echo "  - Solver timeout: 60 seconds"
echo "  - CPUs: $SLURM_CPUS_PER_TASK"
echo "  - Method: Fast multiprocessing with pickle serialization"
echo "Start time: $(date)"
python /home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/scripts/targeted_fba_growth_fast.py

if [ $? -ne 0 ]; then
    echo "Error: targeted_fba_growth_fast.py failed"
    exit 1
fi

echo "FBA analysis complete"
echo "Files created:"
ls -lh "$RESULTS_DIR"/*.parquet 2>/dev/null | grep -E "(singles|doubles|triples|digenic|trigenic)" || echo "  No FBA result files found"
echo ""

# Step 3: Match FBA results to experimental data using fixed script
echo "=== Step 3: Matching FBA to Experimental Data (Using Fixed Script) ==="
echo "Start time: $(date)"
echo "Using dataset.label_df for correct phenotype values"
python /home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/scripts/match_fba_to_experiments.py

if [ $? -ne 0 ]; then
    echo "Error: match_fba_to_experiments.py failed"
    exit 1
fi

echo "Matching complete"
echo "Files created:"
ls -lh "$RESULTS_DIR"/matched_fba_experimental_fixed.* 2>/dev/null || echo "  No matched files found"
echo ""

# Step 4: Generate comparison plots
echo "=== Step 4: Generating Comparison Plots ==="
echo "Start time: $(date)"
python /home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/scripts/plot_fba_comparison.py

if [ $? -ne 0 ]; then
    echo "Error: plot_fba_comparison.py failed"
    exit 1
fi

echo "Plotting complete"
echo "Plot files created:"
ls -lh "$RESULTS_DIR"/fba_comparison*.png 2>/dev/null || echo "  No plot files found"
echo ""

# Step 5: Summary statistics
echo "=== Step 5: Final Summary ==="
echo "End time: $(date)"
echo ""

# Count records in key files
echo "Record counts:"
if [ -f "$RESULTS_DIR/singles_deletions.parquet" ]; then
    echo "  Singles: $(python -c "import pandas as pd; print(len(pd.read_parquet('$RESULTS_DIR/singles_deletions.parquet')))" 2>/dev/null || echo "Unable to count")"
fi
if [ -f "$RESULTS_DIR/doubles_deletions.parquet" ]; then
    echo "  Doubles: $(python -c "import pandas as pd; print(len(pd.read_parquet('$RESULTS_DIR/doubles_deletions.parquet')))" 2>/dev/null || echo "Unable to count")"
fi
if [ -f "$RESULTS_DIR/triples_deletions.parquet" ]; then
    echo "  Triples: $(python -c "import pandas as pd; print(len(pd.read_parquet('$RESULTS_DIR/triples_deletions.parquet')))" 2>/dev/null || echo "Unable to count")"
fi
if [ -f "$RESULTS_DIR/matched_fba_experimental_fixed.parquet" ]; then
    echo "  Matched records: $(python -c "import pandas as pd; print(len(pd.read_parquet('$RESULTS_DIR/matched_fba_experimental_fixed.parquet')))" 2>/dev/null || echo "Unable to count")"
fi

echo ""
echo "All output files:"
ls -lh "$RESULTS_DIR"

echo ""
echo "======================================================================"
echo "Pipeline Complete!"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "VERIFICATION OUTPUTS (Vikas's recommendations):"
echo "  - GPR logic check: gpr_verification_results.json"
echo "  - Biomass/maintenance: biomass_maintenance_verification.json"
echo "  - Sensitivity analysis: sensitivity_analysis_summary.json"
echo "  - Sensitivity plots: glucose_oxygen_sensitivity_*.png"
echo ""
echo "FBA ANALYSIS OUTPUTS:"
echo "  - FBA predictions: singles/doubles/triples_deletions.parquet"
echo "  - Genetic interactions: digenic/trigenic_interactions.parquet"
echo "  - Matched data: matched_fba_experimental_fixed.parquet"
echo "  - Visualization: fba_comparison_latest.png"
echo ""
echo "Review verification outputs to check:"
echo "  1. GPR knockout logic (complexes vs isoenzymes)"
echo "  2. Biomass/GAM/NGAM parameters match Yeast9 standards"
echo "  3. Whether discrete growth bands shift with media conditions"
echo "======================================================================"