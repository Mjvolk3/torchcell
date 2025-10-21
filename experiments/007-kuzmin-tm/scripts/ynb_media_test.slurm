#!/bin/bash
#SBATCH --job-name=media_comparison
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/slurm/output/media_comparison_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/slurm/output/media_comparison_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --mem=50G

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
echo "Media Comparison Test - Addressing Vikas's Discrete Bands Issue"
echo "======================================================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo ""
echo "Testing three media conditions (Suthers et al., 2020 approach):"
echo "  1. Minimal: Glucose + NH4 + O2 + inorganics only"
echo "  2. YNB: Minimal + vitamins/cofactors (0.5 mmol/gDW/h)"
echo "  3. YPD: YNB + 20 amino acids (0.5 mmol/gDW/h each)"
echo ""
echo "Goal: Check if discrete fitness bands (0, 0.15, 0.4, 1.0) persist"
echo "      across media conditions or shift with nutrient availability"
echo ""

RESULTS_DIR="/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/results/cobra-fba-growth"

# Step 1: Extract perturbations if needed
echo "=== Step 1: Preparing Test Dataset ==="
echo "Start time: $(date)"

PERTURBATIONS_FILE="$RESULTS_DIR/unique_perturbations.json"
if [ ! -f "$PERTURBATIONS_FILE" ]; then
    echo "Extracting perturbations from Neo4j..."
    python /home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/scripts/extract_perturbations.py
    if [ $? -ne 0 ]; then
        echo "Error: Failed to extract perturbations"
        exit 1
    fi
else
    echo "Perturbations file exists, using cached version"
fi

echo ""

# Step 2: Test different media conditions
echo "=== Step 2: Testing Three Media Conditions ==="
echo "Using subset of data for quick comparison (--test flag)"
echo ""

# Test 1: Minimal media
echo "--- Testing MINIMAL media ---"
echo "Start time: $(date)"
python /home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/scripts/targeted_fba_with_media_yeastgem.py --media minimal --test

if [ $? -ne 0 ]; then
    echo "Error: Minimal media test failed"
    exit 1
fi

# Test 2: YNB media
echo ""
echo "--- Testing YNB media ---"
echo "Start time: $(date)"
python /home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/scripts/targeted_fba_with_media_yeastgem.py --media YNB --test

if [ $? -ne 0 ]; then
    echo "Error: YNB media test failed"
    exit 1
fi

# Test 3: YPD media
echo ""
echo "--- Testing YPD media ---"
echo "Start time: $(date)"
python /home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/scripts/targeted_fba_with_media_yeastgem.py --media YPD --test

if [ $? -ne 0 ]; then
    echo "Error: YPD media test failed"
    exit 1
fi

echo ""

# Step 3: Compare results across media
echo "=== Step 3: Comparing Fitness Distributions Across Media ==="
echo "End time: $(date)"
echo ""

# Compare the three conditions
if [ -f "$RESULTS_DIR/singles_deletions_minimal.parquet" ] && \
   [ -f "$RESULTS_DIR/singles_deletions_YNB.parquet" ] && \
   [ -f "$RESULTS_DIR/singles_deletions_YPD.parquet" ]; then

    echo "Analyzing fitness distributions..."
    python -c "
import pandas as pd
import numpy as np

minimal = pd.read_parquet('$RESULTS_DIR/singles_deletions_minimal.parquet')
ynb = pd.read_parquet('$RESULTS_DIR/singles_deletions_YNB.parquet')
ypd = pd.read_parquet('$RESULTS_DIR/singles_deletions_YPD.parquet')

# Count unique fitness values
min_unique = len(np.unique(np.round(minimal['fitness'], 3)))
ynb_unique = len(np.unique(np.round(ynb['fitness'], 3)))
ypd_unique = len(np.unique(np.round(ypd['fitness'], 3)))

print('Unique fitness values (more = better):')
print(f'  Minimal: {min_unique}/{len(minimal)} ({min_unique/len(minimal)*100:.1f}%)')
print(f'  YNB:     {ynb_unique}/{len(ynb)} ({ynb_unique/len(ynb)*100:.1f}%)')
print(f'  YPD:     {ypd_unique}/{len(ypd)} ({ypd_unique/len(ypd)*100:.1f}%)')

# Find major bands for each condition
print('')
print('Major fitness bands (>5% frequency):')

for name, df in [('Minimal', minimal), ('YNB', ynb), ('YPD', ypd)]:
    bands = df['fitness'].round(2).value_counts()
    major = bands[bands > len(df) * 0.05].index.tolist()
    print(f'  {name}: {sorted(major)}')

# Check if bands shift
min_bands = minimal['fitness'].round(2).value_counts()
ypd_bands = ypd['fitness'].round(2).value_counts()

min_major = min_bands[min_bands > len(minimal) * 0.05].index.tolist()
ypd_major = ypd_bands[ypd_bands > len(ypd) * 0.05].index.tolist()

if set(min_major) != set(ypd_major):
    print('')
    print('⚠ BANDS SHIFT between minimal and YPD - suggests MEDIA-DRIVEN clustering')
    print('  → Recommendation: Use richer media (YPD) to reduce discrete bands')
else:
    print('')
    print('✓ BANDS CONSISTENT across all media - suggests MODEL-INTRINSIC clustering')
    print('  → Recommendation: Review biomass composition and model constraints')
"
fi

echo ""
echo "======================================================================"
echo "Media Comparison Test Complete!"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Output files:"
echo "  - singles_deletions_minimal.parquet"
echo "  - singles_deletions_YNB.parquet"
echo "  - singles_deletions_YPD.parquet"
echo ""
echo "INTERPRETATION (based on Vikas's recommendations):"
echo ""
echo "If fitness bands SHIFT between media conditions:"
echo "  → Issue is MEDIA-DRIVEN (constraint-based)"
echo "  → Solution: Use YPD media for main analysis"
echo ""
echo "If fitness bands PERSIST across all media:"
echo "  → Issue is MODEL-INTRINSIC"
echo "  → Check: 1) GPR logic, 2) Biomass composition, 3) Bottleneck reactions"
echo ""
echo "To run full analysis with chosen media:"
echo "  python targeted_fba_with_media_yeastgem.py --media YPD  # (no --test flag)"
echo "======================================================================"