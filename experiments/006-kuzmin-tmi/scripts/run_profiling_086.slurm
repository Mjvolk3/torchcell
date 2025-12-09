#!/bin/bash
#SBATCH -p main
#SBATCH --mem=250g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=PROFILE-086
#SBATCH --time=4:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

# Change to torchcell root directory
cd /home/michaelvolk/Documents/projects/torchcell || exit 1
TORCHCELL_ROOT="/home/michaelvolk/Documents/projects/torchcell"

# Create timestamped output directory
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
OUTPUT_DIR="experiments/006-kuzmin-tmi/profiling_results/profiling_086_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${TORCHCELL_ROOT}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

echo "================================================================================"
echo "EXPERIMENT 086: SINGLE-GPU PROFILING WITH 10K SUBSET"
echo "================================================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "Tests:"
echo "  086a: DataLoader profiling (data pipeline only, no model)"
echo "  086b: Model profiling (data + model forward/backward, no optimizer)"
echo "================================================================================"

# EXPERIMENT 086a: DataLoader Profiling (Lazy Hetero)
echo ""
echo "================================================================================"
echo "STEP 1/4: Experiment 086a - Lazy Hetero DataLoader Profiling"
echo "================================================================================"
echo "Config: hetero_cell_bipartite_dango_gi_gh_086_dataloader"
echo "Measures: Data loading + graph processor + collation + CPU→GPU"
echo ""

apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
  --env PYTHONUNBUFFERED=1 \
  /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc "
export PYTHONPATH='${TORCHCELL_ROOT}:\$PYTHONPATH'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/bin/activate
conda activate torchcell

python ${TORCHCELL_ROOT}/experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_lazy.py \
  --config-name hetero_cell_bipartite_dango_gi_gh_086_dataloader \
  2>&1 | tee ${OUTPUT_DIR}/086a_lazy_hetero_dataloader.log
" || { echo "ERROR: Lazy Hetero DataLoader profiling failed"; exit 1; }

echo ""
echo "Lazy Hetero DataLoader profiling completed"
echo ""

# EXPERIMENT 086a: DataLoader Profiling (DANGO)
echo "================================================================================"
echo "STEP 2/4: Experiment 086a - DANGO DataLoader Profiling"
echo "================================================================================"
echo "Config: dango_kuzmin2018_tmi_string12_0_086_dataloader"
echo "Measures: Data loading + graph processor + collation + CPU→GPU"
echo ""

apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
  --env PYTHONUNBUFFERED=1 \
  /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc "
export PYTHONPATH='${TORCHCELL_ROOT}:\$PYTHONPATH'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/bin/activate
conda activate torchcell

python ${TORCHCELL_ROOT}/experiments/006-kuzmin-tmi/scripts/dango.py \
  --config-name dango_kuzmin2018_tmi_string12_0_086_dataloader \
  2>&1 | tee ${OUTPUT_DIR}/086a_dango_dataloader.log
" || { echo "ERROR: DANGO DataLoader profiling failed"; exit 1; }

echo ""
echo "DANGO DataLoader profiling completed"
echo ""

# EXPERIMENT 086b: Model Profiling (Lazy Hetero)
echo "================================================================================"
echo "STEP 3/4: Experiment 086b - Lazy Hetero Model Profiling"
echo "================================================================================"
echo "Config: hetero_cell_bipartite_dango_gi_gh_086_model"
echo "Measures: Data + Model forward/backward (no optimizer)"
echo ""

apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
  --env PYTHONUNBUFFERED=1 \
  /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc "
export PYTHONPATH='${TORCHCELL_ROOT}:\$PYTHONPATH'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/bin/activate
conda activate torchcell

python ${TORCHCELL_ROOT}/experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_lazy.py \
  --config-name hetero_cell_bipartite_dango_gi_gh_086_model \
  2>&1 | tee ${OUTPUT_DIR}/086b_lazy_hetero_model.log
" || { echo "ERROR: Lazy Hetero Model profiling failed"; exit 1; }

echo ""
echo "Lazy Hetero Model profiling completed"
echo ""

# EXPERIMENT 086b: Model Profiling (DANGO)
echo "================================================================================"
echo "STEP 4/4: Experiment 086b - DANGO Model Profiling"
echo "================================================================================"
echo "Config: dango_kuzmin2018_tmi_string12_0_086_model"
echo "Measures: Data + Model forward/backward (no optimizer)"
echo ""

apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
  --env PYTHONUNBUFFERED=1 \
  /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc "
export PYTHONPATH='${TORCHCELL_ROOT}:\$PYTHONPATH'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/bin/activate
conda activate torchcell

python ${TORCHCELL_ROOT}/experiments/006-kuzmin-tmi/scripts/dango.py \
  --config-name dango_kuzmin2018_tmi_string12_0_086_model \
  2>&1 | tee ${OUTPUT_DIR}/086b_dango_model.log
" || { echo "ERROR: DANGO Model profiling failed"; exit 1; }

echo ""
echo "DANGO Model profiling completed"
echo ""

# GENERATE COMPARISON REPORT
echo "================================================================================"
echo "STEP 5/4: Generating Comprehensive Comparison Report"
echo "================================================================================"

# Function to extract average it/s from log file
extract_avg_its() {
    local logfile=$1
    grep -oP '\d+\.\d+(?=it/s)' "$logfile" 2>/dev/null | tail -20 | awk '{sum+=$1; count++} END {if(count>0) printf "%.3f", sum/count; else print "N/A"}'
}

# Extract metrics for all 4 experiments
LAZY_086a_ITS=$(extract_avg_its "${OUTPUT_DIR}/086a_lazy_hetero_dataloader.log")
DANGO_086a_ITS=$(extract_avg_its "${OUTPUT_DIR}/086a_dango_dataloader.log")
LAZY_086b_ITS=$(extract_avg_its "${OUTPUT_DIR}/086b_lazy_hetero_model.log")
DANGO_086b_ITS=$(extract_avg_its "${OUTPUT_DIR}/086b_dango_model.log")

# Calculate ratios
if [[ "$LAZY_086a_ITS" != "N/A" && "$DANGO_086a_ITS" != "N/A" ]]; then
    RATIO_086a=$(echo "scale=2; $DANGO_086a_ITS / $LAZY_086a_ITS" | bc)
else
    RATIO_086a="N/A"
fi

if [[ "$LAZY_086b_ITS" != "N/A" && "$DANGO_086b_ITS" != "N/A" ]]; then
    RATIO_086b=$(echo "scale=2; $DANGO_086b_ITS / $LAZY_086b_ITS" | bc)
else
    RATIO_086b="N/A"
fi

# Generate comprehensive comparison report
cat > ${OUTPUT_DIR}/comparison_086_${TIMESTAMP}.txt <<EOF
=============================================================================
EXPERIMENT 086: COMPREHENSIVE PROFILING COMPARISON
Generated: ${TIMESTAMP}
=============================================================================

CONFIGURATION:
- GPU: Single GPU (no DDP)
- Dataset: 10,000 sample subset
- Batch size: Lazy Hetero=28, DANGO=64
- Steps per run: 100

EXPERIMENTS:
  086a: DataLoader profiling (data pipeline only)
  086b: Model profiling (data + model forward/backward)

=============================================================================
EXPERIMENT 086a: DATALOADER PROFILING (Data Pipeline Only)
=============================================================================

Lazy Hetero DataLoader avg it/s: ${LAZY_086a_ITS}
DANGO DataLoader avg it/s:       ${DANGO_086a_ITS}

Speedup ratio (DANGO / Lazy Hetero): ${RATIO_086a}x

WHAT THIS MEASURES:
- DataLoader fetch from Neo4j/LMDB
- Graph processor (LazySubgraphRepresentation vs Perturbation)
- Collation function (LazyCollater vs default)
- CPU→GPU batch transfer
- ❌ NO model forward/backward

INTERPRETATION:
If ratio is >10×: Data pipeline is severely bottlenecked
  → LazySubgraphRepresentation processes 100-200× more data
  → Consider switching to Perturbation processor or preprocessing

Last 20 it/s measurements (Lazy Hetero):
$(grep -oP '\d+\.\d+(?=it/s)' "${OUTPUT_DIR}/086a_lazy_hetero_dataloader.log" 2>/dev/null | tail -20 | nl)

Last 20 it/s measurements (DANGO):
$(grep -oP '\d+\.\d+(?=it/s)' "${OUTPUT_DIR}/086a_dango_dataloader.log" 2>/dev/null | tail -20 | nl)

=============================================================================
EXPERIMENT 086b: MODEL PROFILING (Data + Model Forward/Backward)
=============================================================================

Lazy Hetero Model avg it/s: ${LAZY_086b_ITS}
DANGO Model avg it/s:       ${DANGO_086b_ITS}

Speedup ratio (DANGO / Lazy Hetero): ${RATIO_086b}x

WHAT THIS MEASURES:
- DataLoader + graph processor + collation + CPU→GPU (from 086a)
- ✅ Model forward pass
- ✅ Model backward pass (gradient computation)
- ❌ NO optimizer step

INTERPRETATION:
Compare 086b ratio vs 086a ratio to isolate model compute:
  - If ratios similar: Model compute adds minimal overhead
  - If 086b ratio > 086a ratio: Model is also slower in Lazy Hetero
  - If 086b ratio < 086a ratio: Model actually faster (unlikely)

Last 20 it/s measurements (Lazy Hetero):
$(grep -oP '\d+\.\d+(?=it/s)' "${OUTPUT_DIR}/086b_lazy_hetero_model.log" 2>/dev/null | tail -20 | nl)

Last 20 it/s measurements (DANGO):
$(grep -oP '\d+\.\d+(?=it/s)' "${OUTPUT_DIR}/086b_dango_model.log" 2>/dev/null | tail -20 | nl)

=============================================================================
DECOMPOSITION ANALYSIS
=============================================================================

Estimated per-component overhead (Lazy Hetero vs DANGO):

1. Data Pipeline Overhead (086a):
   DANGO: ${DANGO_086a_ITS} it/s
   Lazy:  ${LAZY_086a_ITS} it/s
   Ratio: ${RATIO_086a}x slower

2. Model Compute Overhead (086b - 086a):
   If 086a measures data only and 086b measures data+model:
   Model overhead ≈ (1/086b_it/s - 1/086a_it/s)

   Lazy Hetero model time: $(echo "scale=3; 1/${LAZY_086b_ITS} - 1/${LAZY_086a_ITS}" | bc 2>/dev/null || echo "N/A") sec/it
   DANGO model time:       $(echo "scale=3; 1/${DANGO_086b_ITS} - 1/${DANGO_086a_ITS}" | bc 2>/dev/null || echo "N/A") sec/it

3. Expected Full Training Overhead:
   Includes optimizer step (AdamW) - not measured here
   Estimate: add ~5-10% to 086b times

=============================================================================
COMPARISON WITH PREVIOUS PROFILING (524/DDP)
=============================================================================

Previous results (4 GPU DDP, full dataset):
  Lazy Hetero: ~1.6 it/s
  DANGO: ~19.9 it/s
  Ratio: ~12.5× slower

Current results (1 GPU, 10K subset):
  086a ratio: ${RATIO_086a}×
  086b ratio: ${RATIO_086b}×

INSIGHTS:
- If 086a ratio matches 524 ratio: Data pipeline is the bottleneck
- If 086b ratio >> 086a ratio: Model is also significantly slower
- Single GPU should be faster than DDP (no communication overhead)

=============================================================================
NEXT STEPS BASED ON RESULTS
=============================================================================

If ${RATIO_086a}x > 10 (high data pipeline overhead):
  → Test Perturbation() processor instead of LazySubgraphRepresentation
  → Use preprocessed dataset with precomputed masks
  → Profile LazyCollater vs default PyG collation

If ${RATIO_086b}x > ${RATIO_086a}x + 2 (high model overhead):
  → Profile model architecture differences
  → Check for dual forward passes in Lazy Hetero
  → Consider packaging cell_graph into batch

If both ratios ~1-2× only:
  → Previous 12.5× slowdown was from DDP + full dataset
  → Real bottleneck might be in optimizer or DDP communication

=============================================================================
RAW LOG FILES
=============================================================================

086a Lazy Hetero DataLoader: ${OUTPUT_DIR}/086a_lazy_hetero_dataloader.log
086a DANGO DataLoader:       ${OUTPUT_DIR}/086a_dango_dataloader.log
086b Lazy Hetero Model:      ${OUTPUT_DIR}/086b_lazy_hetero_model.log
086b DANGO Model:            ${OUTPUT_DIR}/086b_dango_model.log

=============================================================================
EOF

echo ""
echo "================================================================================"
echo "PROFILING COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
ls -lh ${OUTPUT_DIR}
echo ""
echo "================================================================================"
echo "Comparison Report:"
echo "================================================================================"
cat ${OUTPUT_DIR}/comparison_086_${TIMESTAMP}.txt
echo "================================================================================"

# Exit with success
exit 0
