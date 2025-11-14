#!/bin/bash
#SBATCH -p main
#SBATCH --mem=500g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --job-name=006-086-DATALOADER-PROFILE-DDP
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
OUTPUT_DIR="experiments/006-kuzmin-tmi/profiling_results/dataloader_ddp_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${TORCHCELL_ROOT}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

echo "================================================================================"
echo "DATALOADER PROFILING WITH DDP (4 GPUs)"
echo "================================================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "This measures DataLoader + collation + GPU transfer + DDP overhead"
echo "Model forward/backward replaced with trivial operations"
echo "================================================================================"

# STEP 1: Profile Lazy Hetero dataloader overhead
echo ""
echo "================================================================================"
echo "STEP 1/3: Profiling Lazy Hetero DataLoader (4 GPU DDP, 100 steps)"
echo "================================================================================"
echo "Config: hetero_cell_bipartite_dango_gi_gh_085_dataloader_profile"
echo ""

apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
  --env PYTHONUNBUFFERED=1 \
  /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc "
# Add the project root to Python path
export PYTHONPATH='${TORCHCELL_ROOT}:\$PYTHONPATH'

# Enable expandable memory segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set DDP environment variables
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  ${TORCHCELL_ROOT}/experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_lazy.py \
  --config-name hetero_cell_bipartite_dango_gi_gh_085_dataloader_profile \
  2>&1 | tee ${OUTPUT_DIR}/lazy_hetero_dataloader_profile.log
" || { echo "ERROR: Lazy Hetero profiling failed"; exit 1; }

echo ""
echo "Lazy Hetero profiling completed successfully"
echo ""

# STEP 2: Profile DANGO dataloader overhead
echo "================================================================================"
echo "STEP 2/3: Profiling DANGO DataLoader (4 GPU DDP, 100 steps)"
echo "================================================================================"
echo "Config: dango_kuzmin2018_tmi_string12_0_dataloader_profile"
echo ""

apptainer exec --nv \
  --bind /scratch:/scratch \
  --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
  --env PYTHONUNBUFFERED=1 \
  /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc "
# Add the project root to Python path
export PYTHONPATH='${TORCHCELL_ROOT}:\$PYTHONPATH'

# Enable expandable memory segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set DDP environment variables
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT}

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  ${TORCHCELL_ROOT}/experiments/006-kuzmin-tmi/scripts/dango.py \
  --config-name dango_kuzmin2018_tmi_string12_0_dataloader_profile \
  2>&1 | tee ${OUTPUT_DIR}/dango_dataloader_profile.log
" || { echo "ERROR: DANGO profiling failed"; exit 1; }

echo ""
echo "DANGO profiling completed successfully"
echo ""

# STEP 3: Generate comparison report
echo "================================================================================"
echo "STEP 3/3: Generating Comparison Report"
echo "================================================================================"

# Extract key metrics from logs using grep and awk
echo "Extracting metrics from log files..."

# Function to extract average it/s from log file
extract_avg_its() {
    local logfile=$1
    # Look for patterns like "X.XX it/s" and average the last 10 values
    grep -oP '\d+\.\d+(?=it/s)' "$logfile" 2>/dev/null | tail -20 | awk '{sum+=$1; count++} END {if(count>0) printf "%.3f", sum/count; else print "N/A"}'
}

# Extract metrics
LAZY_AVG_ITS=$(extract_avg_its "${OUTPUT_DIR}/lazy_hetero_dataloader_profile.log")
DANGO_AVG_ITS=$(extract_avg_its "${OUTPUT_DIR}/dango_dataloader_profile.log")

# Calculate ratio if both values are numeric
if [[ "$LAZY_AVG_ITS" != "N/A" && "$DANGO_AVG_ITS" != "N/A" ]]; then
    RATIO=$(echo "scale=2; $DANGO_AVG_ITS / $LAZY_AVG_ITS" | bc)
else
    RATIO="N/A"
fi

# Generate comparison report
cat > ${OUTPUT_DIR}/comparison_dataloader_${TIMESTAMP}.txt <<EOF
=============================================================================
DATALOADER PROFILING COMPARISON (DDP, 4 GPUs)
Generated: ${TIMESTAMP}
=============================================================================

CONFIGURATION:
- Mode: DataLoader profiling (model forward/backward skipped)
- GPUs: 4 (DDP strategy)
- Steps per run: 100
- Batch preparation: Data loading + collation + GPU transfer + DDP sync

PURPOSE:
This profiling measures ONLY the data pipeline overhead, excluding model
computation. The difference between this and full-step profiling reveals
the pure model compute cost.

=============================================================================
LAZY HETERO RESULTS (Hetero Cell Bipartite DANGO GI):
=============================================================================

Average iterations/sec: ${LAZY_AVG_ITS}

Last 20 it/s measurements:
$(grep -oP '\d+\.\d+(?=it/s)' "${OUTPUT_DIR}/lazy_hetero_dataloader_profile.log" 2>/dev/null | tail -20 | nl)

Sample log output:
$(grep -E "Execution mode|DataLoader|it/s|Epoch.*time" "${OUTPUT_DIR}/lazy_hetero_dataloader_profile.log" 2>/dev/null | tail -30)

=============================================================================
DANGO RESULTS:
=============================================================================

Average iterations/sec: ${DANGO_AVG_ITS}

Last 20 it/s measurements:
$(grep -oP '\d+\.\d+(?=it/s)' "${OUTPUT_DIR}/dango_dataloader_profile.log" 2>/dev/null | tail -20 | nl)

Sample log output:
$(grep -E "Execution mode|DataLoader|it/s|Epoch.*time" "${OUTPUT_DIR}/dango_dataloader_profile.log" 2>/dev/null | tail -30)

=============================================================================
COMPARISON SUMMARY:
=============================================================================

DANGO avg it/s:      ${DANGO_AVG_ITS}
Lazy Hetero avg it/s: ${LAZY_AVG_ITS}

Speedup ratio (DANGO / Lazy Hetero): ${RATIO}x

INTERPRETATION:
- These numbers represent ONLY data pipeline overhead (no model compute)
- If ratio is close to 1.0: Data pipelines are comparable
- If ratio is >2.0: Lazy Hetero has significantly higher data overhead
- Compare with full-step profiling to isolate model vs data costs

Expected analysis:
  Full-step ratio = (Model compute ratio) × (Data pipeline ratio)

If full-step shows 40× slowdown but data pipeline is only ${RATIO}×,
then model compute must account for the remaining difference.

=============================================================================
NEXT STEPS:
=============================================================================

1. Compare with full-step profiling results to decompose:
   - Pure model forward+backward cost
   - Pure data pipeline cost

2. If data pipeline ratio is high (>${RATIO}×):
   - Investigate batch collation differences
   - Check DataLoader worker efficiency
   - Profile CPU→GPU transfer sizes

3. If model compute ratio is high:
   - Consider architectural changes (package cell_graph into batch)
   - Optimize message passing patterns
   - Reduce dual forward passes

=============================================================================
RAW LOG FILES:
=============================================================================

Lazy Hetero: ${OUTPUT_DIR}/lazy_hetero_dataloader_profile.log
DANGO:       ${OUTPUT_DIR}/dango_dataloader_profile.log

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
cat ${OUTPUT_DIR}/comparison_dataloader_${TIMESTAMP}.txt
echo "================================================================================"

# Exit with success
exit 0
