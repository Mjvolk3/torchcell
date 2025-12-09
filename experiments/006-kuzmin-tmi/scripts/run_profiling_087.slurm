#!/bin/bash
#SBATCH -p main
#SBATCH --mem=250g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=PROFILE-087
#SBATCH --time=6:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/slurm/output/%x_%j.out

# Change to torchcell root directory
cd /home/michaelvolk/Documents/projects/torchcell || exit 1
TORCHCELL_ROOT="/home/michaelvolk/Documents/projects/torchcell"

# Create timestamped output directory
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
OUTPUT_DIR="experiments/006-kuzmin-tmi/profiling_results/profiling_087_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${TORCHCELL_ROOT}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

echo "================================================================================"
echo "EXPERIMENT 087: COMPREHENSIVE GRAPH PROCESSOR PROFILING"
echo "================================================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "Tests:"
echo "  1. DANGO (Perturbation processor)"
echo "  2. Lazy Hetero (LazySubgraphRepresentation)"
echo "  3. NeighborSubgraph 1-hop"
echo "  4. NeighborSubgraph 2-hop"
echo "  5. NeighborSubgraph 3-hop"
echo "================================================================================"
echo ""

# Common log file for all runs
COMMON_LOG="${OUTPUT_DIR}/087_all_runs.log"

# Function to run profiling for a specific method
run_profiling() {
    local METHOD=$1
    local NUM_HOPS=$2
    local METHOD_DISPLAY=$3

    echo "================================================================================" | tee -a ${COMMON_LOG}
    echo "Running ${METHOD_DISPLAY} Profiling" | tee -a ${COMMON_LOG}
    echo "================================================================================" | tee -a ${COMMON_LOG}
    echo "Config:" | tee -a ${COMMON_LOG}
    echo "  - Method: ${METHOD}" | tee -a ${COMMON_LOG}
    if [ "$METHOD" = "neighbor" ]; then
        echo "  - num_hops: ${NUM_HOPS}" | tee -a ${COMMON_LOG}
    fi
    echo "  - subset_size: 10,000" | tee -a ${COMMON_LOG}
    echo "  - max_steps: 100" | tee -a ${COMMON_LOG}
    echo "" | tee -a ${COMMON_LOG}

    # Build command based on method
    if [ "$METHOD" = "neighbor" ]; then
        CMD="--method neighbor --num-hops ${NUM_HOPS}"
    else
        CMD="--method ${METHOD}"
    fi

    apptainer exec --nv \
      --bind /scratch:/scratch \
      --bind /home/michaelvolk/Documents/projects/torchcell/.env:/home/michaelvolk/Documents/projects/torchcell/.env \
      --env PYTHONUNBUFFERED=1 \
      /home/michaelvolk/Documents/projects/torchcell/rockylinux_9.sif bash -lc "
export PYTHONPATH='${TORCHCELL_ROOT}:\$PYTHONPATH'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/bin/activate
conda activate torchcell

python ${TORCHCELL_ROOT}/experiments/006-kuzmin-tmi/scripts/profile_087.py ${CMD} \
  2>&1 | tee -a ${COMMON_LOG}
" || { echo "ERROR: ${METHOD_DISPLAY} profiling failed" | tee -a ${COMMON_LOG}; return 1; }

    echo "" | tee -a ${COMMON_LOG}
    echo "${METHOD_DISPLAY} profiling completed" | tee -a ${COMMON_LOG}
    echo "" | tee -a ${COMMON_LOG}

    return 0
}

# Run profiling for all 5 methods
run_profiling "dango" 0 "DANGO (Perturbation)" || exit 1
run_profiling "lazy" 0 "Lazy Hetero (LazySubgraph)" || exit 1
run_profiling "neighbor" 1 "NeighborSubgraph 1-hop" || exit 1
run_profiling "neighbor" 2 "NeighborSubgraph 2-hop" || exit 1
run_profiling "neighbor" 3 "NeighborSubgraph 3-hop" || exit 1

# Extract metrics from common log file
echo "================================================================================"
echo "Extracting Results from All Runs"
echo "================================================================================"

# Extract metrics for a specific occurrence from common log
# Usage: extract_metric_nth <pattern> <occurrence_number>
extract_metric_nth() {
    local pattern=$1
    local nth=$2
    grep "$pattern" "${COMMON_LOG}" 2>/dev/null | grep -oP '\d+\.\d+' | sed -n "${nth}p"
}

extract_batch_size_nth() {
    local nth=$1
    grep "Estimated max batch size" "${COMMON_LOG}" 2>/dev/null | grep -oP '\d+' | sed -n "${nth}p"
}

# Metrics appear in order: DANGO, Lazy, 1-hop, 2-hop, 3-hop
# DANGO metrics (1st occurrence)
ITS_DANGO=$(extract_metric_nth "Average iterations/sec:" 1)
GPU_MEM_DANGO=$(extract_metric_nth "Average GPU memory per batch:" 1)
NODES_DANGO=$(extract_metric_nth "Average nodes per sample:" 1)
EDGES_DANGO=$(extract_metric_nth "Average edges per sample:" 1)
MAX_BATCH_DANGO=$(extract_batch_size_nth 1)

# Lazy (2nd occurrence)
ITS_LAZY=$(extract_metric_nth "Average iterations/sec:" 2)
GPU_MEM_LAZY=$(extract_metric_nth "Average GPU memory per batch:" 2)
NODES_LAZY=$(extract_metric_nth "Average nodes per sample:" 2)
EDGES_LAZY=$(extract_metric_nth "Average edges per sample:" 2)
MAX_BATCH_LAZY=$(extract_batch_size_nth 2)

# 1-hop metrics (3rd occurrence)
ITS_1HOP=$(extract_metric_nth "Average iterations/sec:" 3)
GPU_MEM_1HOP=$(extract_metric_nth "Average GPU memory per batch:" 3)
NODES_1HOP=$(extract_metric_nth "Average nodes per sample:" 3)
EDGES_1HOP=$(extract_metric_nth "Average edges per sample:" 3)
MAX_BATCH_1HOP=$(extract_batch_size_nth 3)

# 2-hop metrics (4th occurrence)
ITS_2HOP=$(extract_metric_nth "Average iterations/sec:" 4)
GPU_MEM_2HOP=$(extract_metric_nth "Average GPU memory per batch:" 4)
NODES_2HOP=$(extract_metric_nth "Average nodes per sample:" 4)
EDGES_2HOP=$(extract_metric_nth "Average edges per sample:" 4)
MAX_BATCH_2HOP=$(extract_batch_size_nth 4)

# 3-hop metrics (5th occurrence)
ITS_3HOP=$(extract_metric_nth "Average iterations/sec:" 5)
GPU_MEM_3HOP=$(extract_metric_nth "Average GPU memory per batch:" 5)
NODES_3HOP=$(extract_metric_nth "Average nodes per sample:" 5)
EDGES_3HOP=$(extract_metric_nth "Average edges per sample:" 5)
MAX_BATCH_3HOP=$(extract_batch_size_nth 5)

# Calculate speedups and slowdowns
calc_ratio() {
    local value=$1
    local reference=$2
    if [[ "$value" != "" && "$reference" != "" && "$reference" != "0" ]]; then
        echo "scale=2; $value / $reference" | bc
    else
        echo "N/A"
    fi
}

# Speedups vs Lazy (baseline for hetero methods)
SPEEDUP_1HOP_VS_LAZY=$(calc_ratio "$ITS_1HOP" "$ITS_LAZY")
SPEEDUP_2HOP_VS_LAZY=$(calc_ratio "$ITS_2HOP" "$ITS_LAZY")
SPEEDUP_3HOP_VS_LAZY=$(calc_ratio "$ITS_3HOP" "$ITS_LAZY")

# Speedups vs DANGO (overall fastest)
SPEEDUP_LAZY_VS_DANGO=$(calc_ratio "$ITS_LAZY" "$ITS_DANGO")
SPEEDUP_1HOP_VS_DANGO=$(calc_ratio "$ITS_1HOP" "$ITS_DANGO")
SPEEDUP_2HOP_VS_DANGO=$(calc_ratio "$ITS_2HOP" "$ITS_DANGO")
SPEEDUP_3HOP_VS_DANGO=$(calc_ratio "$ITS_3HOP" "$ITS_DANGO")

# Generate comprehensive summary report
cat > ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt <<EOF
=============================================================================
EXPERIMENT 087: COMPREHENSIVE GRAPH PROCESSOR PROFILING
Generated: ${TIMESTAMP}
=============================================================================

CONFIGURATION:
- GPU: Single GPU (no DDP)
- Dataset: 10,000 sample subset
- Batch size: 64 for DANGO, 28 for others
- Steps profiled: 100 (per run)

PURPOSE:
Compare all 5 graph processing methods under identical conditions:
1. DANGO (Perturbation) - Minimal graph data, fastest baseline
2. Lazy Hetero (LazySubgraphRepresentation) - Full graph + masks
3. NeighborSubgraph (1/2/3-hop) - Variable-sized k-hop neighborhoods

=============================================================================
RESULTS SUMMARY TABLE:
=============================================================================

| Method    | it/s      | Speedup   | GPU Mem   | Est. Max  | Nodes     | Edges     |
|           |           | vs DANGO  | (MB/batch)| Batch     | /sample   | /sample   |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| DANGO     | ${ITS_DANGO:-N/A}      | 1.00×     | ${GPU_MEM_DANGO:-N/A}       | ${MAX_BATCH_DANGO:-N/A}       | ${NODES_DANGO:-N/A}       | ${EDGES_DANGO:-N/A}       |
| LazySubg  | ${ITS_LAZY:-N/A}      | ${SPEEDUP_LAZY_VS_DANGO:-N/A}×     | ${GPU_MEM_LAZY:-N/A}       | ${MAX_BATCH_LAZY:-N/A}       | ${NODES_LAZY:-N/A}       | ${EDGES_LAZY:-N/A}       |
| 1-hop     | ${ITS_1HOP:-N/A}      | ${SPEEDUP_1HOP_VS_DANGO:-N/A}×     | ${GPU_MEM_1HOP:-N/A}       | ${MAX_BATCH_1HOP:-N/A}       | ${NODES_1HOP:-N/A}       | ${EDGES_1HOP:-N/A}       |
| 2-hop     | ${ITS_2HOP:-N/A}      | ${SPEEDUP_2HOP_VS_DANGO:-N/A}×     | ${GPU_MEM_2HOP:-N/A}       | ${MAX_BATCH_2HOP:-N/A}       | ${NODES_2HOP:-N/A}       | ${EDGES_2HOP:-N/A}       |
| 3-hop     | ${ITS_3HOP:-N/A}      | ${SPEEDUP_3HOP_VS_DANGO:-N/A}×     | ${GPU_MEM_3HOP:-N/A}       | ${MAX_BATCH_3HOP:-N/A}       | ${NODES_3HOP:-N/A}       | ${EDGES_3HOP:-N/A}       |

=============================================================================
DETAILED ANALYSIS:
=============================================================================

**Speed Rankings (iterations/sec):**
1. DANGO:     ${ITS_DANGO:-N/A} it/s (baseline - minimal graph data)
2. Lazy:      ${ITS_LAZY:-N/A} it/s (${SPEEDUP_LAZY_VS_DANGO:-N/A}× vs DANGO)
3. 1-hop:     ${ITS_1HOP:-N/A} it/s (${SPEEDUP_1HOP_VS_DANGO:-N/A}× vs DANGO, ${SPEEDUP_1HOP_VS_LAZY:-N/A}× vs Lazy)
4. 2-hop:     ${ITS_2HOP:-N/A} it/s (${SPEEDUP_2HOP_VS_DANGO:-N/A}× vs DANGO, ${SPEEDUP_2HOP_VS_LAZY:-N/A}× vs Lazy)
5. 3-hop:     ${ITS_3HOP:-N/A} it/s (${SPEEDUP_3HOP_VS_DANGO:-N/A}× vs DANGO, ${SPEEDUP_3HOP_VS_LAZY:-N/A}× vs Lazy)

**Memory Efficiency (GPU memory per batch):**
- DANGO:     ${GPU_MEM_DANGO:-N/A} MB (minimal - no graph edges)
- Lazy:      ${GPU_MEM_LAZY:-N/A} MB (full graph for all samples)
- 1-hop:     ${GPU_MEM_1HOP:-N/A} MB (smallest subgraphs)
- 2-hop:     ${GPU_MEM_2HOP:-N/A} MB (medium subgraphs)
- 3-hop:     ${GPU_MEM_3HOP:-N/A} MB (largest subgraphs)

**Graph Size per Sample:**
- DANGO:     ${NODES_DANGO:-N/A} nodes, ${EDGES_DANGO:-N/A} edges (no graph in batch)
- Lazy:      ${NODES_LAZY:-N/A} nodes, ${EDGES_LAZY:-N/A} edges (full 6607-node graph + masks)
- 1-hop:     ${NODES_1HOP:-N/A} nodes, ${EDGES_1HOP:-N/A} edges
- 2-hop:     ${NODES_2HOP:-N/A} nodes, ${EDGES_2HOP:-N/A} edges
- 3-hop:     ${NODES_3HOP:-N/A} nodes, ${EDGES_3HOP:-N/A} edges

**Effective Throughput (samples/sec):**
- DANGO:     $(echo "scale=1; ${ITS_DANGO:-0} * 64" | bc 2>/dev/null || echo "N/A") samples/sec (batch=64)
- Lazy:      $(echo "scale=1; ${ITS_LAZY:-0} * 28" | bc 2>/dev/null || echo "N/A") samples/sec (batch=28)
- 1-hop:     $(echo "scale=1; ${ITS_1HOP:-0} * 28" | bc 2>/dev/null || echo "N/A") samples/sec (batch=28)
- 2-hop:     $(echo "scale=1; ${ITS_2HOP:-0} * 28" | bc 2>/dev/null || echo "N/A") samples/sec (batch=28)
- 3-hop:     $(echo "scale=1; ${ITS_3HOP:-0} * 28" | bc 2>/dev/null || echo "N/A") samples/sec (batch=28)

=============================================================================
KEY FINDINGS:
=============================================================================

1. **DANGO Performance:**
   - Fastest method (${ITS_DANGO:-N/A} it/s) due to minimal graph data
   - Only stores perturbation indices, no graph structure in batches
   - Best for pure interaction prediction without graph context

2. **LazySubgraph vs NeighborSubgraph:**
   - Lazy uses full graph (${NODES_LAZY:-N/A} nodes) with boolean masks
   - 1-hop extracts ~${NODES_1HOP:-N/A} nodes/sample (${SPEEDUP_1HOP_VS_LAZY:-N/A}× faster)
   - 2-hop extracts ~${NODES_2HOP:-N/A} nodes/sample (${SPEEDUP_2HOP_VS_LAZY:-N/A}× faster)
   - 3-hop extracts ~${NODES_3HOP:-N/A} nodes/sample (${SPEEDUP_3HOP_VS_LAZY:-N/A}× faster)

3. **Memory Tradeoffs:**
   - Smaller hop counts allow larger batch sizes
   - 1-hop: max batch ~${MAX_BATCH_1HOP:-N/A}
   - 2-hop: max batch ~${MAX_BATCH_2HOP:-N/A}
   - 3-hop: max batch ~${MAX_BATCH_3HOP:-N/A}
   - Lazy: max batch ~${MAX_BATCH_LAZY:-N/A}

=============================================================================
RECOMMENDATIONS:
=============================================================================

1. **If model needs full graph context:**
   → Consider LazySubgraph despite slower speed
   → All samples see complete 6607-node graph

2. **If local neighborhoods sufficient:**
   → Use 1-hop or 2-hop NeighborSubgraph
   → Significantly faster with reasonable context

3. **If speed is critical:**
   → Use DANGO for pure interaction prediction
   → Trade graph features for maximum throughput

4. **For production training:**
   → Balance between speed and model performance
   → Test model accuracy vs throughput for each method

=============================================================================
OUTPUT FILES:
=============================================================================

All profiling runs (DANGO, Lazy, 1/2/3-hop):
  ${OUTPUT_DIR}/087_all_runs.log

Summary report (this file):
  ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt

SLURM scheduler output:
  experiments/006-kuzmin-tmi/slurm/output/PROFILE-087_${SLURM_JOB_ID}.out

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
echo "SLURM output: experiments/006-kuzmin-tmi/slurm/output/PROFILE-087_${SLURM_JOB_ID}.out"
echo ""
echo "================================================================================"
echo "Summary Report:"
echo "================================================================================"
cat ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt
echo "================================================================================"

# Exit with success
exit 0
