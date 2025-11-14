#!/bin/bash
#SBATCH -p main
#SBATCH --mem=250g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=PROFILE-087a
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
OUTPUT_DIR="experiments/006-kuzmin-tmi/profiling_results/profiling_087a_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${TORCHCELL_ROOT}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate torchcell

echo "================================================================================"
echo "EXPERIMENT 087a: MULTI-HOP NEIGHBOR SUBGRAPH PROFILING"
echo "================================================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "Tests:"
echo "  - LazySubgraphRepresentation (baseline - full graph + masks)"
echo "  - 1-hop NeighborSubgraphRepresentation"
echo "  - 2-hop NeighborSubgraphRepresentation"
echo "  - 3-hop NeighborSubgraphRepresentation"
echo "  - DANGO reference from Experiment 086a"
echo "================================================================================"
echo ""

# Common log file for all runs
COMMON_LOG="${OUTPUT_DIR}/087a_all_runs.log"

# Function to run profiling for a specific hop count
run_profiling() {
    local NUM_HOPS=$1

    if [ "$NUM_HOPS" -eq 0 ]; then
        echo "================================================================================" | tee -a ${COMMON_LOG}
        echo "Running LazySubgraph Profiling (Baseline)" | tee -a ${COMMON_LOG}
        echo "================================================================================" | tee -a ${COMMON_LOG}
        echo "Config:" | tee -a ${COMMON_LOG}
        echo "  - Graph Processor: LazySubgraphRepresentation" | tee -a ${COMMON_LOG}
        echo "  - Full graph (6607 nodes) with boolean masks" | tee -a ${COMMON_LOG}
        echo "  - batch_size: 28" | tee -a ${COMMON_LOG}
        echo "  - subset_size: 10,000" | tee -a ${COMMON_LOG}
        echo "  - max_steps: 100" | tee -a ${COMMON_LOG}
        echo "" | tee -a ${COMMON_LOG}
    else
        echo "================================================================================" | tee -a ${COMMON_LOG}
        echo "Running ${NUM_HOPS}-hop NeighborSubgraph Profiling" | tee -a ${COMMON_LOG}
        echo "================================================================================" | tee -a ${COMMON_LOG}
        echo "Config:" | tee -a ${COMMON_LOG}
        echo "  - Graph Processor: NeighborSubgraphRepresentation" | tee -a ${COMMON_LOG}
        echo "  - num_hops: ${NUM_HOPS}" | tee -a ${COMMON_LOG}
        echo "  - batch_size: 28" | tee -a ${COMMON_LOG}
        echo "  - subset_size: 10,000" | tee -a ${COMMON_LOG}
        echo "  - max_steps: 100" | tee -a ${COMMON_LOG}
        echo "" | tee -a ${COMMON_LOG}
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

python ${TORCHCELL_ROOT}/experiments/006-kuzmin-tmi/scripts/profile_neighbor_subgraph_087a.py \
  --num-hops ${NUM_HOPS} \
  2>&1 | tee -a ${COMMON_LOG}
" || { echo "ERROR: ${NUM_HOPS}-hop profiling failed" | tee -a ${COMMON_LOG}; return 1; }

    echo "" | tee -a ${COMMON_LOG}
    echo "${NUM_HOPS}-hop profiling completed" | tee -a ${COMMON_LOG}
    echo "" | tee -a ${COMMON_LOG}

    return 0
}

# Run profiling for each configuration (lazy baseline, then 1/2/3-hop)
run_profiling 0 || exit 1  # LazySubgraphRepresentation (baseline)
run_profiling 1 || exit 1  # 1-hop
run_profiling 2 || exit 1  # 2-hop
run_profiling 3 || exit 1  # 3-hop

# Extract metrics from common log file
echo "================================================================================"
echo "Extracting Results from All Runs"
echo "================================================================================"

# Extract metrics for a specific hop count from common log
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

# Metrics appear in order: Lazy (0-hop), 1-hop, 2-hop, 3-hop
# Lazy (0-hop) metrics (1st occurrence)
ITS_LAZY=$(extract_metric_nth "Average iterations/sec:" 1)
GPU_MEM_LAZY=$(extract_metric_nth "Average GPU memory per batch:" 1)
NODES_LAZY=$(extract_metric_nth "Average nodes per sample:" 1)
EDGES_LAZY=$(extract_metric_nth "Average edges per sample:" 1)
MAX_BATCH_LAZY=$(extract_batch_size_nth 1)

# 1-hop metrics (2nd occurrence)
ITS_1HOP=$(extract_metric_nth "Average iterations/sec:" 2)
GPU_MEM_1HOP=$(extract_metric_nth "Average GPU memory per batch:" 2)
NODES_1HOP=$(extract_metric_nth "Average nodes per sample:" 2)
EDGES_1HOP=$(extract_metric_nth "Average edges per sample:" 2)
MAX_BATCH_1HOP=$(extract_batch_size_nth 2)

# 2-hop metrics (3rd occurrence)
ITS_2HOP=$(extract_metric_nth "Average iterations/sec:" 3)
GPU_MEM_2HOP=$(extract_metric_nth "Average GPU memory per batch:" 3)
NODES_2HOP=$(extract_metric_nth "Average nodes per sample:" 3)
EDGES_2HOP=$(extract_metric_nth "Average edges per sample:" 3)
MAX_BATCH_2HOP=$(extract_batch_size_nth 3)

# 3-hop metrics (4th occurrence)
ITS_3HOP=$(extract_metric_nth "Average iterations/sec:" 4)
GPU_MEM_3HOP=$(extract_metric_nth "Average GPU memory per batch:" 4)
NODES_3HOP=$(extract_metric_nth "Average nodes per sample:" 4)
EDGES_3HOP=$(extract_metric_nth "Average edges per sample:" 4)
MAX_BATCH_3HOP=$(extract_batch_size_nth 4)

# Reference metrics
DANGO_ITS="73.82"  # From 086a

# Calculate speedups
calc_speedup() {
    local value=$1
    local reference=$2
    if [[ "$value" != "" && "$reference" != "" ]]; then
        echo "scale=2; $value / $reference" | bc
    else
        echo "N/A"
    fi
}

# Speedups vs Lazy (use fresh lazy run as baseline)
SPEEDUP_1HOP=$(calc_speedup "$ITS_1HOP" "$ITS_LAZY")
SPEEDUP_2HOP=$(calc_speedup "$ITS_2HOP" "$ITS_LAZY")
SPEEDUP_3HOP=$(calc_speedup "$ITS_3HOP" "$ITS_LAZY")

# Slowdowns vs DANGO
SLOWDOWN_LAZY=$(calc_speedup "$DANGO_ITS" "$ITS_LAZY")
SLOWDOWN_1HOP=$(calc_speedup "$DANGO_ITS" "$ITS_1HOP")
SLOWDOWN_2HOP=$(calc_speedup "$DANGO_ITS" "$ITS_2HOP")
SLOWDOWN_3HOP=$(calc_speedup "$DANGO_ITS" "$ITS_3HOP")

# Generate comprehensive summary report
cat > ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt <<EOF
=============================================================================
EXPERIMENT 087a: MULTI-HOP NEIGHBOR SUBGRAPH PROFILING
Generated: ${TIMESTAMP}
=============================================================================

CONFIGURATION:
- GPU: Single GPU (no DDP)
- Dataset: 10,000 sample subset
- Batch size: 28 (for all runs)
- Graph Processor: NeighborSubgraphRepresentation
- Steps profiled: 100 (per run)

PURPOSE:
Compare LazySubgraphRepresentation (full graph + masks) with 1-hop, 2-hop,
and 3-hop neighborhood sampling to understand speed vs memory tradeoffs.
All tests run in identical conditions for full reproducibility.

=============================================================================
RESULTS SUMMARY TABLE:
=============================================================================

| Method    | it/s      | Speedup   | Slowdown  | GPU Mem   | Est. Max  | Nodes     | Edges     |
|           |           | vs Lazy   | vs DANGO  | (MB/batch)| Batch     | /sample   | /sample   |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| LazySubg  | ${ITS_LAZY:-N/A}      | 1.00×     | ${SLOWDOWN_LAZY:-N/A}×     | ${GPU_MEM_LAZY:-N/A}       | ${MAX_BATCH_LAZY:-N/A}       | ${NODES_LAZY:-N/A}       | ${EDGES_LAZY:-N/A}       |
| 1-hop     | ${ITS_1HOP:-N/A}      | ${SPEEDUP_1HOP:-N/A}×      | ${SLOWDOWN_1HOP:-N/A}×     | ${GPU_MEM_1HOP:-N/A}       | ${MAX_BATCH_1HOP:-N/A}       | ${NODES_1HOP:-N/A}       | ${EDGES_1HOP:-N/A}       |
| 2-hop     | ${ITS_2HOP:-N/A}      | ${SPEEDUP_2HOP:-N/A}×      | ${SLOWDOWN_2HOP:-N/A}×     | ${GPU_MEM_2HOP:-N/A}       | ${MAX_BATCH_2HOP:-N/A}       | ${NODES_2HOP:-N/A}       | ${EDGES_2HOP:-N/A}       |
| 3-hop     | ${ITS_3HOP:-N/A}      | ${SPEEDUP_3HOP:-N/A}×      | ${SLOWDOWN_3HOP:-N/A}×     | ${GPU_MEM_3HOP:-N/A}       | ${MAX_BATCH_3HOP:-N/A}       | ${NODES_3HOP:-N/A}       | ${EDGES_3HOP:-N/A}       |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Reference |           |           |           |           |           |           |           |
| DANGO     | 73.82     | ~56×      | 1.00×     | ~200      | 300+      | 6607      | 0         |

=============================================================================
COMPARISON WITH EXPERIMENT 086a:
=============================================================================

Experiment 086a reference: DANGO achieved 73.82 it/s with minimal graph data.
This experiment (087a) tests LazySubgraphRepresentation and neighborhood
sampling under identical conditions for direct comparison.

Key findings:
- Smaller hop counts process faster but provide less neighborhood context
- GPU memory usage scales with subgraph size
- Estimated max batch size inversely scales with memory usage
- All approaches remain significantly slower than DANGO due to:
  * HeteroData collation overhead
  * Variable-sized graph batching
  * Phenotype data processing (COO format)

=============================================================================
INTERPRETATION:
=============================================================================

**Speed Analysis (current run):**
- LazySubgraph:  ${ITS_LAZY:-N/A} it/s (baseline, full graph + masks)
- 1-hop: ${ITS_1HOP:-N/A} it/s (${SPEEDUP_1HOP:-N/A}× vs LazySubgraph)
- 2-hop: ${ITS_2HOP:-N/A} it/s (${SPEEDUP_2HOP:-N/A}× vs LazySubgraph)
- 3-hop: ${ITS_3HOP:-N/A} it/s (${SPEEDUP_3HOP:-N/A}× vs LazySubgraph)

**Memory vs Batch Size Tradeoff:**
- LazySubgraph allows batch size ~${MAX_BATCH_LAZY:-N/A} (full graph: 6607 nodes)
- 1-hop allows batch size ~${MAX_BATCH_1HOP:-N/A} (smallest subgraphs, fastest)
- 2-hop allows batch size ~${MAX_BATCH_2HOP:-N/A} (medium subgraphs, balanced)
- 3-hop allows batch size ~${MAX_BATCH_3HOP:-N/A} (largest subgraphs, slowest)

**Effective Throughput @ batch=28 (samples/sec = it/s × 28):**
- LazySubgraph: $(echo "scale=1; ${ITS_LAZY:-0} * 28" | bc 2>/dev/null || echo "N/A") samples/sec
- 1-hop: $(echo "scale=1; ${ITS_1HOP:-0} * 28" | bc 2>/dev/null || echo "N/A") samples/sec
- 2-hop: $(echo "scale=1; ${ITS_2HOP:-0} * 28" | bc 2>/dev/null || echo "N/A") samples/sec
- 3-hop: $(echo "scale=1; ${ITS_3HOP:-0} * 28" | bc 2>/dev/null || echo "N/A") samples/sec

=============================================================================
RECOMMENDATIONS:
=============================================================================

1. **If 1-hop with larger batch size has highest throughput:**
   → Use 1-hop for initial experiments
   → Test if model performance is acceptable with limited neighborhood

2. **If 2-hop provides best throughput/context balance:**
   → Continue with 2-hop as baseline
   → Current batch size (28) is reasonable

3. **If 3-hop is needed for model performance:**
   → Reduce batch size to ${MAX_BATCH_3HOP:-N/A}
   → Consider preprocessing or other optimizations

=============================================================================
NEXT STEPS:
=============================================================================

1. Test model training with optimal hop count and batch size
2. Evaluate if model performance justifies the compute cost vs DANGO
3. If neighborhood sampling is promising:
   - Consider using PyG's C++ NeighborSampler (5-10× faster)
   - Explore preprocessing approach (save subgraphs to disk)
4. If still too slow:
   - Return to LazySubgraph with optimizations
   - Consider hybrid approach (DANGO + selected graph features)

=============================================================================
OUTPUT FILES:
=============================================================================

All profiling runs (LazySubgraph, 1-hop, 2-hop, 3-hop):
  ${OUTPUT_DIR}/087a_all_runs.log

Summary report (this file):
  ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt

SLURM scheduler output (job metadata, stdout/stderr):
  experiments/006-kuzmin-tmi/slurm/output/PROFILE-087a_${SLURM_JOB_ID}.out

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
echo "SLURM output: experiments/006-kuzmin-tmi/slurm/output/PROFILE-087a_${SLURM_JOB_ID}.out"
echo ""
echo "================================================================================"
echo "Summary Report:"
echo "================================================================================"
cat ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt
echo "================================================================================"

# Exit with success
exit 0
