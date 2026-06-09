#!/bin/bash
# experiments/010-kuzmin-tmi/scripts/run_inference_3_merge_and_select.sh
# Recovery pipeline: merge existing shards + gene panel selection
# Use after inference completed but merge failed (ArrowInvalid offset overflow)
#
# Usage: bash experiments/010-kuzmin-tmi/scripts/run_inference_3_merge_and_select.sh

set -e

SCRIPT_DIR="experiments/010-kuzmin-tmi/scripts"
RESULTS_DIR="experiments/010-kuzmin-tmi/results/inference_3"

mkdir -p ${RESULTS_DIR}

echo "============================================================"
echo "INFERENCE 3: MERGE SHARDS + GENE SELECTION"
echo "============================================================"
echo ""

# Step 1: Merge parquet shards (no GPU needed)
echo "[Step 1/2] Submitting shard merge..."
JOB_MERGE=$(sbatch --parsable ${SCRIPT_DIR}/gh_merge_inference_3_shards.slurm)
echo "  Job ID: $JOB_MERGE"
echo "  Script: equivariant_cell_graph_transformer_inference_3.py --merge-only"
echo ""

# Step 2: Gene Panel Selection (depends on merge)
echo "[Step 2/2] Submitting gene selection (depends on $JOB_MERGE)..."
JOB_SELECT=$(sbatch --parsable --dependency=afterok:$JOB_MERGE ${SCRIPT_DIR}/gh_select_12_and_24_gene_top_triples_inference_3.slurm)
echo "  Job ID: $JOB_SELECT"
echo "  Script: select_12_and_24_genes_top_triples_inference_3.py"
echo ""

echo "============================================================"
echo "SUBMITTED"
echo "============================================================"
echo ""
echo "Job chain:"
echo "  Merge:     $JOB_MERGE"
echo "  Selection: $JOB_SELECT → depends on $JOB_MERGE"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel $JOB_MERGE $JOB_SELECT"
echo ""
echo "Final outputs will be in: ${RESULTS_DIR}/"
echo "  - singles_table_panel12_k200.csv"
echo "  - doubles_table_panel12_k200.csv"
echo "  - triples_table_panel12_k200.csv"
echo "  - gene_selection_results.csv"
