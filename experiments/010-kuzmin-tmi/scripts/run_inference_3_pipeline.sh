#!/bin/bash
# experiments/010-kuzmin-tmi/scripts/run_inference_3_pipeline.sh
# Run the complete inference_3 pipeline with job dependencies
#
# Usage: bash experiments/010-kuzmin-tmi/scripts/run_inference_3_pipeline.sh

set -e

SCRIPT_DIR="experiments/010-kuzmin-tmi/scripts"
RESULTS_DIR="experiments/010-kuzmin-tmi/results/inference_3"

echo "============================================================"
echo "INFERENCE DATASET 3 PIPELINE"
echo "============================================================"
echo "Thresholds: max(smf)>1.04, all(smf)>0.90, max(dmf)>1.08, all(dmf)>0.90"
echo "Inference: 4 GPUs via torchrun (each processes 1/4 of dataset)"
echo "Statistical test: Jonckheere-Terpstra (JT) for ordered alternatives"
echo "Power: ~96% at n=8 replicates with 0.04 gap"
echo ""

# Create results directory
mkdir -p ${RESULTS_DIR}
echo "Results will be saved to: ${RESULTS_DIR}"
echo ""

# Step 1: Generate Triple Combinations
echo "[Step 1] Submitting triple generation..."
JOB1=$(sbatch --parsable ${SCRIPT_DIR}/gh_generate_triple_combinations_inference_3.slurm)
echo "  Job ID: $JOB1"
echo "  Script: generate_triple_combinations_inference_3.py"
echo "  Output: inference_3/raw/triple_combinations_list.parquet"
echo ""

# Step 2: Create LMDB Dataset (depends on Step 1)
echo "[Step 2] Submitting LMDB creation (depends on $JOB1)..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 ${SCRIPT_DIR}/gh_inference_dataset_3.slurm)
echo "  Job ID: $JOB2"
echo "  Script: inference_dataset_3.py"
echo "  Output: inference_3/processed/lmdb/"
echo ""

# Step 3: Model Inference (depends on Step 2)
echo "[Step 3] Submitting model inference (depends on $JOB2)..."
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 ${SCRIPT_DIR}/gh_equivariant_cell_graph_transformer_inference_3.slurm)
echo "  Job ID: $JOB3"
echo "  Script: equivariant_cell_graph_transformer_inference_3.py"
echo "  Output: inference_3/inferred/*.parquet"
echo ""

# Step 4: Gene Panel Selection (depends on Step 3)
echo "[Step 4] Submitting gene selection (depends on $JOB3)..."
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 ${SCRIPT_DIR}/gh_select_12_and_24_gene_top_triples_inference_3.slurm)
echo "  Job ID: $JOB4"
echo "  Script: select_12_and_24_genes_top_triples_inference_3.py"
echo "  Output: ${RESULTS_DIR}/singles_table_panel12_k200.csv (and more)"
echo ""

echo "============================================================"
echo "PIPELINE SUBMITTED"
echo "============================================================"
echo ""
echo "Job chain:"
echo "  Step 1 (triples):   $JOB1"
echo "  Step 2 (LMDB):      $JOB2 → depends on $JOB1"
echo "  Step 3 (inference): $JOB3 → depends on $JOB2"
echo "  Step 4 (selection): $JOB4 → depends on $JOB3"
echo ""
echo "Expected timeline:"
echo "  Step 1: ~8 hours (triple generation)"
echo "  Step 2: ~12 hours (LMDB creation)"
echo "  Step 3: ~24-48 hours (model inference, depends on triple count)"
echo "  Step 4: ~2 hours (gene panel selection)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel $JOB1 $JOB2 $JOB3 $JOB4"
echo ""
echo "Final outputs will be in: ${RESULTS_DIR}/"
echo "  - singles_table_panel12_k200.csv"
echo "  - doubles_table_panel12_k200.csv"
echo "  - triples_table_panel12_k200.csv"
echo "  - gene_selection_results.csv"
