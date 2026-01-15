#!/bin/bash
# experiments/012-sameith-kemmeren/scripts/012-sameith-kemmeren
# [[experiments.012-sameith-kemmeren.scripts.012-sameith-kemmeren]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/012-sameith-kemmeren/scripts/012-sameith-kemmeren

set -e  # Exit on error

# Kemmeren & Sameith microarray dataset analysis pipeline
# Executes all 6 analysis tasks in sequence

# Task 1: Metadata verification
python experiments/012-sameith-kemmeren/scripts/verify_metadata.py

# Task 2: Single mutant expression distributions
python experiments/012-sameith-kemmeren/scripts/single_mutant_expression_distributions.py

# Task 3: Double mutant combined heatmap
python experiments/012-sameith-kemmeren/scripts/double_mutant_combined_heatmap.py

# Task 4: Gene-by-gene expression correlation
python experiments/012-sameith-kemmeren/scripts/gene_by_gene_expression_correlation.py

# Task 5: Kemmeren-Sameith overlap analysis
python experiments/012-sameith-kemmeren/scripts/kemmeren_sameith_overlap_analysis.py

# Task 6: Noise comparison analysis
python experiments/012-sameith-kemmeren/scripts/noise_comparison_analysis.py
