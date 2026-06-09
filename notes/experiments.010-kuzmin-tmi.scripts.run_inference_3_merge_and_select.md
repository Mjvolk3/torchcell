---
id: jfpl1d8m3joektkhu28qzsz
title: Run_inference_3_merge_and_select
desc: ''
updated: 1781029062042
created: 1781029062042
---

## 2026.06.09 - Recovering the Inference-3 Run After a Merge Failure

This driver exists as a recovery path for the inference-3 pipeline: when model inference finished but the shard merge crashed (an Arrow `ArrowInvalid` offset overflow), it lets the team resume from the merge step without re-running the expensive GPU inference. It re-submits just the final two stages as a SLURM dependency chain so the panel-selection products are still produced.

### Step order

- Step 1: `sbatch gh_merge_inference_3_shards.slurm`, which runs `equivariant_cell_graph_transformer_inference_3.py --merge-only` (no GPU) to merge the inferred parquet shards.
- Step 2: `sbatch --dependency=afterok:<merge> gh_select_12_and_24_gene_top_triples_inference_3.slurm`, running `select_12_and_24_genes_top_triples_inference_3.py`.
- Final products land in `results/inference_3/`: `singles_table_panel12_k200.csv`, `doubles_table_panel12_k200.csv`, `triples_table_panel12_k200.csv`, and `gene_selection_results.csv`.
