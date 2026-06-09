---
id: ocdsn2e88x8y82alwlul1h3
title: Run_inference_3_pipeline
desc: ''
updated: 1781029068950
created: 1781029068950
---

## 2026.06.09 - One-Command Orchestration of the Full Inference-3 Triple-Selection Run

This driver exists so the entire inference-3 study -- from generating candidate triples through model inference to gene-panel selection -- can be launched as a single chained SLURM submission, with each stage gated on its predecessor's success. Its reason for being is reproducibility and hands-off scheduling: a multi-day run spanning triple generation, LMDB build, multi-GPU inference, and selection is encoded once as an `afterok` dependency chain rather than being babysat stage by stage.

### Step order

- Step 1: `gh_generate_triple_combinations_inference_3.slurm` runs `generate_triple_combinations_inference_3.py` (filters: max(smf) > 1.04, all(smf) > 0.90, max(dmf) > 1.08, all(dmf) > 0.90), producing `inference_3/raw/triple_combinations_list.parquet`.
- Step 2 (afterok Step 1): `gh_inference_dataset_3.slurm` runs `inference_dataset_3.py`, producing `inference_3/processed/lmdb/`.
- Step 3 (afterok Step 2): `gh_equivariant_cell_graph_transformer_inference_3.slurm` runs `equivariant_cell_graph_transformer_inference_3.py` on 4 GPUs via torchrun, producing `inference_3/inferred/*.parquet`.
- Step 4 (afterok Step 3): `gh_select_12_and_24_gene_top_triples_inference_3.slurm` runs `select_12_and_24_genes_top_triples_inference_3.py`, producing the `results/inference_3/` tables and `gene_selection_results.csv`.
- Interaction significance uses the Jonckheere-Terpstra test for ordered alternatives (~96% power at n=8 replicates with a 0.04 gap).
