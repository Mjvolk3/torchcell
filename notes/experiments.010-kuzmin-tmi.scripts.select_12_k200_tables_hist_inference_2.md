---
id: 5c0o5a2wrz71sy9mbghzew7
title: Select_12_k200_tables_hist_inference_2
desc: ''
updated: 1781029089694
created: 1781029089694
---

## 2026.06.09 - Turning the Inference_2 12-Gene Panel Into a Buildable Strain Plan and a Coverage Story

This script makes the chosen inference_2 12-gene panel (k=200) actionable and communicable: it lists exactly which single strains and double crosses are needed, ranks the doubles by how many top-k predicted triples each one unlocks (so construction can be sequenced for fastest coverage), and produces an overlay histogram contrasting the full panel's predicted interactions against the top-k constructible subset. It exists so the panel-selection result can be handed off as a prioritized wet-lab worklist and a single figure that justifies the panel.

### Specifics worth keeping

- Inputs: the inference_2 best-model (Pearson=0.4619) predictions, `gene_selection_results.csv`, and the `top_k_constructible_panel12_k200.csv` / `constructible_triples_panel12_k200.parquet` produced by the selection script.
- Doubles table adds `enables_triple_in_top_k` plus a cumulative "reachable top-k triples as you build pairs in order" column for sequencing construction.
- Outputs: `singles_table_*`, `doubles_table_*`, `triples_table_*` CSVs under `results/inference_2/`, and a timestamped overlay histogram in `ASSET_IMAGES_DIR/010-kuzmin-tmi/`.
