---
id: q161ealm324qnyn9vp1ohtk
title: 12_panel_inference_2_queried_data_tables
desc: ''
updated: 1770258304235
created: 1769795775780
---
...

## 2026.06.09 - Grounding the Inference_2 Panel Singles in Existing Single-Mutant Fitness

This script enriches the inference_2 12-gene panel's singles table with single-mutant fitness (SMF) measurements from Costanzo2016, Kuzmin2018, and Kuzmin2020, so the team can see which panel genes already have published baseline fitness before building strains. It exists as the inference_2 (singles-only) precursor to the fuller inference_3 queried-data workflow, establishing per-source SMF coverage for the chosen panel.

### Specifics worth keeping

- Reads `singles_table_panel12_k200.csv`, matches each gene via a frozenset single-gene key against each dataset's index, and reports per-source match counts.
- Scope is Phase 1 / singles only (no doubles or triples), unlike the inference_3 version.
- Output: `singles_table_panel12_k200_queried.csv` under `results/inference_2/`.