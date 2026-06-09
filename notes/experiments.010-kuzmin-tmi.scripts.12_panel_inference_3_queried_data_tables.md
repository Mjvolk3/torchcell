---
id: vs30nu08bplt5gsgd6r6iga
title: 12_panel_inference_3_queried_data_tables
desc: ''
updated: 1781028965315
created: 1781028965315
---

## 2026.06.09 - Grounding the Inference_3 Panel in Existing Experimental Measurements

This script checks how much of the inference_3 12-gene panel is already measured in the literature by enriching its singles and doubles tables with experimental fitness and interaction values from Costanzo2016, Kuzmin2018, and Kuzmin2020. It exists so the team knows which panel strains and crosses have prior data (avoiding redundant construction and enabling source-vs-source and model-vs-experiment comparison) before committing to the wet-lab build; its queried CSVs are the direct input to the fitness-comparison script.

### Specifics worth keeping

- Builds per-dataset frozenset gene-key indices, then matches singles by single-gene key and doubles by gene-pair key.
- Singles pull SMF (Costanzo2016, Kuzmin2018, Kuzmin2020); doubles pull both DMF (fitness) and DMI (interaction + p-value) from all three sources.
- Outputs `singles_table_panel12_k200_queried.csv` and `doubles_table_panel12_k200_queried.csv` under `results/inference_3/`, plus per-source match counts.
