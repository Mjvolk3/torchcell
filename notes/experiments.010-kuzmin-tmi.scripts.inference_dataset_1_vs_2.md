---
id: ap9tymydt09gs7eyokuid8s
title: Inference_dataset_1_vs_2
desc: ''
updated: 1781029006715
created: 1781029006715
---

## 2026.06.09 - Quantify how the inference-2 filter reshapes the selected gene panel

This script exists to answer whether changing the triple-selection criterion from a flat fitness > 1.0 cut (inference-1, 275M triples) to an iterative fitness-improvement filter (inference-2, 479K triples) actually changes which genes get prioritized for experimental construction. It compares the top 12-gene panels (k=200) from the two selection strategies and renders a Venn diagram plus side-by-side gene lists, providing the evidence used to justify the inference-2 filter as producing a substantively different, more deliberately curated panel rather than a cosmetic variant.

### Specifics worth keeping

- Reads the two `singles_table_panel12_k200.csv` files (inference-1 from `results/`, inference-2 from `results/inference_2/`) under `EXPERIMENT_ROOT`.
- Output is a timestamped comparison figure under `ASSET_IMAGES_DIR/010-kuzmin-tmi/`; this is an analysis/visualization script, not a dataset builder.
- The takeaway it reports: the iterative-improvement filter yields a largely different panel, with only the overlapping subset shared out of 12.
