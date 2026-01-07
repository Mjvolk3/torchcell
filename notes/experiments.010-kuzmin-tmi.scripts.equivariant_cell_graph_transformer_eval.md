---
id: j4l5v76dwu87lvc9u2ssjyd
title: Equivariant_cell_graph_transformer_eval
desc: ''
updated: 1767754413697
created: 1767747273449
---

## Overview

Full evaluation script for CellGraphTransformer on validation and test sets. Unlike training-time metrics (which subsample for visualization), this script:

- Collects ALL predictions (no subsampling)
- Computes comprehensive metrics beyond MSE/Pearson
- Saves predictions to Parquet for downstream analysis
- Logs to the SAME wandb group as training for direct comparison

## Usage

```bash
python experiments/010-kuzmin-tmi/scripts/equivariant_cell_graph_transformer_eval.py \
  --config-name equivariant_cell_graph_transformer_eval
```

## Outputs

| Output                | Location                                                                             |
|-----------------------|--------------------------------------------------------------------------------------|
| Predictions (Parquet) | `experiments/010-kuzmin-tmi/results/eval/predictions_{val,test}_{timestamp}.parquet` |
| Metrics               | Logged to wandb under same training group                                            |
| Visualizations        | `notes/assets/images/010-kuzmin-tmi-eval/`                                           |

## Additional Metrics Computed

Beyond the standard MSE/Pearson from training:

| Metric                           | Description                                                         |
|----------------------------------|---------------------------------------------------------------------|
| MAE                              | Mean absolute error                                                 |
| Spearman                         | Rank correlation                                                    |
| R²                               | Coefficient of determination                                        |
| Wasserstein                      | Distribution similarity                                             |
| JS/KL divergence                 | Distribution divergence (histogram-based)                           |
| Precision/Recall/F1 at threshold | Classification metrics for "strong" interactions (\|score\| > 0.08) |

## Related Investigation

During development of this script, we encountered metrics discrepancies that led to an extensive debugging investigation. The issue turned out to be a simple dataset path mismatch.

See: [[False Torchmetrics Bug Bc Wrong Dataset Path|experiments.010-kuzmin-tmi.false-torchmetrics-bug-bc-wrong-dataset-path]]

## Related Scripts

- [[equivariant_cell_graph_transformer|experiments.010-kuzmin-tmi.scripts.equivariant_cell_graph_transformer]] - Training script
- [[equivariant_cell_graph_transformer_inference_1|experiments.010-kuzmin-tmi.scripts.equivariant_cell_graph_transformer_inference_1]] - Inference on 37M triples
