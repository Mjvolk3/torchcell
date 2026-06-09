---
id: 61e8za03ff4lhoip99yilqx
title: Equivariant_cell_graph_transformer_inference_2
desc: ''
updated: 1781028972223
created: 1781028972223
---

## 2026.06.09 - Score the Inference-2 Triples with the Trained Cell-Graph Transformer

This script exists to turn the inference_2 candidate triples into model-predicted gene-interaction scores, which are the actual deliverable the 010 workflow ranks to nominate constructible strains for the lab. It loads a trained CellGraphTransformer checkpoint and runs it over the InferenceDataset built from the strict-threshold triple set, applying the same normalization (and inverse transform) used in training so predictions land back in interpretable fitness units.

### Why these choices matter

- Uses the Perturbation graph processor (not SubgraphRepresentation) to match how the transformer consumes perturbed-gene inputs.
- Streams predictions to Parquet with dictionary-encoded gene columns to keep hundreds of millions of rows in a few GB instead of tens of GB of CSV.
- Includes adaptive batch-size backoff on CUDA OOM and per-index gene-name streaming from LMDB, since the dataset is too large to hold gene metadata in memory.
