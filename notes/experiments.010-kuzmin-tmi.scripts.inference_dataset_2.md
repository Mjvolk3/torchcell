---
id: 7qaaz7n43rz3800jrmjeejk
title: Inference_dataset_2
desc: ''
updated: 1781028851803
created: 1781028851803
---

## 2026.06.09 - Materialize the inference-2 triple panel as a model-scorable LMDB

This script turns the stricter inference-2 candidate triples (filtered on SMF > 1.10 and DMF > max(SMF) + 0.03) from their parquet list into a graph-model-ready LMDB dataset, so the trained 010-kuzmin-tmi interaction model can score every candidate triple deletion at inference time. It exists to bridge raw combinatorial gene-triple selections and the model's expected experiment-graph input, defining lightweight inference-only experiment, reference, and phenotype classes so that triples without measured fitness values can still flow through the same datamodel and graph-processing path used in training.

### Specifics worth keeping

- Consumes `DATA_ROOT/data/torchcell/experiments/010-kuzmin-tmi/inference_2/raw/triple_combinations_list.parquet` and writes to the sibling `processed/lmdb/`, the canonical input read by the inference-2 scoring run.
- Defines `InferencePhenotype`/`InferenceExperiment`/`InferenceExperimentReference` so fitness can be absent (the quantity to be predicted), with a unit-fitness reference for relative interaction scoring.
- Parallelizes parquet-to-LMDB serialization with a `ProcessPoolExecutor` over mega-batches to handle the large triple counts without exhausting memory.
- `InferenceDataset` here is the shared base reused by inference_dataset_3.
