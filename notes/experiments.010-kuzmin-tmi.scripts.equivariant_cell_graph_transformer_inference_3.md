---
id: 4dt5nlu9epi1zqn79gxd5ee
title: Equivariant_cell_graph_transformer_inference_3
desc: ''
updated: 1781028979118
created: 1781028979118
---

## 2026.06.09 - Score the Inference-3 Triples Across Multiple GPUs for the Powered Improvement Test

This script exists to produce model-predicted interaction scores over the much larger inference_3 candidate set, sized so the downstream Jonckheere-Terpstra test of monotonic fitness improvement has the statistical power the relaxed thresholds were chosen to provide. It is the inference_2 scoring script scaled up for multi-GPU execution, because the relaxed-threshold triple pool is large enough that single-GPU scoring would be impractical.

### Why these choices matter

- Adds torchrun-based distributed inference: each rank scores a contiguous 1/N block, writes its own Parquet shard, and rank 0 merges the shards into a single index-sorted output (also runs single-GPU with no torchrun).
- Casts gene columns to large_string during the merge to avoid the 2GB Arrow offset overflow that arises when consolidating roughly 465M rows of gene names.
- Provides a `--merge-only` entry point so an interrupted run's existing shards can be merged without re-acquiring GPUs.
- Otherwise mirrors inference_2: same checkpoint-driven CellGraphTransformer, Perturbation processor, training-matched transforms, and streaming Parquet output.
