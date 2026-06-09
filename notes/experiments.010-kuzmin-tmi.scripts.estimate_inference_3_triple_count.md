---
id: 3e09fkxmsraaqfbgf37ofik
title: Estimate_inference_3_triple_count
desc: ''
updated: 1781028986022
created: 1781028986022
---

## 2026.06.09 - Scope the Inference-3 Compute Budget Before Committing to Generation

This script exists to answer "how many triples will the relaxed inference_3 thresholds actually produce?" before paying the cost of the full generation-and-scoring pipeline. By estimating candidate counts under both the strict inference_2 rules and the relaxed inference_3 rules from the Costanzo2016 data alone, it lets the decision about whether the relaxed thresholds yield a tractable yet statistically sufficient set be made up front, rather than discovered after a multi-hour run.

### Why these choices matter

- Reports a side-by-side count for inference_2 vs inference_3 and the inflation ratio, making the compute trade-off of relaxing thresholds explicit.
- Prints a threshold sweep over SMF and DMF cutoffs so the chosen values can be sanity-checked against how many genes and pairs survive at each level.
- Uses only Costanzo2016 (not the full multi-dataset DMF union) and counts rather than writes triples, keeping it a fast scoping tool distinct from the actual generators.
