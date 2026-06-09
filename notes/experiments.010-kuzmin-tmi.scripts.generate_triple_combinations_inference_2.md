---
id: z89z06vjhg9y86myfidp08w
title: Generate_triple_combinations_inference_2
desc: ''
updated: 1781028992936
created: 1781028992936
---

## 2026.06.09 - Build the Inference-2 Candidate Triple Set Under the Strict Monotonic-Improvement Hypothesis

This script exists to construct the pool of three-gene combinations that the 010 inference workflow will score, scoped to the experiment's central scientific claim that fitness improves at every step of construction (f_WT < f_i < f_ij < f_ijk). It encodes that hypothesis as concrete data filters so that downstream model scoring only spends compute on triples whose measured single- and double-mutant fitness already exhibit the strict iterative-improvement pattern, and it removes any triple already present in the training data (TMI) so that what survives is genuinely new and constructible in the lab.

### Why these choices matter

- Uses Costanzo2016 as the single-mutant fitness source specifically because it is the lowest-noise measurement (sigma = 0.063), keeping the strict thresholds defensible.
- Strict thresholds: all singles > 1.0 with max > 1.10, and all doubles > 1.0 with max beating the best single by at least 0.03 (a gap, not a fixed cutoff) -- this is the demanding variant that inference_3 later relaxes.
- Excludes SGD essential genes up front, since a lethal deletion cannot participate in a viable constructible strain.
- Streams output to Parquet and offers a `--plot-only` path that redraws the filtering diagnostic from a saved `generation_summary.txt`, avoiding a multi-hour regeneration just to iterate on the figure.
