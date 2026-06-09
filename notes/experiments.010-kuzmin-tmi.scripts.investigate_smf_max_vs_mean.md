---
id: 7hpsjocqt6viy4igl421zff
title: Investigate_smf_max_vs_mean
desc: ''
updated: 1781029048211
created: 1781029048211
---

## 2026.06.09 - Auditing Why Low-Fitness Genes Slipped Through the Inference-3 Filters

This investigation exists to explain a trust-breaking surprise: YJR060W (mean single-mutant fitness ~0.59) survived the SMF_BASELINE = 0.90 filter in `generate_triple_combinations_inference_3.py`, which should have excluded it. The script's reason for being is to pin down whether the filter aggregated replicates with `max()` rather than `mean()` -- a subtle bug that would silently admit sick genes into the triple panel and contaminate downstream interaction predictions. It produces a per-gene max-vs-mean comparison so the team can decide whether to re-run generation with a corrected aggregation.

### Specifics worth keeping

- Audits all 12 panel genes plus a list of eight problematic doubles (mean DMF < 0.90) and confirms whether YJR060W actually appears in `triple_combinations_list.parquet`.
- Saves `results/inference_3/smf_max_vs_mean_panel12.csv` (the max-vs-mean summary table).
- Decision rule the script frames: if max-replicate SMF > 0.90, the `max()` aggregation is the root cause and generation should be re-run with `mean()`; if not, the predictions came from a different triples list than the current script would produce (a pipeline mismatch).
