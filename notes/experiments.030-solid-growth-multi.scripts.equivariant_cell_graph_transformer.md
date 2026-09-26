---
id: vsdb25c0lchba8aeberhlht
title: Equivariant_cell_graph_transformer
desc: ''
updated: 1790417007621
created: 1790417007621
---

## 2026.09.26 - The 025 script on the 030 build, one row per entry

Ported from `experiments/025-solid-growth/scripts/equivariant_cell_graph_transformer.py`. What changed: the arm comes from [[experiments.030-solid-growth-multi.scripts.arm_030]]; the processor emits the source-dataset index beside every value; the model gets `dataset_token` (readout mode, dim 8, `vocab_size` = the vocabulary length); the task runs `per_entry=True` with `essentiality_eval` (the 698 held-out singles read under `SmfCostanzo2016Dataset` and `GeneEssentialitySgdDataset`, logged as `val_ess/auroc_{released,matched}_{smf,sgd}`); the normalizer reads the committed constants (`transforms.fit_stats`) and the script refuses a file whose training-set fingerprint differs from the arm's; `follow_batch` always carries `phenotype_values`; `trainer.limit_train_batches` / `limit_val_batches` exist for the smoke; under `smoke.enabled` the `SyntheticTokenOffset` transform runs after the normalizer and, after `fit`, `run_smoke_check` writes `results/smoke/<config>_seed<seed>.json`.

Validation and test log the 025-comparable number: one value per pinned triple under its own screen's token (`val/gene_interaction/Pearson`, the checkpoint monitor), beside the per-entry, per-token and cross-token Pearson.

Configs: `cgt_030_s3_r_tok_fit_000` (table), `cgt_030_s3_r_tok_embfit_001` (the first arm, composite + fitness, three seeds on mmli), `cgt_030_smoke_tok_000` / `cgt_030_smoke_ctl_000`.
