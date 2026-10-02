---
id: oqdeojhvb7omr5p6scywxsz
title: Make_normalization_stats_030
desc: ''
updated: 1790417022329
created: 1790417022329
---

## 2026.09.26 - Constants fitted on the training ENTRY rows

On 030 `label_df` holds one arbitrary entry per record (the last written), so the population the per-entry loss trains on is the entry table of `closure_recompute_030.py` (`closure/entries.parquet`) restricted to the arm's training records. First run (2026-09-26, arm `cgt_030_s3_r_tok_embfit_001`): 1,045,618 training records; gene_interaction n 2,339,105 mean -0.003434 sd 0.050581; fitness n 2,360,393 mean 0.885911 sd 0.157893. Written to `results/normalization_stats_030_s3_r_ess.json` with `train_index_sha256`, which the training script checks against the arm it is about to train. Compare 025: gene_interaction sd 0.0444 over the whole build and 0.0633 over the triples alone.
