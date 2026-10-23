---
id: prvnu19zw9uw2my4yvi8s8v
title: Training_design_043_tables
desc: ''
updated: 1792755758586
created: 1792755758586
---

## 2026.10.23 - Tables of the 043 training-design review

Writes the generated tables of `notes-tex/multimodal/043-training-design/tables/` from result
files only, with the sha256 of every input in `tables/provenance.json`:

- `t1-measured-rows.tex`: registered-window means of the overnight seed-0 runs
  (`results/overnight_2026_10_22_window_means.json`) against the linear ladder
  (`results/linear_evals_043.json`) on fold 1 and fold 0, and the Kemmeren transfer.
- `t2-batch-share-rows.tex`: training records per dataset (prepare job 6134 `meta.json`) and
  the exact expected weight of each dataset in the batch objective, by enumerating the 12,870
  compositions of a batch of 8 (multivariate hypergeometric for the records sampler, equal
  multinomial for the balanced sampler).
- `t3-runs-rows.tex`: job, W&B run link, arm, epochs and window of each overnight run.
- `macros.tex`: the numbers the document's prose quotes.

```bash
python experiments/043-expression-proteome-morphology/scripts/training_design_043_tables.py
```
