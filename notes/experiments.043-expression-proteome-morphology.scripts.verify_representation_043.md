---
id: cfww8e1qw1f532hnf1fd5gu
title: Verify_representation_043
desc: ''
updated: 1792638798223
created: 1792638798223
---

## 2026.10.21 - Store-wide verification of the input representation

Script: `experiments/043-expression-proteome-morphology/scripts/verify_representation_043.py`; launcher [[experiments.043-expression-proteome-morphology.scripts.gh_verify_representation_043]].

Every record goes through `Raw043Dataset.__getitem__` with `representation=RepresentationConfig(route="A")`, which is the path training reads. Isolate records are resolved again under route B. Keys only. Outputs: `results/verify_representation_043.json` (per dataset and genotype class: complete records, records and genes per typed gap, mark / operator / extra-token distributions, position gaps, route A against B, the segregant rule outcomes), `results/verify_representation_043.md` (the table), and `$DATA_ROOT/data/torchcell/experiments/043-expression-proteome-morphology/representation_043/verify_records.parquet` (one row per record).

Runs:

- Job 6119 (commit `a10fa7814`, 4 min 0 s): it found the one Muenzner 2024 proteome record with an empty perturbation list (`data_19795`, the panel's S288C row) classed as a zero-mark deletion. Fixed in `b87697a81` (class `reference`).
- Job 6120 (commit `b87697a81`, 3 min 37 s): the reported run. 173,617 records, 89,358 complete, segregant cross-check 0 mismatches over 82,467 records. Table and readings in [[experiments.043-expression-proteome-morphology]] (section 2026.10.21 - Input representation complete).

Memory: the cgroup read 11 GB anonymous memory during 6119; the rest of its 64 GB was LMDB page cache, and there was no OOM event. `sacct` MaxRSS (128 GB for 6120) sums the forked workers' shared pages and overstates the peak.
