---
id: u33kebp6m36nj3wwjlcnaqn
title: Kuzmin2020_per_replicate_review
desc: ''
updated: 1795581303990
created: 1795581303990
---

## 2026.11.24 - Kuzmin 2020 per-replicate review and the HO genotype fix

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/kuzmin2020_per_replicate_review.py`. Results: `experiments/036-dataset-fixes-before-kg-build/results/kuzmin2020_per_replicate_review.json`. Run on GilaHyper with `--before-root $DATA_ROOT/data/torchcell/smf_kuzmin2020` (the dev store, read only) and `--after-root $DATA_ROOT/tmp/scratch/per-replicate-kuzmin2020/smf_kuzmin2020` (the #1202 build).

It reads Tables S1, S3 and S5 (sha256 `4180577d...`, `625420be...`, `63240539...`), Kuzmin 2018 Data S1 (`c82b60f9...`) and the Costanzo 2016 single-mutant fitness standard (`3b9d3351...`), and reports four things.

- **Table S5 single mutants (#1202).** 480 rows, HO (YDL227C) in ORF1 on 240 and in ORF2 on 240, never on both or neither side. 472 state a fitness: 238 with HO in ORF1, 234 with HO in ORF2, 472 distinct genes.
- **`Array single mutant fitness` of Tables S1 and S3 (#1176 lead).** Constant per array strain in each table (S1 1,172 strains, S3 4,540, 1,159 in both, 1,146 of those differ by more than 1e-3). S3's value equals the Costanzo 2016 26 degree standard on all 4,510 strains it gives a value for (max |difference| 0.0; 29 S3 strains are absent from it, 1 is blank); S1's value matches neither Kuzmin 2018 Data S1 (1,158 of 1,170 differ, r 0.872) nor the Costanzo 2016 standard at 26 degrees (1,157 of 1,170 differ, r 0.877).
- **Combined-mutant colony count back-solve** (the method of `test_kuzmin_n_backsolve.py`, digenic rows): best n = 6 on Table S1 (537,885 rows; median-p gap 0.0385 at n = 6, 0.0576 at n = 4) and n = 4 on Table S3 (94,864 rows; gap 0.003). The loader keeps n = 4.
- **Smf store before vs after the fix.** 472 records on both sides; 238 genotypes changed and every one was an HO deletion before; max |fitness difference| 0.0; the only other differing field is `replicate_id` (absent in the older store, `None` now) on all 472 experiments and references.
