---
id: ndzz7cvq102xb6bj4bxdvig
title: Make_global_split_043
desc: ''
updated: 1792634281948
created: 1792634281948
---

## 2026.10.21 - The fold file v1

`experiments/043-expression-proteome-morphology/scripts/make_global_split_043.py` writes
`experiments/043-expression-proteome-morphology/conf/splits/global_split_v1.json`, validated by
the pydantic record `GlobalSplit043` in
`experiments/043-expression-proteome-morphology/scripts/global_split_043.py` (which also answers
`role(dataset, key)` and refuses a store whose raw index hash differs, `check_store`). Slurm
wrapper `scripts/gh_make_global_split_043.slurm`; the run is job 6046 (4 min 3 s, 16 reader
processes; job 6045 failed on the one Muenzner 2024 record with an empty genotype, the FY4
reference strain, now keyed `reference:FY4`). A summary with every count is
`results/global_split_v1_summary.json`.

What the draw does, with `numpy.random.default_rng(0)` consumed in a fixed order:

- **Genes**: every gene deleted in Kemmeren 2014, Hu 2007, Jackson 2020, Nadal-Ribelles 2025,
  Messner 2023, Zelezniak 2018 or Ohya 2005 (4,908 genes, all inside the 6,607 index after the
  dataset module's name resolution). Stratum = dataset-membership pattern (38 patterns); each
  stratum is permuted and dealt round-robin into 10 folds, the deal continuing across strata.
  83 pinned genes (the 82 Sameith 2015 singles, and HO `YDL227C`, the Jackson reference) are
  dealt into folds 2 to 9 only.
- **Isolates**: 979 strain names (Caudal 2024, Muenzner 2024 protein and turnover), stratum =
  membership pattern (Caudal, Muenzner protein, turnover): 704 in Caudal and protein, 184 Caudal
  only, 55 in all three, 36 protein only; five folds.
- **Segregants**: 82,467 records keyed `f"{dataset}:{cross}:{segregant_id}"`, dealt within
  (dataset, cross, panel) with panel the medium, so Boocock's pooled SM panel (393) and its de
  novo cells (79,123) are separate strata; five folds.
- **Promoters**: 71,158 distinct Vaishnav 2022 80-mers (every record has its own); five folds.

Realized sizes against the primer's expected fold-0 shares (`tables/t11-plan.tex`):

| dataset | unit | expected test units (fold 0) | realized test units (fold 0) | realized validation units (fold 1) | realized training units |
|---|---|---|---|---|---|
| Kemmeren 2014 | gene | about 140 | 140 | 140 | 1,202 |
| Messner 2023 | gene | about 448 | 441 (457 records) | 441 | 3,594 |
| Nadal-Ribelles 2025 | gene | about 306 | 302 (602 records) | 301 | 2,456 |
| Hu 2007 | gene | about 26 | 19 | 19 | 224 |
| Zelezniak 2018 | gene | about 10 | 9 | 9 | 79 |
| Jackson 2020 | gene | (not given) | 0 | 0 | 12 |
| Caudal 2024 | isolate | (part of about 196) | 188 | 189 | 566 |
| Muenzner 2024 protein | isolate | about 159 | 160 | 159 | 476 |
| Muenzner 2024 turnover | isolate | about 11 | 11 | 11 | 33 |
| all isolates | isolate | about 196 | 196 | 196 | 587 |
| Boocock 2025 | segregant record | about 15,903 | 15,904 | 15,903 | 47,709 |
| N'Guessan 2025 | segregant record | about 590 | 590 | 591 | 1,770 |
| Vaishnav 2022 | distinct 80-mer | about 14,232 | 14,232 | 14,232 | 42,694 |
| Sameith 2015 Dm | double | 72 (split B) | 72 | n/a | n/a |
| Ohya 2005 | genotype | probe | 4,678 probe | n/a | n/a |
| Sameith 2015 Sm | single | excluded | 82 excluded | n/a | n/a |

Hu realizes 19 test genes against the expected 26 (262 records over a tenth by stratum; the
expected share was a plain tenth). **Jackson 2020 has no gene in fold 0 or 1**: of its 12
deletion genes, 6 are pinned to training (HO and five TFs that are Sameith 2015 singles:
YJL110C, YKR034W, YIR023W, YFL021W, YER040W), and the 6 free ones (YNL314W, YBL103C,
YOL067C, YDR463W, YEL009C, YHR006W; each sits in a small membership stratum, where the
continuing deal lands one or two genes) fell in folds 2 to 9, so Jackson can be scored on
split A by no arm under v1. A v2 that wants a Jackson read would have to force at least one
free Jackson gene into fold 0.
