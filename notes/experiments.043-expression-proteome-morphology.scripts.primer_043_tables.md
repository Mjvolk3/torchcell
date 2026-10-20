---
id: erh67ylrlpslslhtdwl4c0s
title: Primer_043_tables
desc: ''
updated: 1792445303674
created: 1792445303674
---

## 2026.10.19 - Tables of the 043 datasets primer

Writes the LaTeX tables of `notes-tex/multimodal/043-datasets-primer/tables/` from
`results/store_census.json`, `results/cross_dataset_correlation.json`,
`results/within_dataset_ceilings.json` and two 019 ceiling files on the
`multimodal-phenotype-retrospective` worktree (`expression_ceiling_all.json` for Caudal,
Messner, Zelezniak and Nadal-Ribelles; `morphology_noise_ceiling.json` for Ohya). The
sha256 of every input goes to `tables/provenance.json`. Media names are read from up to
3,000 records of each dev-tree LMDB. Run with `make tables` in the document directory or

```bash
python experiments/043-expression-proteome-morphology/scripts/primer_043_tables.py
```

## 2026.10.19 - Plan, tensor-shape and owner-decision tables

`t11-batches.tex` is replaced by `t11-plan.tex` (the sets of the owner's plan: training,
splits A, B and C, the Ohya probe, the dropped Sameith Sm, with record counts from
`results/data_shapes_043.json`). Two new row files feed landscape longtables:
`t12-shapes-rows.tex` (every dataset as tensors, from the full shape pass, slurm job 5902)
and `t13-decisions-rows.tex` (the owner-decision checklist, condensed in `DECISIONS` from
the decision lists of `sections/input-design.tex` and `sections/output-design.tex`, plus
two that the plan itself raises). The existing tables regenerate byte-identical.

## 2026.10.20 - Review pass: reconciled decisions, fold-0 plan, data per model part

- `DECISIONS` now holds 29 rows (I1 to I8, O1 to O10, P1, P2, S1 to S7, D1, E1) with the
  reconciled decisions of the 2026-10-20 review pass: option 1 is the main option and the
  proposed column names its alternatives. Numbers from this experiment's result files are
  `<<key>>` tokens filled by `facts()` (an unfilled token raises); numbers from other
  experiments (019, 030, 031, 035) are quoted with their runs.
- `plan_table` gives the expected fold-0 sizes of splits A to E under the global fold file
  (a tenth of each deletion dataset's genes after the 82 Sameith singles are pinned, a
  fifth of the 979 isolates by strain name, of each segregant panel and of the promoter
  variants); the fold file itself is not yet written.
- New `data_per_part` writes `t14-data-per-part.tex`: records, genotypes or reference
  contexts, gene-level values and distinct genes or alleles per model part, from
  `data_shapes_043.json`, `store_census.json` and `distinct_alleles.json`
  ([[experiments.043-expression-proteome-morphology.scripts.count_distinct_alleles]]).
- `ROLES` carries the split each dataset feeds.
