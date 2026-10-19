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
