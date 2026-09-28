---
id: vdfoyhgx8xwxjqwayrgi92p
title: Notes_tex_tables
desc: ''
updated: 1790379803722
created: 1790379803722
---

## 2026.09.25 - Every table of the 031 notes-tex document from the result CSVs

Writes `notes-tex/031-inhibitor-tolerance-data/tables/t1-axes.tex` through
`t8-chemsim.tex` as booktabs fragments with a GENERATED + SOURCE header, reading
`results/axes_table.csv`, the overlap CSVs, `shared_compounds*.csv`, `structure_*.csv`,
`noise_summary.csv`, `condition_noise_vanacloig2022.csv`, `top_matches_hom.csv`, and
when present `embedding_coverage.csv` and `chemical_similarity_summary.csv`. Run through
`make tables` in the document directory. Nothing in `tables/` is edited by hand.
