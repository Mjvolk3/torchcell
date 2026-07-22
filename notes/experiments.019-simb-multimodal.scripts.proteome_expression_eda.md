---
id: weoyek6ixiqtuy8bsr45lxr
title: Proteome_expression_eda
desc: ''
updated: 1784758735624
created: 1784758735624
---

Companion note for
`experiments/019-simb-multimodal/scripts/proteome_expression_eda.py`.

Proteome ↔ expression EDA: pulls `ProteomeMessner2023Dataset` (SM) and
`MicroarrayKemmeren2014Dataset` (SC) from the served DB, aligns the two YKO
single-deletion collections strain-by-strain and gene-by-gene, then reports the
overlap census, per-gene / per-strain mRNA↔protein correlations, and a ridge
linear map (held-out R²) in both directions, plus three figures.

Findings + media-confound verdict live in the analysis note:
[[experiments.019-simb-multimodal.proteome-expression-eda]].
