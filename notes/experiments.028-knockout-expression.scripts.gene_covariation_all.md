---
id: jnxms9slx75w137xpbttw64
title: Gene_covariation_all
desc: ''
updated: 1789268789438
created: 1789268789438
---

## 2026.09.12 - Gene-pair co-variation in every panel, one ordering

Author's objection to the proteome figure: a natural-isolate panel (Caudal) set beside a
single-deletion panel (Messner) cannot be compared strain by strain. It is not; the
comparison is at the gene-pair level, and this script makes that explicit for every panel
at once. Within one panel two genes get one Pearson across that panel's own strains, so a
panel becomes one gene x gene matrix on a common gene list; two panels are compared by the
Spearman between the upper triangles. Nothing is aligned across strains.

Panels: Messner (4,549 deletions, protein), Caudal (943 isolates), Kemmeren (1,484
deletions), Sameith (82 single deletions), Nadal A (2,243 deletions with >= 50 cells). Two
gene lists: the 1,645 proteins measured in all five (each in >= 80% of a panel's strains),
and the 4,337 genes measured in the four mRNA panels.

Spearman between matrices, protein gene list:

| | Caudal | Kemmeren | Sameith | Nadal A |
|---|---|---|---|---|
| Messner | 0.34 | 0.30 | 0.24 | 0.05 |
| Caudal | | 0.47 | 0.37 | 0.08 |
| Kemmeren | | | 0.68 | 0.10 |
| Sameith | | | | 0.03 |

Expression gene list (4,337 genes): Kemmeren-Sameith 0.64, Caudal-Kemmeren 0.39,
Caudal-Sameith 0.30, Nadal A -0.02 to 0.09.

How much co-variation each panel holds (protein list; sd of the gene-pair r, fraction
beyond |r| 0.3): Messner 0.16 / 7%, Caudal 0.26 / 28%, Kemmeren 0.17 / 16%, Sameith
0.28 / 28%, Nadal A 0.09 / 4%. The Nadal matrix is nearly flat.

Every matrix is drawn in the ordering from average-linkage clustering of Kemmeren's matrix
on that gene list, so a block in Kemmeren can be looked for at the same place in every
other panel: Kemmeren's two large blocks appear in Caudal, in Sameith and, fainter, in
Messner; nothing appears in Nadal A. The ordering favors Kemmeren visually; the Spearman
values do not depend on it.

![](./assets/images/028-knockout-expression/gene_covariation_all.svg)

Results: `experiments/028-knockout-expression/results/gene_covariation_all.json`. In the
document as `fig:gene-covariation-all` under `sec:readouts-proteome`. Related:
[[experiments.028-knockout-expression.scripts.proteome_expression_covariation]],
[[experiments.028-knockout-expression.scripts.cross_study_structure]].
