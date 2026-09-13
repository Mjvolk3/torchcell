---
id: xlitld8mbhhxd17ixi7dptr
title: Proteome_replication_zelezniak
desc: ''
updated: 1789279182076
created: 1789279182076
---

## 2026.09.13 - A second knockout proteome against Messner and against Kemmeren

Zelezniak 2018 (Ralser lab, SWATH, SM medium, 726 proteins, 97 kinase deletions, three
replicates against a 12-replicate wild type; stored values are the SVA-adjusted log
signal). Messner 2023 (same lab, one measurement per strain over the HIS3 reference)
covers 89 of the 97, Kemmeren 94, all three 87. Values are log ratios over each study's
own wild type, z-scored per gene across the shared strains.

| comparison | shared deletions | per deletion, median r | per gene, median r |
|---|---|---|---|
| Zelezniak vs Messner, protein vs protein | 89 | 0.08 | 0.09 (7% above 0.3) |
| Zelezniak protein vs Kemmeren mRNA | 94 | 0.01 | 0.03 |
| Messner protein vs Kemmeren mRNA, same 87 | 87 | 0.01 | 0.02 |

Gene co-variation on the 711 shared proteins across the 87 strains: Zelezniak vs Messner
0.15, Zelezniak vs Kemmeren 0.07, Messner vs Kemmeren 0.22; Messner's own 87-strain matrix
against its full 4,549-strain matrix 0.80, so 87 strains do reproduce a panel's own
structure, and the 0.15 is between-study disagreement, not strain count. Abundance level
(the two wild types) Spearman 0.77 over 714 proteins.

Reading: the two proteomes agree on how much each protein is made (0.77) and barely on
what a kinase deletion does to it (0.08 per deletion). Hypothesis, untested: Zelezniak's
SVA adjustment removed variation shared with Messner, or most kinase deletions move the
proteome too little for either study to measure the same small change; the right tail of
a (deletions at r 0.3 to 0.65) would be the responsive ones. Either way a knockout
proteome has not been shown to replicate itself here any better than the transcriptome
replicates across platforms (Kemmeren vs Sameith, same lab and platform, is 0.74).

![](./assets/images/028-knockout-expression/proteome_replication_zelezniak.svg)

Results: `experiments/028-knockout-expression/results/proteome_replication_zelezniak.json`.
Related: [[experiments.028-knockout-expression.scripts.proteome_expression_covariation]],
[[experiments.028-knockout-expression.scripts.gene_covariation_all]].
