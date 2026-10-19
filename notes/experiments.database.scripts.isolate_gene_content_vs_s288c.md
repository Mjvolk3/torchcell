---
id: 5hjw2jnzb2tpod0e87cwqpr
title: Isolate_gene_content_vs_s288c
desc: ''
updated: 1792388996309
created: 1792388996309
---

## 2026.10.19 - How far an isolate's gene content sits from S288C

Reads the built Caudal 2024 and Muenzner 2024 stores through their dataset classes and
counts, per isolate, the perturbations the loader stored against S288C: accessory ORFs
present that the reference lacks, reference ORFs absent, sequence variants and natural
loss-of-function alleles. Presence and absence are reference membership (whether the
pangenome column maps to an S288C name), not population frequency, as
[[torchcell.datasets.scerevisiae.caudal2024]] records. Output:
`experiments/database/results/isolate_gene_content_vs_s288c.json` (gitignored results tree).

Measured 2026-10-19:

| store | isolates | accessory present, median (max) | reference absent, median (max) | variants, median | LoF, median | distinct accessory | distinct absent |
|---|---|---|---|---|---|---|---|
| CaudalPanTranscriptome2024Dataset | 943 | 78 (352) | 126 (453) | 5,069 | 362 | 1,614 | 1,118 |
| ProteomeMuenzner2024Dataset | 795 | 78 (352) | 125 (420) | 5,027 | 359 | 1,643 | 1,112 |

So a typical isolate is about 50 ORFs smaller than BY4741's 6,607 and differs from it in
presence at about 200 ORFs; the larger difference is the roughly 5,000 variants inside the
shared genes. The figures are quoted in the state section of
`notes-tex/database/database-expansion-200` ([[experiments.database.expansion-200]]).
