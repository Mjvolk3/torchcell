---
id: 4elyeytb0rnqekkunmzo0tk
title: Parsons2006_release_checks
desc: ''
updated: 1791747634694
created: 1791747634694
---

## 2026.10.11 - The release measurements the Parsons 2006 loader cites

Reads Tables S4 and S7 from the raw mirror through the loader's parsers, and Tables S5 and S9 plus the SGD essentiality gene set read-only; writes `experiments/database/results/parsons2006_release_checks.json`. Cited by [[torchcell.datasets.scerevisiae.parsons2006]].

| measurement | value |
|---|---|
| Table S7 strains, ratios, blanks | 4,111; 329,254; 7,848 |
| cells with Table S4 -log10 p > 5, positive ratio among them | 5,550; 5,521 (99.5%) |
| strains with one Table S5 non-functional tag | 180 |
| value SD, lone-tag strains vs the rest (ratio) | 0.530 vs 0.471 (1.13; halving would be below 0.707) |
| Table S9 multidrug-resistant strains, in Table S7 | 174; 166 |
| Table S7 ORFs in the SGD essentiality gene set | 24 |
