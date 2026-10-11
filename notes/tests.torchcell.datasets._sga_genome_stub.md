---
id: 3tnqcgswdi6uvpmbltxvxzy
title: _sga_genome_stub
desc: ''
updated: 1791701425306
created: 1791701425306
---

## 2026.10.11 - A genome on which every SGA pin and refusal holds

Test helper for the Costanzo 2016 and Kuzmin 2020 loader tests. The loaders re-check all 16 rename pins and 35 refusals of [[torchcell.datasets.systematic_name_currency]] against the injected genome before reading a row, so `SgaGenomeStub(live=...)` derives a genome from those tables (each renamed name an alias of its ORF alone, each merged name resolving to its locus, the four non-gene loci `YER109C`, `YFL056C`, `YIL167W`, `YIR043C` resolving to themselves as `non_gene_feature`) plus the fixture's live genes; every other name resolves as retired.
