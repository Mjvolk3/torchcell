---
id: 2sa9mr51tavtuhmp84dngw1
title: Protein_fold_change_refusals_kang_lim
desc: ''
updated: 1791531982398
created: 1791531982398
---

## 2026.10.09 - Why two of #770's five papers stay refused

`ProteinFoldChangePhenotype` landed in the #770 / #753 wave. Of the five papers the P.
putida supplementary-data audit found releasing a protein-level fold change, three are
addressed by that class and two are not, for reasons on other axes. This script measures
both so the refusals are bytes, not readings.

Output of the committed run (repo root, `DATA_ROOT=/scratch/projects/torchcell-scratch`):

```
== Kang 2026 Supplementary Table S6 ==
si/si1.docx sha256 c7d4567fae037c7392e39b855cc71f83b7b4a5d96699c3f18b991a25c57f09e0
caption: Table S6. List of top 20 accessory genes upregulated and downregulated by sgRNA targeting PP_4854.
columns (10): ['Protein Group', 'Protein Names', 'Protein', 'Protein Description', 'Fold Change', 'Log2 (Fold Change)', 'p-Value (Equal Variance)', '(-Log10 (p-Value))', 'Category', 'Rank']
rows including the header: 41; data rows: 40
distinct keys in 'Protein Group': 40
first three keys: ['Q88HX1', 'Q88DG9', 'Q88C64']
rows with p < 0.05: 29 of 40
UniProtKB db_xrefs, pinned P. putida KT2440: 0
UniProtKB db_xrefs, pinned E. coli MG1655:   4281
```

Zero UniProtKB cross-references on the pinned *P. putida* assembly is what refuses Kang's
table: the new `uniprot_db_xref` identifier route reads exactly that annotation entry, and
the 4,281 on MG1655 is the comparison that shows the route works where the annotation
carries it (Gupta 2024 resolves 3,225 of 3,262 proteins through it). Note the count is
per GenBank flat file: globbing both the GCA and GCF copies of the MG1655 assembly
double-counts to 8,562, which is why the script names one file.

Lim 2025's seven proteome sheets each carry one `log2_Fold_change_A/B` column whose two
`log2_mean_*` arms always include an evolved isolate (`A10F63I1` or `A12F53I1`), so every
released contrast needs a genotype that #731 says cannot be written. The direction also
flips between the two sheet families: arm A is the evolved isolate in the four
`Proteome_*` sheets and the parent in the three `IPL400vs*` sheets.

Full reasoning, with the per-sheet row counts and arms, in
[[torchcell.datasets.pputida.kang2026]] and [[torchcell.datasets.pputida.lim2025]].
