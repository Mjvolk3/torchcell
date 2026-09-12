---
id: 6znt68b1ax45kngjsht7ip4
title: Nadal_paper_deleteome_comparison
desc: ''
updated: 1789190272348
created: 1789190272348
---

## 2026.09.12 - The paper's own comparison to the deleteome, rerun

Script: `experiments/028-knockout-expression/scripts/nadal_paper_deleteome_comparison.R`.
Inputs are the paper's released objects in the raw mirror (`Figures_Rev.R`, `DEG.Rdata`,
`FC_genotype.Rdata`, `deleteome_all_mutants_controls.txt`, all md5-verified against
Zenodo 14062629). The code path is the paper's "Supp Fig 1 I-J" block, line for line,
with the column selection made explicit (the paper's `grepl(make.names(j), ...)` is a
regex in which `.` matches the `-` and space of the raw column names).

What the paper computes per genotype with more than 5 cells (874 genotypes shared with
the deleteome): `javsma`, the Spearman between the single-cell scanpy logFC and the
microarray M over the genes present in both; `MAsig`, the microarray genes at |M| > 1
and p < 0.05; `JAnsig`, the single-cell DEG count (p < 0.05, |logFC| >= 1). It plots
only log(MAsig) against log(JAnsig) (Supplementary Fig. 1i) and the text calls that "a
consistent correlation" with no coefficient.

- Supplementary Fig. 1i reproduced: Spearman 0.234 over 874 genotypes.
- `javsma`, computed by the paper's script and not shown: median 0.013, IQR -0.006 to
  0.037, 1 of 874 above 0.2. Sentinel genes removed: median unchanged at the third
  decimal. This is the same number experiment 028 measures from the loader's records.
- The single-cell DEG count is a cell-count artifact: Spearman -0.67 with cells per
  genotype (fewer cells, more genes with a zero mean in the genotype, more |logFC| = 23
  sentinels that pass |logFC| >= 1). The microarray count does not track cells (-0.065).
- Kemmeren's responsive call reproduced from the deleteome table with the 58 WT-variable
  genes and YDL196W excluded: 772 of 1,487 profiles (51.9%) have >= 4 genes at FC > 1.7
  and p < 0.05; the paper reports 53% nonresponsive. Without the exclusion the count is
  994, so the WT-variable genes alone flip 222 mutants to responsive.

Outputs under `$DATA_ROOT/data/torchcell/nadal_ribelles_perturbseq2025/recomputed/`:
`paper_deleteome_comparison.tsv`, `kemmeren_responsive.tsv`,
`deleteome_extra_profiles.tsv` (the wt-ypd vs wt, wt-matA vs wt and wt-by4743 vs wt
profiles, M and p). Drawn by
[[experiments.028-knockout-expression.scripts.nadal_replication_coverage]].
