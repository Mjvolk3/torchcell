---
id: 0f4mnrah288b9o3dbhuyhj8
title: Nadal_pseudobulk_recompute
desc: ''
updated: 1789173834256
created: 1789173834256
---

## 2026.09.11 - Recomputing the Nadal-Ribelles fold change from the single-cell object

Script: `experiments/028-knockout-expression/scripts/nadal_pseudobulk_recompute.R`, run
with the `r-seurat` conda env (R 4.3, Seurat 5.3.0). Input `seus_split.RData` from
Zenodo 14062629 (md5 `65bb56efd8120f32f65c044de5f040aa`, sha256 `da99869c...`), added to
the raw mirror `$DATA_ROOT/torchcell-raw/nadalRibelles2025/` on 2026-09-11 together with
the paper's DE scripts (`DEGs_summary.R`, `summary.genotypes_Rev.R`, `Figures_Rev.R`,
`DEG.Rdata`) and the Kemmeren table the paper compared against
(`deleteome_all_mutants_controls.txt`); every md5 matches the Zenodo record.

The control object holds 326,438 cells x 6,951 genes (RNA `counts` and log-normalized
`data`), 3,506 genotype labels in `assignment_consensus2` (3,505 `bc-<ORF>` plus `WT`,
500 WT cells from 94 clones spread over all 14 batches), the deleted ORF in `kogene`.
Three statistics per (genotype, gene) against the pooled WT cells, written as gzipped
TSV to `$DATA_ROOT/data/torchcell/nadal_ribelles_perturbseq2025/recomputed/`:

- **A `pseudobulk_log2fc`**: summed UMI -> CPM -> log2((cpm_g + 1) / (cpm_wt + 1)), absent
  below 10 summed UMI (16.2M of 24.4M cells reported); sd 2.20.
- **B `seurat_avg_log2fc`**: Seurat's FindMarkers `avg_log2FC` (per-cell means, pseudocount
  1); sd 0.41.
- **C `scanpy_logfc`**: scanpy's `rank_genes_groups` formula on the same means; sd 11.6 with
  22.6% of cells at |C| >= 20, which reproduces the stored values' sentinel rate (stored
  20.2%) and matches their sentinel status cell by cell 98.1% of the time. Off the
  sentinels the two agree at r = 0.51 (median absolute difference 0.26), so the stored
  numbers are this statistic computed on a somewhat different cell set or normalization.

Also written: `pseudobulk_umi.tsv.gz` (raw summed UMI, genes x genotypes),
`genotype_by_batch.tsv`, `genotypes.tsv` (label, kogene, n_cells). The paper's own scripts
confirm the provenance of the stored numbers: `DEGs_summary.R` reads per-genotype CSVs
"from Jaren" (scanpy output with `names`, `logfoldchanges`, `pvals`), keeps `df[, 1:2]`,
and counts DEGs at `pvals < 0.05` and |logFC| >= 1; `Figures_Rev.R` (Supp Fig 1I) computes
a per-genotype Spearman against Kemmeren's M column (`javsma`) but plots only the counts
of significant genes.

Result: none of the three recomputed statistics recovers agreement with Kemmeren
([[experiments.028-knockout-expression.scripts.cross_study_recomputed]]); the reason is the
cell-to-genotype assignment
([[experiments.028-knockout-expression.scripts.nadal_assignment_purity]]).
