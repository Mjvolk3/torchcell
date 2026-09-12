---
id: nhf9cs06jbfwbp2bxawcwjq
title: Nadal_batch_replication
desc: ''
updated: 1789190265039
created: 1789190265039
---

## 2026.09.12 - Within-study replication of a Nadal-Ribelles genotype profile

Script: `experiments/028-knockout-expression/scripts/nadal_batch_replication.R`
(R + Seurat env `~/miniconda3/envs/r-seurat`, 12 min on the 5.9 GB `seus_split.RData`).
The question: Kemmeren against Sameith gives the microarray platform a test-retest of
0.74 per strain; the single-cell panel had none, and without it a cross-study zero could
not be read. Control condition, raw UMI, per-genotype pseudobulk against the WT cells of
the SAME cartridge batch (13 of 14 batches have >= 15 WT cells).

- Cross-batch, same genotype (>= 15 cells in each of two batches, 17,554 pairs): median
  r 0.043; different genotypes across the same two batches, cell counts matched: 0.022.
  1.8% of same-genotype pairs above 0.2 (null 0.2%). The r does not rise with cells
  (Spearman 0.02; median 0.042 at 15 to 30 cells, 0.051 above 120).
- Same pairs against the POOLED WT reference: 0.113 for same and 0.111 for different
  genotypes. The batch effect is real (WT batch against WT batch: sd of the log2 FC
  0.75, 16% of genes beyond 2-fold) and it is what a pooled reference puts into every
  profile; a per-batch reference removes it.
- Split-half within one batch, WT split too (786 genotype-batches with >= 30 cells):
  median r 0.080; the other-genotype null 0.051. A first version compared both halves to
  the SAME WT cells and read 0.37: the shared reference's sampling noise, not signal.
  Do not repeat.
- Depth: median 934 UMI and 492 genes per cell over 326,438 control cells.

Reading: at 15 to 100 cells a Nadal-Ribelles genotype pseudobulk does not reproduce
itself, by 0.02 to 0.03 above a different genotype. The cross-study zero against
Kemmeren is what an r of 0.04 within the study predicts. Drawn by
[[experiments.028-knockout-expression.scripts.nadal_replication_coverage]]; the
per-genotype cross-batch r feeds
[[experiments.028-knockout-expression.scripts.cross_study_structure]].

Outputs under `$DATA_ROOT/data/torchcell/nadal_ribelles_perturbseq2025/recomputed/`:
`batch_replication_pairs.tsv`, `batch_replication_null.tsv`, `split_half_pairs.tsv`,
`wt_batch_effect.tsv`, `cell_depth.tsv`, `genotype_meta_means.tsv` (per-genotype means of
every numeric metadata column, including the authors' `iESR_Gasch2017_UCell` and
`percRibo`).
