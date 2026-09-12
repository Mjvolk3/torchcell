---
id: mx4aksqgh2656bjiegi6bi3
title: Nadal_assignment_purity
desc: ''
updated: 1789173841605
created: 1789173841605
---

## 2026.09.11 - The cell-to-genotype assignment is about 40% pure

Script: `experiments/028-knockout-expression/scripts/nadal_assignment_purity.R` (`r-seurat`
env), output `$DATA_ROOT/data/torchcell/nadal_ribelles_perturbseq2025/recomputed/assignment_purity.tsv`.

A cell that carries deletion g cannot express g. For each of 3,314 control genotypes
whose deleted ORF is in the count matrix, the fraction of its own cells with at least
one UMI of that ORF is compared with the fraction of WT cells detecting it.

- Among the 41 genotypes whose deleted gene WT detects in >= 50% of cells, the genotype's
  own cells still detect it in a median 46% (IQR 37-60%), 0.64 of the WT rate; 38 of 41
  are indistinguishable from WT on their own deleted gene, none is clean (< 0.1).
- Per-cell counts of the twelve most expressed deleted genes are a mixture, not a uniform
  reduction: `bc-YLR044C` (pdc1) cells carry PDC1 at mean 13.4 UMI vs 15.0 in WT with 9% of
  cells at zero (WT 1%); `bc-YDR382W` (rpp2b) 10.1 vs 14.6; `bc-YDL081C` (rpp1a) 4.6 vs
  12.5 with 20% zero vs 1%. A Poisson at those means has essentially no zeros, so the
  zero cells are the true deletion cells and the rest express the gene like WT.
- Excess-zero purity, p = (z_g - z_wt) / (1 - z_wt), the share of a genotype's cells that
  are true deletion cells: median 0.36 (IQR 0.25-0.40) at WT detection >= 0.5 (n 41) and
  0.40 (IQR 0.26-0.52) at >= 0.2 (n 281).

Batch is not the cause: 3,307 of 3,506 genotypes span three or more of the 14 batches
and the 500 WT cells span all of them. What produces the impurity (the consensus of clone
and genotype barcodes, ambient barcode reads in the microwell format, or strain identity
in the library) is not measured here. The consequence is: every per-genotype pseudobulk
in the released object is roughly 60% other genotypes, which is why the recomputed fold
changes agree with Kemmeren no better than the stored ones, why agreement rises with a
strain's effect size (a strong effect survives 60% dilution) and why it does not rise with
the number of cells.

## 2026.09.11 - The read-through alternative, tested and rejected

The cassette design (paper Methods: heterologous terminator shortened from 262 to 43 nt
"enabling the use of the endogenous terminator"; the targeted amplification "has coverage
of the endogenous terminator") means the pTEF1-URA3-barcode transcript ends at the deleted
gene's own terminator, so 3'-end reads from it land on the deleted locus. Under that
mechanism a genotype's own cells would detect the deleted gene at a pTEF1 level regardless
of native expression. Measured (2,724 genotypes with >= 20 cells): own-cell detection
tracks the WT detection level all the way down (Spearman 0.64, ~0.6x in every bin), and
for the 1,586 deleted genes WT detects in < 5% of cells the own cells detect it in a
median 2.9% (2.1% of them above 20%). The cassette does not supply the counts; the
excess-zero purity reading stands, with the caveat from the signature tests that it is
confirmed only where a response is detectable.
