---
id: xf5dfc7aasskwpqnx1b1lbi
title: Nadal_ribelles2025
desc: ''
updated: 1784006057031
created: 1784006057031
---

## 2026.07.14 - Nadal-Ribelles 2025 pseudobulk Perturb-seq build

Loader: `torchcell/datasets/scerevisiae/nadal_ribelles2025.py`
(`NadalRibellesPerturbSeq2025Dataset`). Source: Nadal-Ribelles et al. 2025,
*Nat. Commun.* 16, doi:10.1038/s41467-025-57600-4; data Zenodo
10.5281/zenodo.14062629.

### What it is

Genome-scale single-cell **Perturb-seq** of an RNA-barcoded YKO deletion library
(KANMX4 marker swapped for URA3 so the genotype barcode in the URA3 3'UTR is
polyA-readable), profiled under CONTROL and osmostress (0.4 M NaCl, 15 min).
**LOCKED representation: pseudobulk-per-genotype + dispersion, NOT per-cell** (the
43 GB single-cell object is not used).

### Representation

- New phenotype family `PseudobulkExpressionPhenotype` (schema.py), a sibling of the
  RNA-seq expression family: `expression_log2_ratio` (per-gene pseudobulk log2 FC vs
  same-condition WT; scanpy `logfoldchanges`, Wilcoxon DE) + two per-genotype
  single-cell scalars — `dispersion` (`sd_lvscore_scaledFU2`, SD of the scaled SVD
  leverage score; WT ~= 1) and `n_cells` (`cell_number`). Both scalars are the POINT
  of the pseudobulk+dispersion decision. `dispersion`/`n_cells` are optional
  (default None) → backward-compatible additive fields.
- Genotype: one `MarkerDeletionPerturbation(marker="URA3")`; the source genotype
  barcode label is carried on **`strain_id`** (new optional field on
  `MarkerDeletionPerturbation`, default None → backward-compatible). This preserves
  replacement strains (`bc_YBR020W-1`/`-2`) that delete the same ORF as DISTINCT
  records (they carry different dispersion/n_cells).
- Environment: 2 records per genotype — control (base YPD) and osmostress (YPD +
  0.4 M NaCl `SmallMoleculePerturbation`, `duration_hours=0.25`).
- Reference: per-condition WT — `expression_log2_ratio` 0 for every gene, carrying
  the WT dispersion (~1.001) and n_cells (control 500 / NaCl 458). Two references.

### Build result (6188 records)

- Control 3091 + NaCl 3097 = **6188** records (one per genotype×condition); 3150
  distinct strains; 0 ORF-labels unparsed.
- **Ragged** gene vectors (per-comparison 0-count genes dropped): min 4223, max
  6313, median ~5837; union 6796. Gene *common* names → current-R64 systematic ORFs
  via the genome alias table; a gene not tested for a genotype is KEY-ABSENT (never
  0). 35.2M gene-values kept; ~1.13M (3.1%) dropped as unresolvable (ncRNA/retired
  symbols e.g. `15S-RRNA`, `SNR*`, `RUF*`); 15.7k within-record alias collisions
  deduped (two source names → one ORF, keep first).
- Verification (`RNASEQ_DATASETS`, `run_rnaseq`) **L0–L4 PASS**: L4 SGD gene
  containment 1.000 (6352 measured genes). L1 uniqueness extended to key on
  (strain_id, environment) so a strain profiled in both conditions is two records
  (backward-compatible with Caudal's single-condition survey).

### Provenance flag for review — assay-phase TEMPERATURE

The paper does NOT explicitly state the temperature of the 6-h YPD growth +
15-min NaCl treatment of the profiled cells. Only the **48-h URA- recovery is
stated, at 25 °C** (Methods, "Yeast Growth and harvest for the Perturb-seq
experiments"). **30 °C** is used as the documented representative (standard
*S. cerevisiae* growth temperature, and the temperature this paper uses for its own
growth-curve validations) and is **FLAGGED for review**. It is a shared constant
across both conditions and every genotype, so it does not affect any mutant-vs-WT
log2 FC — only the environment metadata. `GROWTH_TEMP_C` in the loader.

## 2026.09.11 - Two findings about the stored values, from experiment 028

1. **The stored `expression_log2_ratio` is scanpy's `logfoldchanges`, and one cell in five is
   a zero-mean sentinel.** The formula returns +-23 to +-33 whenever one group's mean is zero;
   20.2% of control cells carry such a value and 22.6% of genes carry it in more than half
   of their records. The paper only used these values above a mean-count filter (1.27) and a
   p threshold. A consumer must treat |value| >= 20 as absent, and even then the values are
   not on a microarray scale (sd 0.89 vs 0.23).
2. **The cell-to-genotype assignment (`assignment_consensus2`) is about 40% pure**, measured
   from the deleted gene's own detection in its genotype's cells: median excess-zero purity
   0.40 (n 281 genotypes whose deleted gene WT detects in >= 20% of cells); pdc1-labeled
   cells carry PDC1 at 13.4 UMI vs 15.0 in WT. Consequently every per-genotype pseudobulk
   is ~60% other genotypes, and the panel agrees with Kemmeren on 914 to 1,002 shared
   deletions at a per-strain median of ~0.01 whether the fold change is the stored one or
   recomputed from raw UMI three ways. As released, the dataset is not usable as a
   knockout-expression target; the `dispersion` scalar is a separate readout.

Scripts and numbers: [[experiments.028-knockout-expression.scripts.cross_study_ko_expression]],
[[experiments.028-knockout-expression.scripts.cross_study_recomputed]],
[[experiments.028-knockout-expression.scripts.nadal_assignment_purity]]. The single-cell
object and the paper's DE scripts are now in the raw mirror (md5s match Zenodo 14062629).

## 2026.09.12 - Methods facts from a full reread, and the within-study replication

From the mirrored paper and SI (nadal-ribellesSinglecellResolvedGenotypephenotype2025),
verbatim quotes in the session's extraction:

- Platform is Singleron GEXSCOPE microwell (High Density Matrix), NOT 10x: 220,000 cells
  per cartridge, 29 cartridges (14 control, 15 NaCl), one NovaSeq S4 lane. Median 1,200
  molecules and 550 genes per cell as reported; 934 UMI and 492 genes in the released
  control object.
- Growth: URA- recovery 25 C 48 h to saturation; 5 ul into 200 ul YPD in 96-well plates
  for 6 h to OD660 0.6 to 0.8 (temperature not stated); plates pooled into a flask,
  15-min control or 0.4 M NaCl on the pool, methanol fixation. The pooled window is the
  treatment, not the growth, so pooled competition is not a plausible confounder.
- Assignment: barcode read from the whole-transcriptome library (artificial contig) AND a
  targeted amplicon of the URA3 3'UTR; a cell called by only ONE library keeps that call
  (OR rule), conflicts are dropped; multi-genotype cells need one genotype in > 70% of
  reads. No purity or doublet rate is reported; doublets, low-quality and unassigned
  cells are one 350,913-cell removed category. Replacement strains of one gene carry an
  IDENTICAL barcode and are told apart only by reference-contig naming (bc-X, bc-X-2).
- 90% of genotype barcodes map within 300 bp of the intended locus; the other 10% are
  removed for detailed analysis (Supplementary Data 1, not mirrored).
- The only deleteome comparison is a DEG-count scatter (their Fig. S1i); the profile
  Spearman their script computes (median 0.013) is not shown. See
  [[experiments.028-knockout-expression.scripts.nadal_paper_deleteome_comparison]].
- Within-study replication: same-genotype cross-batch r 0.043 vs 0.022 null; split-half
  0.080 vs 0.051. See [[experiments.028-knockout-expression.scripts.nadal_batch_replication]].
  The Messner proteome, Kemmeren and Caudal agree pairwise on gene co-variation at 0.31
  to 0.48; this panel agrees with none at 0.05 to 0.11
  ([[experiments.028-knockout-expression.scripts.proteome_expression_covariation]]).
