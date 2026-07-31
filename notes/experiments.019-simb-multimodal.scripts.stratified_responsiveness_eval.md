---
id: go00jxx8ybhmuxlw01a00ah
title: Stratified_responsiveness_eval
desc: ''
updated: 1785531966147
created: 1785531966147
---

## 2026.07.31 - Ruling out responsiveness composition as the cause of the 6x seed swing

An identical A0 config scored `val/mean/pearson_per_feature` at 0.1527 / 0.0235 / 0.0661 on seeds 0/1/2, and because `seed` selects the SPLIT, that +/-0.129 swing is larger than every architectural effect of round _008 combined (graph routing +0.020, decoder capacity 0.000, self-indicator -0.001). This script asks whether the swing is composition -- whether a val draw happened to be heavy in Kemmeren NON-RESPONSIVE deletions, whose profiles are flat by construction and so carry almost no variance for a per-feature correlation to latch onto. Measured on all three seeds it is not: the random split is already responsiveness-balanced, and the non-responsive strains are not flat.

- **Balance is already even, so composition cannot explain the swing.** Percent responsive among labelled expression strains, from `results/stratified_responsiveness_seed{0,1,2}.json`:

  | split | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | train | 48.1% (594/1236) | 47.6% (593/1245) | 47.9% (593/1237) |
  | val | 49.7% (77/155) | 51.7% (78/151) | 52.6% (81/154) |
  | test | 48.7% (75/154) | 50.3% (75/149) | 46.8% (72/154) |

  Val responsiveness moves 2.9 percentage points across the three seeds whose headline metric moves 6x. Nothing here for the stratified index in `kemmeren_responsiveness_index.py` to fix.

- **"Non-responsive" is not flat -- it is about 60% as dispersed.** Seed 0 val targets: `SD(y)` = 0.2820 responsive vs 0.1702 non-responsive (ratio 1.66); `mean|y|` = 0.1645 vs 0.0999. Seeds 1/2 reproduce the ratio (1.50, 1.55). Only the tail separates cleanly -- `frac(|y| > 1)` = 0.0112 vs 0.0019 -- and both are about 1% of measurements. So the non-responsive half cannot be written off as inherently unpredictable, and this measurement alone does not invalidate the 0.7746 replicate ceiling as the target.

- **The dilution is per-GENE, not per-strain.** Over the val strain-by-gene target matrix, the median per-gene SD across strains is 0.1442 / 0.1472 / 0.1476 (seeds 0/1/2) and the across-strain variance concentrates hard:

  | share of genes | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | top 1% | 17.4% | 15.5% | 15.6% |
  | top 5% | 37.7% | 34.6% | 34.2% |
  | top 10% | 49.5% | 46.7% | 45.9% |
  | top 25% | 68.0% | 66.4% | 64.8% |
  | top 50% | 84.5% | 84.0% | 82.6% |

  The bottom half of genes carry 15-17% of the variance while contributing half the terms of `pearson_per_feature`, which is an unweighted mean over the ~6,127 measured genes. That is the concrete dilution the pooled metric suffers, and it is stable across seeds -- a property of the assay, not of the draw.

- **Two guardrails, each from a bug this script hit.** `phenotype_values` is the COO array over EVERY phenotype on a record, so reading it raw pulled CalMorph morphology in beside the log2 ratios and produced `SD = 1993` on values that live in +/-5; the script now masks by `phenotype_type_indices` down to the expression types. And the GEO labels are COMMON names (BCK1) while the dataset keys SYSTEMATIC names (YJL095W), so the join goes through `genome.resolve_gene_name` and the match rate is printed rather than assumed -- a silent partial join would look like a clean result computed on whichever genes happened to map.

- **Only the DATA LEVEL pass actually ran.** `parse_args` accepts `--seed` and `--split` only, so the MODEL LEVEL analysis the docstring describes (`--checkpoint`: per-group `pearson_per_feature`, per-strain `pearson_per_instance`) is not implemented in the landed script and no model-level number was measured. Nothing is retrained and no index is rebuilt.

- **One bookkeeping gap, flagged not fixed.** The balance pass reports 534 val records for seed 0 (155 labelled + 379 unlabelled) while the dispersion pass keeps exactly the 155 labelled ones (seed 2 keeps 156, including one unlabelled strain). Hypothesis (untested): the balance loop's `any("expression" in str(t).lower() ...)` check over `phenotype_types` does not restrict to records that actually store an expression VALUE, so its `n_expr` column is really "all records in the split". Do not quote the unlabelled counts as expression records until that is checked.
