---
id: 76ztn5xg5gyi2btjwb2bkuq
title: Cross_dataset_similarity
desc: ''
updated: 1790375609651
created: 1790375609651
---

## 2026.09.25 - Does Hillenmeyer carry Vanacloig's per-gene response structure?

Builds one gene x condition matrix per dataset from the served records, oriented so
negative is a fitness defect (Vanacloig's log2 ratio as stored, Hillenmeyer's
fitness-defect scores negated). The condition is the coarse label, compound or else
physical factor or else temperature, so Hillenmeyer's several doses, generation counts
and screens of one compound are averaged. Records whose queried genotype is not a single
gene are dropped before the pivot.

Outputs under `results/` (`<partner>` is `hom` or `het`):

- `reliability.csv`: per condition, `1 - mean(SE^2) / var(response)` over genes, the
  share of across-gene variance not attributable to the served replicate noise. The SE of
  a condition-level mean is the pooled record SE divided by the square root of the number
  of averaged records. Conditions with no served SE (HET, and the single-array HOM
  conditions) are NaN.
- `cross_spearman_<partner>.csv`: Spearman between every Vanacloig condition and every
  partner condition over the shared queried genes (at least 1,000 pairs).
- `top_matches_<partner>.csv`: the two best partner conditions per Vanacloig condition.
- `shared_compounds_<partner>.csv`: for benomyl, methyl methanesulfonate and the sodium
  acetate vs acetic acid near-pair, the correlation, the rank of the true match among all
  partner conditions, and the overlap of bottom-5% hits with a one-sided Fisher test.
- `structure_<partner>.csv`: the per-gene mean-response correlation across all
  conditions, and a Mantel-style Spearman between the two gene-gene profile-correlation
  matrices on 1,500 sampled genes against a 20-permutation gene-shuffle null, plus the
  median, 95th percentile and maximum of the cross matrix.

Figures (true-size SVG + PNG, timestamped) go to
`notes/assets/images/031-env-chemgen-inhibitor-tolerance/`: the cross-correlation heatmap
against the top-40 partner conditions by max |rho|, the per-condition reliability index
with the best cross rho overlaid, and the shared-compound scatters. They are embedded and
read in [[experiments.031-env-chemgen-inhibitor-tolerance]].
