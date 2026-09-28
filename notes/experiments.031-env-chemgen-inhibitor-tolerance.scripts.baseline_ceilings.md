---
id: 7i0v5d1mv7m0evuflx4ouu7
title: Baseline_ceilings
desc: ''
updated: 1790481658783
created: 1790481658783
---

## 2026.09.26 - Linear and kNN against the reliability ceiling

Measures what ridge and k-nearest-neighbors actually reach on Vanacloig against the ceiling
the reliability work established. Two cold-start splits, two targets, ceilings recomputed per
target. Matrix is 3,598 genes by 41 compounds. All 12 molecule encoders cover every compound.

**Compound cold-start, leave one compound out, 41 folds.** Only a molecular feature can help,
because the held-out compound was never dosed.

| target | model | Spearman | Pearson | ceiling | fraction of ceiling |
|---|---|---|---|---|---|
| centered | fcfp4_count ridge | 0.311 | 0.350 | 0.835 | 0.42 |
| centered | random neighbor, k=3 | 0.020 | 0.004 | 0.835 | 0.00 |
| centered | mean of training profiles | -0.001 | -0.011 | 0.835 | 0.00 |
| raw | mole_static ridge | 0.341 | 0.393 | 0.838 | 0.47 |
| raw | gene mean over training compounds | 0.212 | 0.255 | 0.838 | 0.30 |
| raw | random neighbor, k=3 | 0.109 | 0.136 | 0.838 | 0.16 |

The centered target subtracts each gene's mean over the TRAINING compounds, refit inside every
fold, which removes the gene main effect and leaves the compound-specific signal. On that
target the no-feature baseline predicts a constant and has no correlation, so the null is a
random training neighbor instead. Molecule features beat it by more than an order of magnitude,
0.311 against 0.020, positive in 93% of folds, Wilcoxon p 1e-10. The signal is real and it is
not a gene main effect.

**Gene cold-start, five gene folds by 41 compounds, 205 scores.** Sequence and protein
embeddings for genes never seen in training.

| features | Spearman | Pearson | fraction of ceiling | folds positive | Wilcoxon p |
|---|---|---|---|---|---|
| prott5 ridge | 0.044 | 0.068 | 0.08 | 0.85 | 7e-26 |
| esm2 ridge | 0.043 | 0.059 | 0.07 | 0.81 | 2e-25 |
| calm ridge | 0.043 | 0.063 | 0.08 | 0.81 | 6e-26 |
| codon_freq ridge | 0.030 | 0.045 | 0.05 | 0.77 | 3e-18 |

Real but negligible. The signal is unambiguously above zero and reaches only 5 to 8% of the
ceiling, so a strain representation built from sequence does not currently predict an unseen
gene's chemogenomic response.

**Three findings worth carrying forward.**

- Ridge beats kNN on both targets and every encoder. A linear map from the molecule embedding
  to the whole gene profile is the better simple model, not neighbor transfer.
- The encoder ranking does not favor the large pretrained models. FCFP4, a plain RDKit
  feature-class fingerprint, is best on the centered target, and the RDKit 2D descriptors are
  second. This matches the published finding that count fingerprints are hard to beat.
- Within-dataset chemistry works far better than cross-dataset chemistry. The earlier
  cross-dataset chemistry-to-response correlation was at most 0.09, while within Vanacloig the
  same encoders reach 0.31 on the compound-specific target. One medium, one dose basis and one
  readout is what makes the difference, which is consistent with only methyl methanesulfonate
  transferring between datasets.

Headroom is large. The best feature set reaches 42 to 47% of the ceiling, so more than half the
predictable variance is unclaimed by these models.

Result files: `results/baseline_ceilings_compound_cold.csv`,
`results/baseline_ceilings_gene_cold.csv`, `results/baseline_ceilings_summary.csv`
