---
id: 2l8k2qun3dinzs7aw4f1fcz
title: Dataset_joinability
desc: ''
updated: 1790492149110
created: 1790492149110
---

## 2026.09.27 - Can these datasets be joined into one training target

Tests the assertion that pooling makes, which is about the LABEL and not only the input: two
records measuring the same gene under the same compound must mean the same thing before one
loss can score them. A cell is one (gene, compound) pair from records that dose exactly one
compound and perturb exactly one gene.

**Finding 1: the scales differ by a factor of twenty.** Standard deviation of the served
response over cells: Hillenmeyer HET 0.38, Vanacloig 0.70, Hoepfner 1.29, Hillenmeyer HOM 2.35,
Wildenhain 7.50. A pooled squared-error loss without a per-source scale is a Wildenhain loss
regardless of record counts.

**Finding 2: the response polarity is NOT shared, and this is the one that matters.** The
loaders record each source's own definition verbatim, and they split into two groups:

| dataset | reports | sick strain is | skew |
|---|---|---|---|
| Vanacloig | log2(inhibitor/control) | negative | -2.6 |
| Hoepfner | adjusted MADL sensitivity | negative | -4.4 |
| Wildenhain | OD600 z-score | negative | -4.6 |
| Hillenmeyer HET | log2(control/treatment) fitness defect | positive | +1.2 |
| Hillenmeyer HOM | fitness-defect z-score | positive | +5.6 |

Two independent checks. The heavy tail falls on the declared side in all five (the skew column).
And the declared polarities predict the SIGN of all 10 pairwise correlations correctly, binomial
p 0.002 under a coin-flip null.

The cost of ignoring this is large and worst where the overlap is largest: as served,
Hillenmeyer HOM against Hoepfner reads -0.198 over 100,276 shared cells and Hillenmeyer HET
against Hoepfner -0.118 over 170,837. Pooling without orienting trains on contradictory labels.

**Finding 3: after orienting, agreement is real, positive and small.** Median 0.068 over the ten
pairs, maximum 0.198. The two Hillenmeyer arms, same laboratory and medium, reach 0.117 over
407,072 cells. Against Vanacloig everything is below 0.08 because it shares 2 compounds with
each genome-wide partner and 10 with Wildenhain.

**Conclusion for the design.** A shared SCALE is justified; a shared RESPONSE is not. These
datasets agree about which genes are generally fragile far more than about how a gene answers a
particular compound, so pooling should be expected to buy a gene prior and a wider compound
space rather than more labels for the same task. The target becomes
`y_tilde = s_k * (y - median_k) / sd_k` with `s_k` the declared sick-sign and `sd_k` computed on
the TRAIN split only, and the residual per-source scale rides on the dataset token.

Figure: `notes/assets/images/031-env-chemgen-inhibitor-tolerance/dataset_joinability.svg`.
Result files: `results/dataset_distributions.csv`, `results/cross_dataset_pair_overlap.csv`,
`results/polarity_check.csv`. Rendered as tables t11 and t12 and Figure 2 of
`notes-tex/031-unified-representation`. See
[[experiments.031-env-chemgen-inhibitor-tolerance.mermaid.dataset-unification]].
