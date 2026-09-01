---
id: kh60z2vdirzprmo0tvnmkuu
title: Figure Candidates
desc: ''
updated: 1787871660496
created: 1787871660496
---

## 2026.08.27 - Candidate Panels for the Two Draw.io Figures

152 figures exist in `notes/assets/images/008-xue-ffa/` across 9 categories, plus 671
network overlays. This is the shortlist to review before storyboarding, grouped by the role
each panel could play. Nothing here is composed yet; panel letters are deliberately absent
so arrangement and A/B/C labeling stay a draw.io decision.

**Restyling status.** Everything below marked `legacy` was written before the repo figure
standards and does not use `PLOT_PALETTE`, `PANEL_WIDTHS_MM`, Arial 6 pt, or true-size SVG.
Those need a restyling pass before they go in a paper figure. Panels marked `standards` were
built to them.

### Figure 1 candidate: the epistasis result

| panel | file | status | what it shows |
| --- | --- | --- | --- |
| model agreement | `upset_trigenic_Total_Titer` | legacy | 76 of the trigenic interactions are called by all four models; the disagreement structure is the rest of the bars |
| consensus + selection | `notable_interaction_selection` | standards | agreement histogram, per-readout sign composition, and the top 8 by absolute tau across all four models |
| scale dependence | `linear_vs_log_scale_sign_test` | standards | 15 of 27 linear-positive triples lose that sign on the log scale, and no negative ever becomes positive |
| effect distribution | `multiplicative_ffa_distributions_and_volcano_3_delta_normalized` | legacy | tau distribution and volcano per readout |
| significance summary | `multiplicative_ffa_significance_summary_3_delta_normalized` | legacy | counts by readout and order |

### Figure 2 candidate: structure and engineering consequence

| panel | file | status | what it shows |
| --- | --- | --- | --- |
| network overlay | `ffa_multigraph_overlays/multiplicative/Total_Titer/*_enriched.png` | legacy | the interaction network on a biological graph; five enriched Total Titer variants exist, listed below |
| pathway context | `ffa_bipartite_network` | legacy | TF, gene, reaction and FFA bipartite layout |
| KO trajectories | `ffa_epistatic_path_panels_top` | standards | all six KO orders per triple, with the badge for interaction sign and support |
| trajectory ladder | `ffa_total_titer_trajectories` | standards | the order-agnostic ladder over all 120 triples |
| graph enrichment | `trigenic_connected_graph_enrichment` | legacy | interaction overlap with each biological graph |

### The five enriched Total Titer overlays to choose from

All under `ffa_multigraph_overlays/multiplicative/Total_Titer/`, about 5.5 MB each:

- `..._Genetic_Interactions_connected_enriched.png`
- `..._Physical_Interactions_enriched.png`
- `..._STRING_12_0_Coexpression_connected_enriched.png`
- `..._STRING_12_0_Experimental_connected_enriched.png`
- `..._TFLink_enriched.png`

`TFLink` is the most defensible for a TF deletion panel, since it is the TF-to-target graph
and the perturbations are transcription factors. That is a judgement, not a measurement.

### Open decisions

1. **Consensus threshold.** Requiring all four models is the most conservative option and it
   is what the current selection uses. Relaxing it changes the positive count and almost
   nothing else:

   | threshold | interactions | positive | negative |
   | --- | --- | --- | --- |
   | any 1 of 4 | 91 | 8 | 83 |
   | at least 2 of 4 | 87 | 6 | 81 |
   | at least 3 of 4 | 78 | 0 | 78 |
   | all 4 | 73 | 0 | 73 |

   Negatives are insensitive to the choice. Positives exist only at 1 or 2 models, so the
   consensus rule is what removes them, not the data.

2. **Which model leads**, if consensus is dropped. Per model on total titer, FDR < 0.05:
   multiplicative 86 (11 positive), additive 84 (6 positive), GLM log-link 83 (0), log-OLS
   84 (0).

3. **C14:0.** 20 consensus interactions, every one positive, against 0 negative. No other
   readout behaves this way. Worth deciding whether this is biology or an artifact before it
   goes in a figure.

### How BH is applied here

Benjamini-Hochberg runs **separately within each model**, on that model's own tests for the
plotted readout (119 trigenic interactions). It is never pooled across models, which would
be wrong because the four models score the same measurements and their p-values are strongly
dependent. The consensus rule is a conjunction of four separately controlled sets, which is
more conservative than any single one.

The `fdr_corrected_p` column stored in the results CSVs is a different quantity: it pools all
714 triple-by-readout tests. See [[plan.008-xue-ffa-epistasis-audit.2026.08.15]].
