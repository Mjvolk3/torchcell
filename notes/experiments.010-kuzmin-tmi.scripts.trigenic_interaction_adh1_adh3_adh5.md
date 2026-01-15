---
id: 8d6jgn9zwlqh7m2vjffdbu4
title: Trigenic_interaction_adh1_adh3_adh5
desc: ''
updated: 1768335437588
created: 1768335437588
---

## Case Study: ADH/ALD Triple Knockouts for 2,3-Butanediol Production

**Reference:** Ng et al. 2012 (DOI: 10.1186/1475-2859-11-68)

Predicts trigenic interaction scores (τ) for triple knockouts that redirect metabolic flux from ethanol to 2,3-butanediol.

### Model Predictions (CellGraphTransformer, Pearson=0.4619)

| Strain     | Genes             | τ (predicted) | Interpretation |
|------------|-------------------|---------------|----------------|
| B2C-a1a3a5 | ADH1Δ ADH3Δ ADH5Δ | **+0.0089**   | Neutral        |
| B2C-a1a3a6 | ADH1Δ ADH3Δ ALD6Δ | **-0.0012**   | Neutral        |
| B2C-a1a5a6 | ADH1Δ ADH5Δ ALD6Δ | **-0.0023**   | Neutral        |

**Threshold:** |τ| > 0.08 for "strong" genetic interactions (Kuzmin et al.)

### Key Finding

All predictions are **neutral** — consistent with Ng et al. 2012: "further deletion of more ADH genes did not result in a more drastic increase in 2,3-butanediol production."

The severe phenotypes (33-54% biomass) arise from **multiplicative single-gene effects** (especially ADH3 with SMF = 0.445), not from higher-order genetic interactions.

### Training Data Gap

- ADH3 (YMR083W): **0 experiments** in training data
- Model must generalize entirely from graph structure

### Related

- Costanzo2016 lookup: [[experiments.010-kuzmin-tmi.scripts.adh_ald_costanzo2016_lookup]]
- Detailed analysis: [[scratch.2026.01.09.125909-model-literature-application]]
