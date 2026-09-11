---
id: 60xy13gkx1i9s0thqvye9j8
title: Kinetic_predictor_spread
desc: ''
updated: 1789100304538
created: 1789100304538
---

## 2026.09.10 - Where the admitted predictors disagree

Per-unit log10 range across the admitted predictors, grouped three ways, each a property of
the unit known before any predictor runs: EC class, substrate heavy-atom count, and number of
genes in the catalytic unit. Wilson 95 percent intervals on every fraction.

Measured 2026.09.10 over 3,717 units: k_cat median 0.739 decades and 30.4 percent above one
decade overall, rising monotonically with substrate size to 74.5 percent for substrates above
60 heavy atoms, and 61.6 percent for two-gene complexes. K_M is flat in substrate size and
concentrated instead in reactions with no EC annotation.

Writes `results/kinetic_predictor_spread.json` and one six-panel figure. Full accounting:
[[experiments.026-metabolism-flux.kinetic-coverage-closure]].

## 2026.09.11 - Recomputed on the verified tables

Reads `<predictor>/verified/<parameter>.parquet`. Measured 2026.09.11 over 3,698 units:
k_cat median 0.738 decades and 29.9 percent above one decade, and the concentration in large
substrates and multi-gene complexes is unchanged. Figure
`kinetic_predictor_spread_2026-09-11-00-09-18.svg`.
