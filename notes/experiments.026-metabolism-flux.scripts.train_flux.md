---
id: jsjscjz6gmefq2u151raoot
title: Train_flux
desc: ''
updated: 1788314338597
created: 1788314338597
---

## 2026.09.01 - The five-arm diagnostic sweep

Betaxanthin (Cachera 2023) and the 19-amino-acid deletion panel (Mulleder 2016), both out of
the already-built `fig6_pigment_transfer` dataset. Five arms, each changing one thing:
`pooled`, `flux_off`, `flux_free`, `flux_anchored`, `flux_nullspace`.

**Two decode facts cost real time and are worth remembering.** `phenotype_sample_indices`
indexes the experiment WITHIN a genotype and is not offset across the batch, so it is not
the batch row; `phenotype_values_batch` is. And betaxanthin and the metabolome share the
label `metabolite_level`, so they are separated only by group WIDTH (1 versus 19), never by
position: measured over 1,200 records, the betaxanthin group comes first in only ~11 % of
co-measured genotypes.

The loop is seed-major on purpose, so any interrupted prefix is a balanced experiment across
all five arms rather than complete data for some arms and none for others.

Full write-up: [[experiments.026-metabolism-flux.enzyme-constrained-thermodynamic-flux-layer]]
