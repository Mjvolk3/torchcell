---
id: eqcuc8zqombl7ctqefcu5sj
title: Thermo_tier_d_probe
desc: ''
updated: 1789101829198
created: 1789101829198
---

## 2026.09.11 - What this script is

Probes every metabolite the eQuilibrator group decomposer refuses, one compound at a time,
against three routes: the reactant-contribution half of component contribution, a
same-composition surrogate under a stated spread bound, and neither. Prices only where
`standard_dg_formation` returns a value, and gives every other compound a verdict and a
remedy.

Result: 20 of 30 are monatomic ions needing a literature table, 10 have no defensible route,
and 0 of the 29 reactions they block close.

Findings: [[experiments.026-metabolism-flux.pka-method-selection]].
