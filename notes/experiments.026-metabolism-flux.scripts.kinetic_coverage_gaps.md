---
id: jm2go59wpvuteppoosp6snq
title: Kinetic_coverage_gaps
desc: ''
updated: 1789100281973
created: 1789100281973
---

## 2026.09.10 - Why each catalytic unit is not measured or fully predicted

Re-derives the k_cat and K_M tiers through the same functions the flux layer calls, then
attaches one blocker enum to every catalytic unit that is neither measured nor predicted by
every admitted predictor: `no_uniprot_mapping`, `sequence_missing`, `complex_rule_unhandled`,
`reaction_has_no_substrate`, `substrate_structure_missing`, `predictor_failed`, `other`. Two
populations are reported separately, since they have different fixes: units on the
organism-default tier, and units missing at least one admitted predictor.

Measured 2026.09.10 over 3,728 units: every one of the 176 units without a prediction was
blocked by `substrate_structure_missing`, and no other blocker fired at unit level. After the
structure fix the residual is 10 units, each with the reason its substrates carry.

Writes `results/kinetic_coverage_gaps.json`; the pre-fix run is kept as
`results/kinetic_coverage_gaps_before.json`. Full accounting:
[[experiments.026-metabolism-flux.kinetic-coverage-closure]].

## 2026.09.11 - Reads the verified tables

The verified pair table is now authoritative for which substrates have a structure, so a
name match whose composition disagrees with the GEM's formula no longer counts as one, and
the resolver is called with the substrate map and the homolog mirror. Measured 2026.09.11:
96 measured / 782 homolog / 2,820 predicted / 30 default for both parameters, with all 30
defaults blocked by `substrate_structure_missing`.
