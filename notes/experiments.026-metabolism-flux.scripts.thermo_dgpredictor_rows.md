---
id: ud5gevv9gaj5g56syczcfqs
title: Thermo_dgpredictor_rows
desc: ''
updated: 1789101185705
created: 1789101185705
---

## 2026.09.11 - What this script is

Names the 33 rows of yeast-GEM 9.0.2's `model_rxnDeltaG.csv` that come from dGPredictor in
kJ/mol rather than from yETFL in kcal/mol, so the parser's blanket 4.184 conversion can be
undone for them. Identification is by absence from yETFL's curated model, cross-checked
against the curation script's index block and against an eQuilibrator unit back-solve that
turned out to be uninformative and is reported as such.

Findings: [[experiments.026-metabolism-flux.yeast9-delta-g-provenance]].
