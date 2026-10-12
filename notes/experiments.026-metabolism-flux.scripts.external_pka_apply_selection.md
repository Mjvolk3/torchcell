---
id: 8btauid4an2hxen7phs623g
title: External_pka_apply_selection
desc: ''
updated: 1789101178218
created: 1789101178218
---

## 2026.09.11 - What this script is

Re-prices the 48 external-pKa lipids and the tier-a targets with the method
`pka_method_selection.json` chose, under the pinned protonation convention (the pH 7 major
microspecies is submitted to the compound bridge, not the neutral parent). Five
subcommands, one per environment: `targets`, `ladders`, `microspecies`, `combine`, `price`.
Every re-priced compound carries its method, version, weight sha256s, conditions, pKa
ladder and the resulting dGf'0.

Results and the per-class breakdown:
[[experiments.026-metabolism-flux.lipid-structure-writer]].
