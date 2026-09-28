---
id: zj5lc6lo4t9cio2ngx54qm1
title: Yeast9_chemical_accounting
desc: ''
updated: 1790544447539
created: 1790544447539
---

## 2026.09.27 - How complete is the chemical accounting of a yeast cell

Measures yeast-GEM 9.0.2 as an INVENTORY of chemical species rather than as a flux model, to
scope whether a retrobiosynthetic enumeration of yeast metabolism is worth starting.

| measurement | value |
|---|---|
| distinct chemical species (compartment copies collapsed) | 1,378 |
| species carrying a SMILES | 894 |
| structures naming ONE molecule (no unassigned stereocenter) | **273** |
| structures naming a SET of stereoisomers | 621 |
| reactions with a structure for every participant | 2,156 of 4,131 (52%) |
| species reachable from 267 boundary metabolites | 1,374 of 1,378 in 4 rounds |
| dead-end metabolite entries (our definition) | 747 of 2,806 (27%) |

**Three findings.**

1. **Only a fifth of the model is pinned to a single molecule.** 273 of 1,378 species. A SMILES
   with an unassigned stereocenter names a set, so it cannot carry a meaningful atom mapping or
   a unique thermodynamic value. No paper in the mirror reports this for any yeast
   reconstruction, which makes it the cleanest opening.
2. **Connectivity gives no useful bound.** Firing a reaction whenever any one substrate is
   present reaches 99.7% of species in four rounds. The rule is deliberately generous and the
   result is that graph reachability says only that the network is dense, so bounding the
   intermediate space needs reaction rules with atom mapping rather than connectivity.
3. **Half the reactions are structurally complete**, which is the practical ceiling on applying
   reaction rules to the model as released.

**One number to treat carefully.** Our dead-end count of 747 of 2,806 is not directly
comparable to the published Yeast8 figure of 464 of 2,742
([[chenGenomescaleModelingYeast2022]]); the definitions may differ and ours is the more
inclusive reading. The definition used is in the script docstring.

Figures: `yeast9_accounting.svg` (2x3) and `yeast9_scope.svg` (1x3), both in
`notes/assets/images/031-env-chemgen-inhibitor-tolerance/`. Document:
`notes-tex/031-yeast-chemical-space`. Literature numbers come from
[[experiments.031-env-chemgen-inhibitor-tolerance.scripts.yeast9_literature_counts]] and are
drawn hatched so measured and reported are never confused.
