---
id: icxp7f8nncnrkwe2ipoolap
title: Yeast_biomass_composition
desc: ''
updated: 1790546410163
created: 1790546410163
---

## 2026.09.27 - Yeast dry-mass composition from the model's own biomass equation

The mirrored literature states yeast protein, RNA, lipid and cell wall as separate rounded
figures that do NOT add to one and say nothing about DNA or the small-molecule pool
([[experiments.031-env-chemgen-inhibitor-tolerance.scripts.yeast9_literature_counts]]).
yeast-GEM 9.0.2 carries a complete answer that needs no lookup: its biomass pseudo-reaction
draws one unit each from seven pools, and every pool lists its constituents with coefficients
in mmol/gDW, so coefficient x formula mass gives g/gDW directly.

| pool | percent of dry mass | published range |
|---|---|---|
| protein | **46.8** | 40-50 |
| carbohydrate | **38.3** | 25 (cell wall only) |
| RNA | **6.4** | 10 |
| unaccounted | 3.9 | |
| lipid | **3.5** | 10 |
| cofactor | **0.48** | |
| DNA | **0.39** | |
| ion | **0.25** | |
| **total** | **96.1** | |

**The question this answers.** After protein, carbohydrate, RNA and lipid, is the rest 5-10%
small molecules or DNA? **Neither.** DNA is 0.39% and cofactor plus ion is 0.73%. The
accounting closes to 96.1% and the 3.9% shortfall is the few constituents with no formula.

**Two caveats.** The model's carbohydrate pool runs above the 25% cell-wall figure because it
also carries storage glycogen and trehalose. And cofactor plus ion is a **lower bound** on the
small-molecule pool: it holds what the biomass equation requires, not the whole metabolite
pool. A soluble pool measured by extraction would be larger; the E. coli figure for that
quantity is 3 to 3.9%.

**The one approximation, and it is self-checking.** The protein pool's constituents are
amino-acyl tRNAs whose formulas carry an `R` for the tRNA, and the lipid backbones use the same
device. `formula_mass` ignores `R`, counting the moiety contributed to the polymer and not the
carrier. If that were wrong the pools would not sum to one; they sum to 96%, and the total is
printed on every run so the check is visible rather than asserted.

**Consequence for the chemical accounting.** The species that a full physical accounting is
about are numerous and are under one percent of the cell by mass, so the exercise is about
identity and connectivity rather than mass balance.

Result file: `results/yeast_biomass_composition.csv`. Rendered as Figure 2b of
`notes-tex/031-yeast-chemical-space`.
