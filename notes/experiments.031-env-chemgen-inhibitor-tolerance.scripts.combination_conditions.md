---
id: 1iqpq53f8ub18m2avhdpekl
title: Combination_conditions
desc: ''
updated: 1790411845910
created: 1790411845910
---

## 2026.09.26 - Which served conditions dose more than one compound

Asked whether any served dataset carries a multi-inhibitor or multi-drug environment, which
is the environment-side analog of a double-gene perturbation. Ran
`experiments/031-env-chemgen-inhibitor-tolerance/scripts/combination_conditions.py` over the
three flattened stores.

- **Vanacloig 2022: none.** All 143,218 records dose exactly one compound.
- **Hillenmeyer 2008 HOM: 3 conditions, 2 pairs.** Lithium chloride 100 mM plus tacrolimus
  1 ug/mL (4,629 genes) is dose-matched to its singles. The two sodium chloride 600 mM plus
  tacrolimus conditions dose tacrolimus at 0.05 and 0.1 ug/mL, and HOM's tacrolimus singles
  are at 1.0 ug/mL, so they are not.
- **Hillenmeyer 2008 HET: 26 conditions, 10 pairs, 20 dose-matched over 7 pairs,
  113,783 records.** Every agent of every pair is also dosed alone in the same arm; the
  dose-matched flag asks whether it is dosed alone at the same value and unit.
- **The methotrexate by 5-fluorouracil block is a complete dose-matched 3 by 3
  checkerboard**: methotrexate at 125, 250 and 500 uM crossed with 5-fluorouracil at 19.2,
  38.4 and 76.8 uM, with all three methotrexate singles and all three 5-fluorouracil singles
  present at those same doses, 5,704 genes per cell.
- Other dose-matched pairs: 5-fluorouracil plus leucovorin (2 cells), amphotericin B plus
  flucytosine (2), fluconazole plus flucytosine (1), fluconazole plus itraconazole (2),
  flucytosine plus itraconazole (1), methotrexate plus leucovorin (3).
- Two other loaders build two-molecule environments that are not combination stress:
  `mormino2022` carries acetic acid plus anhydrotetracycline, where the second molecule is
  the CRISPRi inducer and is constant across every record, and `mota2024` carries one acid
  plus a typed pH edit, which is a physical perturbation rather than a compound.
- `wildenhain2015` has a 128 by 128 chemical-chemical synergy layer that the loader
  deliberately leaves out of scope, stated in its module docstring. That layer has no
  genotype axis.

Why this matters for the extrapolation question: the dose confound that blocks the Lian 2019
rounds comparison (singles at 5 mM, doubles at 10 mM) does not apply here, because the
dose-matched combination cells sit at exactly the doses of their own singles.

Result file: `experiments/031-env-chemgen-inhibitor-tolerance/results/combination_conditions.csv`
