---
id: rctoh6djr40ce82zye99xo5
title: Constraints
desc: ''
updated: 1788314310364
created: 1788314310364
---

## 2026.09.01 - Genome-scale model to constraint tensors

Pure functions of a `cobra.Model`. Nothing here knows the word "yeast", which is what makes
the port to another organism a matter of supplying a different model.

Two things worth knowing before using it.

**The thermodynamics are not in the SBML.** `grep -c deltaG yeast-GEM.xml` returns 0, so a
model loaded through cobra has no free energies at all. They live in
`data/databases/model_metDeltaG.csv` and `model_rxnDeltaG.csv`, which is what
`load_thermo_table` reads.

**That file uses two missing-value conventions.** Most gaps are the sentinel `10000000`, but
51 metabolites and 120 reactions are a literal `NaN`. Filtering only the sentinel leaves the
NaNs, which propagate through every sum and produce NaN gradients from a run whose coverage
number still reads 87 %. Rejecting both gives 2,389 metabolites (85.1 %) and 3,210 reactions
(77.7 %), which are the model's own curation counts.

`null_space_basis` and `independent_balance_rows` are the two spectral helpers: the first
gives an orthonormal basis of ker S so a flux can be made exactly mass-balanced, the second
gives the rank(S) = 2,593 independent rows so the balance penalty does not weight some error
directions several times over.

Full write-up: [[experiments.026-metabolism-flux.enzyme-constrained-thermodynamic-flux-layer]]
