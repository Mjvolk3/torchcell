---
id: 4y5f3p2eom0zsjur4ceqa6i
title: Flux Module
desc: ''
updated: 1788504666912
created: 1788504666912
---

## 2026.09.04 - Consolidated account of the thermo module

The scattered notes on the enzyme-constrained thermodynamic flux module are consolidated
into one typeset document:

- Source: `notes-tex/027-metabolic-flux-module/`
- Built PDF: `notes/assets/pdf-output/` is not used here; the PDF sits beside its source at
  `notes-tex/027-metabolic-flux-module/027-metabolic-flux-module.pdf`
- Build: `make -C notes-tex/027-metabolic-flux-module` and `make check`

### What it consolidates

| Section | Drawn from |
|---|---|
| What the module computes | `torchcell/metabolism/flux_layer.py` (the authority), [[torchcell.models.equivariant_cell_graph_transformer.mermaid.metabolic-module]], [[torchcell.models.equivariant_cell_graph_transformer.mermaid.type-i-ii]] |
| The three parameter tables | [[experiments.026-metabolism-flux.kinetic-and-thermodynamic-parameter-datasets]] |
| Thermodynamic coverage | `experiments/026-metabolism-flux/results/unresolved_metabolites.json`, `thermo_plot_summary.json` |
| The ablation registry | `ARMS` in `experiments/026-metabolism-flux/scripts/train_flux.py` |
| What has been measured | `flux_arms_summary.json`, the 020 Optuna studies, `merzbacher_comparison_figures.json` |
| The Delta run | `experiments/027-betaxanthin-metabolic/README.md` and the job's own preflight |

### The three facts most worth carrying out of it

**The module is neither Type I nor Type II.** By the letter of the taxonomy it is Type I,
but every other Type I instrument acts within gene index space at dimension d, while this
one leaves for reaction space under a fixed external stoichiometry and its parameters are
tabulated constants rather than learned weights. Its output is a latent physical state, and
the phenotypes are read off it.

**K_M is built and not wired.** Three K_M tables exist (UniKP, EITLEM, Boost_KM at 93.0%
coverage) and the saturation term is inactive in the code. kcat is wired but resolves from
the Open Enzyme Database, not from the five predicted tables, so it runs at 4.0% measured
plus a default. Current runs test the module's structure and thermodynamics, not its
predicted kinetics.

**026 tested the module against a broken control.** `train_flux.py` builds the weak
architecture family (2 graphs, learnable embeddings, MSE, hidden 32), measured at 0.04-0.16,
while the strong family (9 graphs, prot_T5, CRPS, hidden 90) measures 0.32-0.43. Its
`flux_off` arm at 0.0837 is the right control for that sweep and is not a CGT baseline.

### Two corrections to earlier claims

**The ChemAxon blocker does not apply to betaxanthin.** All the pathway intermediates are
already in the eQuilibrator compound cache, so no compound needs creating. Separately,
upstream `equilibrator-assets` merged `pkas_provided_externally` on 2026-08-15, so even a
genuinely novel compound no longer needs cxcalc; that is master-only, not in the pinned
0.6.0.

**The coverage shortfall is not a mapping gap.** Applying every identifier route available
gains 28 metabolites and 24 reactions, 0.58 points. The summary also under-reported the
problem: 332 metabolites fail to match, and a further 375 match a cache record carrying no
structure, so total unpriced is 707 not 332. Of the 679 still unpriced after recovery, 249
are a real ceiling (placeholder formulas, no group decomposition) and 430 are a
structure-curation gap.

**And one new bug found in the build already in use:** 31 metabolites carry a formation
energy from a compound whose heavy-atom composition disagrees with the GEM formula (every
dolichol, `glycogen`, and `protein`/`carbohydrate`/`sterols` matched to concrete molecules),
touching 14 covered reactions. The composition guard that finds this is written but not yet
applied to `resolve_compound`.
