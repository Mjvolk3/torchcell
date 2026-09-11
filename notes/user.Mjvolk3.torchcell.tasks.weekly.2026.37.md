---
id: c22nukvicut8hu977vjk3d0
title: '37'
desc: ''
updated: 1789139102284
created: 1789139102284
---

## 2026.09.11

- [x] Overnight metabolic-module run, driver record and cross-stream summary: [[experiments.027-betaxanthin-metabolic.overnight-2026-09-11]]
- [x] kcat and KM coverage closure, homolog tier, organism-agnostic pipeline: [[experiments.026-metabolism-flux.kinetic-coverage-closure]]
- [x] pKa method selection (2,400 compounds, compartment pHs), external repricing, ion table, ceiling ladder: [[experiments.026-metabolism-flux.pka-method-selection]], [[torchcell.metabolism.thermo_pipeline]]
- [x] Projection flux head, directional-box second law, sampler, covariance wiring, twelve cells: [[experiments.027-betaxanthin-metabolic.projection-head]], [[torchcell.metabolism.flux_layer]]
- [x] Trainer feasibility block, checkpoints, amino-acid head, six-seed parity: [[experiments.027-betaxanthin-metabolic.wave6]]
- [x] Wave 7: supervise the flux layer's own learned concentration with Mulleder's absolute amino acids, concentration window check and widening: [[experiments.027-betaxanthin-metabolic.wave7-amino-acid-concentration]]
- [x] Wave 7: the second law as a differentiable flux-force coupling, `fluxforce` channel, the gradient the directional box does not pass, dissipation readout: [[experiments.027-betaxanthin-metabolic.wave7-flux-force-head]], [[torchcell.metabolism.flux_layer]]
- [x] Thermodynamic curation closure: 16 curated records under an admission rule, cancellation and non-chemical reaction rules, the carrier pseudo-pair and conserved-moiety routes, the accession composition guard (6 wrong prices withdrawn, glycogen re-derived per residue), and the verified tables: [[experiments.026-metabolism-flux.thermo-curation-closure]], [[torchcell.metabolism.thermo_curation]]
- [ ] Rerun the betaxanthin parity on one kcat table; max-based balance tolerance; margin-tightened head cells 1736/1737 readout
