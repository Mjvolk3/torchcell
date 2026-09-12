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
- [x] Wave 7: supervise the flux layer's own learned whole-cell concentration with Mulleder's absolute amino acids. Concentration readout +0.0145 (sd 0.0093, n = 3) over pooled at 200 epochs, while the turnover readout on the identical configuration collapses (0.002, spread 0.000); window widened to 1 M after 10.6 percent of measurements were found above the shipped ceiling: [[experiments.027-betaxanthin-metabolic.wave7-amino-acid-concentration]]
- [x] Wave 7: the second law as a differentiable flux-force coupling. `fluxforce` channel with four ends (box, balance, force, boxforce), the gradient the directional box does not pass, the sign flips the projection introduces, the unconditional dissipation readout, and eight cells (jobs 1745, 1749, 1755, 1761, 1764, 1767, 1770, 1771): the force end matches the sign clamp's exactness and beats it by +0.083 paired over three seeds while the clamp escapes 0 of 3, and the boxforce end meets the Genesis median rule on 100 of 100 top strains: [[experiments.027-betaxanthin-metabolic.wave7-flux-force-head]], [[torchcell.metabolism.flux_layer]]
- [x] Thermodynamic curation closure: 16 curated records under an admission rule, cancellation and non-chemical reaction rules, the carrier pseudo-pair and conserved-moiety routes, the accession composition guard (6 wrong prices withdrawn, glycogen re-derived per residue), and the verified tables: [[experiments.026-metabolism-flux.thermo-curation-closure]], [[torchcell.metabolism.thermo_curation]]
- [x] `notes-tex/026-kinetics-thermodynamics` rewritten around route attribution: every formation and reaction energy carries a named route and every gap a named failure class, with the recovery routes (external pKa cascade, structure writers, polymer conventions, curated records, composition guard) and the kinetic tiers as new figures and generated tables: [[experiments.026-metabolism-flux.kinetic-and-thermodynamic-parameter-datasets]]
- [x] `notes-tex/metabolic-module-report` updated with the closed counts, the evidence-class mix per parameter table and the route-attribution figure: [[experiments.026-metabolism-flux.enzyme-constrained-thermodynamic-flux-layer]]
- [ ] Rerun the betaxanthin parity on one kcat table; max-based balance tolerance; margin-tightened head cells 1736/1737 readout
- [x] Wave 7 parity on the shipped verified kcat table (3 seeds, jobs 1743/1748/1752/1756/1757/1759, paired difference -0.0550, sd 0.0904) and the max-based balance rule on the wave-6 projection checkpoints, where the one cell that met the Genesis criterion meets it only on the median: [[experiments.027-betaxanthin-metabolic.wave7-parity-one-table]]

## 2026.09.12

- [x] CatPred admission study: mirrored the release, ran it on the pinned pairs, measured the training overlap, scored it against the Wilson rule (rejected for k_cat by one hit, admitted for K_M), and analyzed the per-prediction variance: [[experiments.026-metabolism-flux.catpred-admission]]
- [x] Third balance rule (ratio over rows that turn over, 1e-3 of the flux scale with 1e-2 and 1e-4 sensitivity) on 18 checkpoints including the boxforce and force-end cells; Genesis 0.00 under both honest rules: [[experiments.027-betaxanthin-metabolic.wave7-parity-one-table]]
