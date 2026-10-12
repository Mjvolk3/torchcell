---
id: 5j3jic81vkcw2c5s0zo507v
title: Thermo_structure_attempt
desc: ''
updated: 1789110601216
created: 1789110601216
---

## 2026.09.11 - What this script is

Takes every species still carrying `no_structure_available` as far as a structure goes, then
stops, and records per compound which of three things stopped it.

Two findings. The protein-linked residues (`[protein]-L-lysine`, the cytochrome c species,
Apocytochrome c) have structures that carry an ATTACHMENT POINT, so the species is a residue
inside a macromolecule and can never carry a formation energy of its own; that is detected
from the structure, not the name, and moves them to `placeholder_structure`, a permanently
closed reason. The metalloporphyrins (heme a, heme o, siroheme) come back
`composition_mismatch`: the only structure the GEM's own accessions reach is the IRON-FREE
macrocycle, a different molecule from the GEM's iron-containing species, so the retrieval
never reaches the right molecule at all.

Findings: [[experiments.026-metabolism-flux.pka-method-selection]].
