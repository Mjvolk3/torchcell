---
id: o7wfmxjv61ovsi47wqy1mcf
title: Fetch_ion_formation_energies
desc: ''
updated: 1789109628960
created: 1789109628960
---

## 2026.09.11 - What this script is

Builds `ion_formation_energies.json`: standard formation energies for monatomic aqueous
ions, which component contribution cannot reach at all. The CODATA Key Values PDF is
retrieved, sha256-pinned and mirrored under `$DATA_ROOT`, then PARSED rather than
transcribed, and the Gibbs energy is derived from the tabulated enthalpy and entropy through
a formation reaction written out in the docstring. Every derived value is checked against
the independently published NBS value for the same ion and the run fails outside 2 kJ/mol;
the worst deviation over the eight ions is 0.774 kJ/mol. The Legendre transform it records
is asserted equal to eQuilibrator's own to 8.9e-16 kJ/mol.

Findings: [[experiments.026-metabolism-flux.pka-method-selection]].
