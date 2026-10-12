---
id: pw25dz917ouaflq2olsctvi
title: Thermo_ion_gauge_check
desc: ''
updated: 1789109636415
created: 1789109636415
---

## 2026.09.11 - What this script is

Measures which reaction energies depend on the ion table's absolute values by running the
whole pipeline twice, once as built and once with every ion shifted by 1,000 kJ/mol, and
comparing the reactions that move against the reactions that are elementally unbalanced.

Result: 22 reactions touch a tabled ion, 6 move, and all 6 are elementally unbalanced sinks
and exchanges. The absolute values change no balanced reaction's energy.

Findings: [[experiments.026-metabolism-flux.pka-method-selection]].
