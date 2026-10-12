---
id: igm8hyypgsy96p37a7ljwfz
title: Thermo_polymer_repeat_unit
desc: ''
updated: 1789110593775
created: 1789110593775
---

## 2026.09.11 - What this script is

Prices a polymer species whose GEM formula is a repeat unit rather than a molecule, and only
after checking that the model's STOICHIOMETRY is per repeat unit, which it verifies through
the model's own mass balance. The monomer is read off the GEM's hydrolysis reaction, not off
the name. The glycosidic correction is measured on this table's own priced values: 8
hydrolyses over 6 distinct disaccharides give -12.1 +/- 6.8 kJ/mol, so the per-residue
correction is +6.05 and the sigma the surrogate carries is 3.41 kJ/mol in quadrature with
the monomer's own.

Priced: glucan, chitin, dextrin, starch, over 6 metabolite rows. Refused: pectin, whose
hydrolysis reaction does not charge-balance, and mannan, for which the model carries no
hydrolysis reaction that identifies a monomer.

Findings: [[experiments.026-metabolism-flux.pka-method-selection]].
