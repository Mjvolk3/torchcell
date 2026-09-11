---
id: edswqwrpw2ljv0n6knn1ake
title: Thermo_acyl_transfer_writer
desc: ''
updated: 1789109643888
created: 1789109643888
---

## 2026.09.11 - What this script is

A structure writer whose evidence is the REACTION rather than the name: in an acyl transfer
the product is the substrate alcohol with one hydrogen replaced by the acyl group, and both
are already in the model with structures. Composes the product, runs the same formula
acceptance test the lipid writer uses, and prices nothing; pricing is
`external_pka_apply_selection.py`'s job.

It corrects a wrong structure: the pilot carried `methionyl acetate` as the mixed anhydride
of METHIONINE (`C7H13NO3S`), while yeast-GEM declares `C6H12O2S` and makes the species from
METHIONOL in `r_4690`. The pilot had no formula acceptance test.

Findings: [[experiments.026-metabolism-flux.lipid-structure-writer]].
