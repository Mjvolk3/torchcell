---
id: zdymy9td8d0vhnkyvfe9wyg
title: Kinetic_substrate_writer
desc: ''
updated: 1789105468349
created: 1789105468349
---

## 2026.09.11 - The named compounds no catalog carries

The kinetics-side companion to the lipid writer, for substrates whose formula has no R, so
the GEM has committed to a definite composition and only the molecule is in question. The
chain length is DERIVED: the template is built for every length in a range and the GEM's own
formula selects one, which turns a modeling choice into a lookup against a constraint the
GEM already made. Every candidate passes the same neutralized-formula test the
cross-reference route uses.

Measured 2026.09.11: 2 written and 3 refused. `C26H47O9P` selects dolichol-4 for dolichyl
D-mannosyl phosphate uniquely out of n = 2 to 11, and the acyl-CoA template writes
cis-dec-3-enoyl-CoA. Refused with the mismatch recorded: N,N'-diformyldityrosine (the GEM's
formula and charge do not describe the named species under either reading),
phosphatidylethanolamine backbone (an R-group placeholder whose R was deleted from the
formula), and hydrogen sulfide (`HS` at charge 0 is not a molecule).

Writes `results/kinetic_substrate_structures.json`, read alongside the lipid writer's output
by `improve_kinetic_coverage.py --stage verify`. Full accounting:
[[experiments.026-metabolism-flux.kinetic-coverage-closure]].
