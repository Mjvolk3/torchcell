---
id: a707uow4qx9hhb7k0ye4rkk
title: Improve_kinetic_coverage
desc: ''
updated: 1789100289501
created: 1789100289501
---

## 2026.09.10 - Resolve the missing substrate structures, predict the newly reachable units

Three stages. `--stage resolve` prices the blocked units' substrates through the MetaNetX
cross-reference route (the GEM annotates an older MNXref release, so its ids are not rows in
the current one) and through the formula-verified lipid structures, accepting a candidate
only when RDKit's formula for the SMILES equals the GEM's formula with both neutralized.
`--stage predict` runs one admitted predictor over the new rows only and appends to that
predictor's parquet with an append manifest. `--stage report` re-resolves both tables and
writes the coverage ceiling, including how much of the predicted tier rests on a structure
that fails the composition check.

Measured 2026.09.10: 75 of 86 blocked substrates priced, 165 of 176 units unblocked, the
organism-default tier down from 170 to 10 units for both parameters. The fix is additive by
construction: a metabolite the existing pipeline already priced is never re-resolved.

Writes `results/kinetic_coverage_structures.json`,
`results/kinetic_coverage_new_pairs.parquet` and `results/kinetic_coverage_closure.json`.
Full accounting: [[experiments.026-metabolism-flux.kinetic-coverage-closure]].

## 2026.09.11 - Two more stages: verify, and predict-verified

`--stage verify` re-tests every substrate structure the predictions rest on against the
GEM's formula, repairs what the verified routes can and drops what they cannot, and writes
the verified pair table plus a wider substrate map for the measured tier to match against.
`--stage predict-verified --predictor <name>` brings one predictor up to that pair set,
reusing any `(gene, SMILES)` it has already scored and writing
`<predictor>/verified/<parameter>.parquet` with a manifest naming the sha256 of the table it
supersedes. `--stage report` now resolves four variants so the structure fix, the
substrate-matched measured tier and the homolog tier are separable.

Measured 2026.09.11: 97 of 1,425 structures failed, 30 repaired, 67 unrepairable, 208 pairs
dropped and 19 units lost. The verified tables carry zero failing structures.
