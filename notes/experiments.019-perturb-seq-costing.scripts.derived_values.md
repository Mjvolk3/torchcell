---
id: le3nlaylu8qwqc4y8zlwlul
title: Derived_values
desc: ''
updated: 1787625563294
created: 1787625563294
---

## 2026.08.24 - Numbers the review states that no other script emitted

`experiments/019-perturb-seq-costing/scripts/derived_values.py`

Written after an audit found several numbers in `notes-tex/microbe-perturb-seq`
that were hand arithmetic in the `.tex` rather than script output, which the repo
artifact rule forbids. Each was individually defensible and none was
reproducible.

What it computes, and why each is here rather than in the script it draws from:

- **`transcript_sensitivity()`** is a correction, not a transcription. Sec. 4.1
  printed 137 / 359 / 699 cells per perturbation, which paired the two-fold
  coefficient `A = 16.3` with `phi = 1.1` -- a `phi` calibrated to the MEASURED
  1.34-fold coefficient `A = 90.9`. That pairing implies a floor of `A*phi = 18`
  cells in a section that relies on 100. `A` and `phi` are pinned together by
  `A*phi = 100`, so each effect size implies its own `phi` and they must travel
  in pairs. Both self-consistent pairings are computed so the mix cannot recur,
  and the superseded values are kept with a note saying not to quote them.
- **`depletion_lever()`** -- the $99,160 rRNA figure, a difference of two budget
  rows quoted in four places.
- **`sublibrary_counts()`** and **`fourth_plate()`** -- the 54 / 11 sublibrary
  counts and the fourth barcode plate. The per-run cost was written as "$150"
  with no derivation; it is $147, and the split matters: a fourth BARCODE round
  is a third LIGATION round, so $135 is the ligase line growing by half and only
  $12 is the extra plate.
- **`sublibrary_item_reconciliation()`** -- why $55 per sublibrary is not the
  $53 the itemized lines sum to (Brettner's own leeway).

These are compositions of values other scripts publish, so they live here rather
than inside `cost_model`: putting a "subtract these two budget rows" helper in
the model would imply the model owns the claim. It does not; the prose does.

Writes `results/derived_values.json`.
