---
id: le3nlaylu8qwqc4y8zlwlul
title: Derived_values
desc: ''
updated: 1787625563294
created: 1787625563294
---

## 2026.08.24 - Numbers the review states that no other script emitted

`experiments/024-perturb-seq-costing/scripts/derived_values.py`

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

## 2026.08.24 - `plate_set_lifetime()`, and separating two failure modes

Review round 2 pushed back on the start-up cost being amortized over 215 protocol
runs: if freeze-thaw retires a plate before it is drawn down, the amortization is
fiction and the real per-run cost is higher than quoted. Fair, and it needed a
number rather than an assertion.

The answer is that the two failure modes are separable, and separating them makes
the objection answerable by aliquoting rather than by a shorter plate life.

- **Depletion is a volume budget.** 215 round-3 withdrawals is the binding one of
  the three capacities Brettner et al. state, and it is a hard count. At a full
  round-1 plate that is 103 million cells barcoded, and the depleted split-pool
  row needs 13 runs for a genome-scale screen at 250 cells per target gene, so
  one plate set is about 16 genome-scale screens.
- **Freeze-thaw is a cycle budget on whichever plate is being pipetted from**, and
  aliquoting decouples it from the volume budget. Split a source into `w` working
  plates in ONE thaw and the source sees one cycle whatever follows, while each
  working plate serves `215/w` runs and therefore sees `215/w` cycles. Keeping
  every plate under a tolerance of `f` needs `ceil(215/f)` working plates: 22 at
  `f=10`, 9 at 25, 3 at 100. Empty plates and one thaw are nearly free against
  $7,699, so the limit becomes a plate-count decision, not lost capacity.

`f` is NOT measured and is not in the mirror, so the swept tolerances are
illustrative and the function records `freeze_thaw_tolerance_measured: False`.
What survives whichever value is true is the shape of the answer. The one piece
of indirect evidence is that Gaisser et al.'s troubleshooting guide treats
remaking the working plates as the standard response to a barcoding failure,
which is not what a protocol expecting 215 cycles per plate would say.

`runs_per_genome_screen_250` is recomputed through `CM.budget_for` rather than
transcribed, so it tracks the budget model instead of drifting from it.
