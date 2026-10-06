---
id: 44r8x2r4b2mcsoaiubw0yz1
title: Test_hoepfner2014
desc: ''
updated: 1790773263406
created: 1790773263406
---

## 2026.09.30 - Phase 16: the Dryad fetch, the polarity convention, the HIP/HOP confound

Twenty to thirty-one tests (one data-gated skip), 50 to 65 percent alone and 92 with the synthetic sibling (the batch commit at 500,000 records, lines 1394-1395, stays uncovered). `_dryad_get` against a fake session (a plain body, the Anubis challenge solved at nonce 26 with and without a base prefix, throttle back-off, giving up after 80 tries); `_fetch_from_dryad` skipping an empty chunk; `_resolver` with `genome=None`; three weak pre-existing tests strengthened to exact values.

Polarity: HIP and HOP cells of -3.0 are both stored as -3.0 and the units string says "negative = hypersensitive"; the loader reorients neither arm. This is the negative-is-sick convention the memory note records for Hoepfner (Hillenmeyer is the positive-is-sick one; the phase brief had them swapped and the writer corrected it). Confound: for one compound, dose and study the HIP and HOP environments differ in exactly `duration_generations`, `duration_hours` and `provenance_gaps`, both references on the same BY4743 diploid genome, the ploidy-duration confound recorded for experiment 033.

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.

## 2026.10.06 - Phase 21: the remaining one-line branches

- `_strain_construction`: one entry keeps plate and well, two agreeing entries drop them; a lab mismatch and a batch mismatch are each tested and raise with the sorted sets.
- `_load_ic30`: both MoA sheets contribute (novel-MoA CMB 99 -> 4.0), any other sheet is ignored, a MoA sheet without `IC30 (uM)` is skipped whole, digit ids only (`12`, ` 45 ` kept; `CMB7`, blank id, blank value skipped).
- `_column_census`: a CMB991 column is detection and positive control.
- `_iter_records`: an excluded CMB4019 cell is counted, and a short row (five fields) is kept under the detection rule (3 of 6 >= 0.5 * 6) but gets no record for the kept column past its end.
- Left uncovered: the 250,000-record commit branch, the deposit's existing-file branch and the dose-outside-IC30 append in `_write_drop_ledger`, which need a full build.
