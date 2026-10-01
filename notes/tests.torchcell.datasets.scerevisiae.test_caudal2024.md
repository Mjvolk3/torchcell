---
id: 89lo1xt9so6tyi2w34q0sem
title: Test_caudal2024
desc: ''
updated: 1790777401208
created: 1790777401208
---

## 2026.09.30 - Phase 17: the four consumed columns, rounding, the reference

New paired file, eight tests, 94.6 to 100 percent with the two existing Caudal files. The loader reads only `Strain, systematic_name, count, tpm`, so the released header's extra columns change nothing and no replicate column is consumed; two allele rows summed; a count of 12.5 rounding half-to-even to 12; the reference as the mean over isolates carrying the gene (round(16.25) = 16); a table missing `tpm` refused with pandas' exact message; the SGD FASTA's last tagged record flushed; the stale raw matrix used; `main`'s exact stdout.

Findings: an existing raw file is never re-hashed (line 324); a blank `systematic_name` is dropped silently by groupby's `dropna` (463); a zip with no `.tab` member raises a bare `StopIteration` (448); the unextractable-member guard can never fire on a real archive and, forced, drops that gene's variants without a log (522-523).

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Finding retired: `test_a_stale_raw_matrix_is_kept_and_built_from` is now `test_a_stale_raw_matrix_is_refused_at_build_time` (the matrices are verified against the genomes tier). Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.

## 2026.10.01 - Fix PR for the pinned findings

Retired the bare StopIteration and silent unextractable-member findings (issue #541): a zip with zero or two `.tab` members raises `MissingTabMemberError` and a forced unextractable member raises `UnextractableMemberError`, both with exact messages. The blank `systematic_name` finding stays pinned as record-changing (459,790 affected rows in the built isolates).
