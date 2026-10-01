---
id: 6y5tj57irevrb67v1wqeqby
title: Test_wildenhain2015
desc: ''
updated: 1790759627881
created: 1790759627881
---

## 2026.09.30 - Phase 12: exact records, refusals, every download path

Eight to twenty-four tests (one mirror skip), 73 to 97 percent. Two `pytest.approx` checks tightened to exact (SD `math.sqrt(2)`, SE 1.0); `model_dump()` equality for the two-screen record and the single-screen record with its gaps; reference index [[0, 2], [1, 3]]; the log counts (8 datapoints, 1 non-strain row, 5 cells); refusals (an ORF not in the genome, two compounds with one canonical name, zero SD); `_canonical_common_names`; `deposit_raw_mirror`, `load_manifest` and `manifest_sha256`; a build through symlinks into `raw/`.

Findings: the datapoint key is the z string, so `-4.0` and `-4.00` count as two screens with SD 0 and the build aborts (lines 614, 725); `download` with no mirror deposited fails as a bare `FileNotFoundError` on `manifest.json` (553).

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.

## 2026.10.01 - z-key and manifest findings retired (issue #520)

Retired: `test_two_z_strings_of_equal_value_abort_the_build` and `test_download_without_a_manifest_raises_file_not_found`. Now asserted: `-4.0` and `-4.00` in one cell build one record with z -4.0, n 1 and the two single-screen dispersion gaps; `download()` with no manifest raises the exact `RuntimeError` naming the manifest path and the deposit step.

Review follow-up (same day): `test_a_non_finite_or_unparseable_z_refuses_naming_the_cell` (`nan`/`nan`, `inf`, `n/a`) asserts the exact messages and that no store is left.
