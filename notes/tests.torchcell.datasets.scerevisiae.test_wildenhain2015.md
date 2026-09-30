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
