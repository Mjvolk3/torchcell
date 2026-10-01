---
id: 1ri3t9rtjy956o6gftoznhj
title: Test_hillenmeyer2008_synthetic
desc: ''
updated: 1790562883031
created: 1790562883031
---

## 2026.09.27 - The Hillenmeyer 2008 loader through a real genomes registry

Twelve tests with the registry manifest and FASTAs under a `tmp_path` `DATA_ROOT`; the raw mirror is written by the loader's own `deposit_raw_mirror` with no network. Loader coverage 90%; the `genome=None` branch (lines 986 to 991) builds a real genome and stays untested. Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.
