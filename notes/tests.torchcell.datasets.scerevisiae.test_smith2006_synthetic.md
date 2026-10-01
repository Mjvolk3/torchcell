---
id: 7lwmang8rkjvkgknxy4yj7u
title: Test_smith2006_synthetic
desc: ''
updated: 1790562905536
created: 1790562905536
---

## 2026.09.27 - The Smith 2006 loader built end to end

Seven tests. Nothing installed can write a BIFF `.xls` (no xlwt; xlrd 2 only reads), so the one-line loader `_read_table` is stubbed and everything after it runs for real, download and deposit included. Loader coverage 95%. Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.
