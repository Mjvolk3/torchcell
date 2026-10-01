---
id: gs1v12k83g5b5ev2wuk01pq
title: Test_nadal_ribelles2025_synthetic
desc: ''
updated: 1790562913084
created: 1790562913084
---

## 2026.09.27 - The Nadal-Ribelles 2025 loader on tiny .Rdata files

Seven tests; the `rdata` package the loader reads with also writes (`rdata.write_rda`), so the full build runs on tiny `.Rdata` files (not validated against R, which is not installed). Loader coverage 95%. Finding: a gene with no standard name is stored as `perturbed_gene_name="nan"`, the string, instead of falling back to the ORF (nadal_ribelles2025.py lines 244 and 344); the genome attribute table gives NaN for a missing standard name, and the audit measured 940 of 6188 records in the dev LMDB carrying "nan". Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.

## 2026.10.01 - Fix PR for the pinned findings

The control `ptbs` table no longer repeats `bc-YBR020W-1` (repeats are now refused, issue #541); the normalized index has three labels.
