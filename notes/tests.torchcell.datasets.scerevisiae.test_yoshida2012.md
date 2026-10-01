---
id: f06ltzcq6456b284adck1ud
title: Test_yoshida2012
desc: ''
updated: 1790550009510
created: 1790550009510
---

## 2026.09.27 - The Yoshida 2012 organic-acid loader from its embedded Table 3

The only raw file is `paper.pdf`, whose presence is all PyG checks, so a placeholder is written and the values come from the module-level `TABLE_3` literal; `build_metabolite_s_id_map` (which loads Yeast9) is replaced by a stub that records its argument and returns `s_0001` to `s_0005` for the five acids. Five tests: the records with their organic-acid values and metabolite ids, the reference, the side files, the download paths. Loader coverage from this file 92% (the mirror-copy branch at lines 352 to 353 and `main()` remain). Finding: `download()` returns early without hashing an already-present raw file although its docstring says the sha256 is verified (yoshida2012.py line 334). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Phase 16: both ledger lines, the analyte-restricted reference, refusals

Five to twelve tests, 92 to 100 percent. Both ledger lines; the reference restricted to the analytes a strain measured, `target_metabolite_ids` None when only phosphate is present; a negative SD refused with "SE for acetate must be non-negative"; `download` and `main`.

Findings: a name shaped like a systematic ORF skips the genome, so a nonexistent YAL999W is stored (line 358); an alias with two candidate ORFs silently takes the first (364); the same gene written by common and by systematic name gives duplicate genotypes; a NaN cell is stored as a NaN level and SE.

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Finding retired: the present-PDF Finding in `test_download_stages_the_mirror_pdf_only_after_verifying_it` now asserts the build-time refusal. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.
