---
id: x0wje8reqxrkvsq369ekl1o
title: Test_cachera2023_synthetic
desc: ''
updated: 1790549970018
created: 1790549970018
---

## 2026.09.27 - The Cachera 2023 CRI-SPA betaxanthin loader on a synthetic CSV

`GA1_2_4_6.csv` is written into `<root>/raw/`; the genome stub implements `resolve_gene_name` and `feature_index["standard_to_ids"]`, which `canonical_common_names` reads (current ORFs YAL001C TFC3, YBR001C NTH2, YCR001W without a standard name, YDR001C NTH1, plus aliases). Five tests: records by `model_dump()` equality with the betaxanthin statistics, the reference, the side files, the download refusal. Loader coverage from this file 81%; the `urlopen` branch (lines 183 to 199) and `main()` remain. Findings: `gene_set.json` lists the cassette genes `CYP76AD1`, `DOD`, `YBR249C` and `YPR060C` beside the deletions (cachera2023.py line 336), and a count of one or a blank std stores the SE as `nan` inside the dict rather than `None` (lines 258 to 262). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.
