---
id: oxqn4lof2hxtyjxpm6s8d24
title: Test_ohya2005
desc: ''
updated: 1790549962179
created: 1790549962179
---

## 2026.09.27 - The Ohya 2005 CalMorph loader on two synthetic SCMD matrices

Both TSVs are written into `<root>/raw/`; the genome stub implements only `resolve_gene_name` and returns real `GeneNameResolution` objects (YAL001C, YDR001C, YER001W, YFR001W current; YBR002C renamed to YBR001C; YER002W renamed to YER001W, which is also a strain, so both keep their names; YCR001W retired). Four tests: the records with their CalMorph parameter dicts, the renamed and retired handling, the side files, the download refusal. A missing CalMorph cell drops the row (ohya2005.py line 265), which [[tests.torchcell.datasets.scerevisiae.test_ohnuki2018]] contrasts with the 2018 loader storing 0.0. Loader coverage from this file 94% (the mirror-copy branch at line 179 and `main()` remain). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Phase 16: the ledger, the truncated warning, refusals

Four to fourteen tests, 94 to 100 percent. The ledger; the warning truncated at 20 names; a duplicate-spelling matrix; a non-numeric cell refused; `default_genome()` called once; both files copied.

Findings: `TCV` in `_CV_PREFIXES` (line 105) is not a real prefix, so a TCV column is routed to the CV traits and refused by the schema; `create_experiment` is a bare `pass` returning None (292); duplicate spellings give two records; the publication is Ohya 2005 (PMID 16365294) while the 2026.09.29 verification (issue #491) records the matrices as the Suzuki 2018 CalMorph 1.2 re-analysis.

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.
