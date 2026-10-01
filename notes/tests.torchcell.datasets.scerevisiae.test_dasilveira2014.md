---
id: 47bc0lvqesnqoqei6vvxdt2
title: Test_dasilveira2014
desc: ''
updated: 1790549954207
created: 1790549954207
---

## 2026.09.27 - The da Silveira dos Santos 2014 lipidome loader on synthetic tables

Both workbooks are written with openpyxl into `<root>/raw/`; the genome stub carries the two attributes the loader reads, `gene_set` (YAL001C, YBR001C) and `alias_to_systematic` (YBR002C to YBR001C, YAL003W to YAL001C). Five tests: records by `model_dump()` equality with the lipid values, the reference, the side files, the alias resolution, the download refusal on the real digest. Loader coverage from this file 92%; the successful mirror-copy branches (lines 183, 189, 195) need the real sha256-pinned files and `main()` a real `DATA_ROOT`. Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Phase 16: the four ledger lines, name stripping, the all-blank refusal

Five to twelve tests, 92 to 100 percent. The four ledger lines including the duplicate-ORF warning; name stripping; a lipid no wild-type row measured; the empty unresolved suffix; the all-blank mutant refusal; both workbooks copied; `main`.

Findings: a lipid no wild-type row measured has no reference entry (lines 246, 361); an all-blank mutant raises only after the store is open (305), leaving `data.csv` with `n_lipids` 0 and a retry that serves 0 records; `gene_set` membership is case-sensitive (201).

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.

## 2026.10.01 - Findings retired (issues #537, #546)

Retired the Phase 16 findings: a lipid no WT row measured and an all-blank mutant row are refused with exact messages and leave no `data.csv` and no store; padded and lowercase systematic names resolve (`ybr001c` -> YBR001C) with the full three-lipid reference.
