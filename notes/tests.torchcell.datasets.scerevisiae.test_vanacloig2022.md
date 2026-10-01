---
id: gon1ixezx4fwjr6ls8blmz2
title: Test_vanacloig2022
desc: ''
updated: 1790762190983
created: 1790762190983
---

## 2026.09.30 - Phase 13: a gzipped GEO matrix with columns summing to one million

Ten to twenty-nine tests, 77 to 97 percent. A filler row brings every column total to 1e6 so CPM equals the raw count; responses worked by hand (YAL001C/Furfural paired 3.0 with SD 1.0, YAL001C/MMS pooled 2.0 with SD 1.0, YAL002W/Furfural 1.0 with SD 1.0, SE 1/sqrt(3)); `model_dump()` equality for the paired full record, the pooled MMS record and the barcodeless record; the ledger (48 source records, 7 kept, 41 dropped, each rule's count and items); the reference index paired {0, 1, 2, 5, 6} versus pooled {3, 4}; refusals (an unparseable column, no control columns, 2 replicates instead of 3, two barcodes for one ORF, the manifest with no record, a missing mirror file, a differing digest, no manifest); the deposit manifest, `manifest_sha256`, `_canonical_common_names`.

Findings, citing issue #501: the stored value is CPM against every released row, not the paper's TMM (lines 865-868); SodiumGlyoxylate, never reported by the paper, is served; DMSO is dropped as a vehicle and the environment of a DMSO-delivered compound names no DMSO; MBO is dropped. New: a single NaN count drops the whole row although the rule says "every count column is missing" (760 against 811); a row with no barcode is served with `barcode ""` (758); equal nonzero replicates keep an SD of exactly 0 (904).

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.
