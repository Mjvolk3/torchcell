---
id: 1zqrjdj8afa694esovxe6g0
title: Test_xue2025
desc: ''
updated: 1790481910580
created: 1790481910580
---

## 2026.09.26 - Hermetic build of the Xue 2025 FFA loader over a synthetic two-sheet workbook

`Supplementary Data 1_Raw titers.xlsx` is written with openpyxl into `<root>/raw/`: an `Abbreviations` sheet with the ten (gene, code) rows copied from `_EXPECTED_CODE_TO_GENE`, and a `raw-titer (mg-L)` sheet whose row 0 is a label row and whose columns 1-15 are the five FFAs (C14:0, C16:0, C18:0, C16:1, C18:1) times three replicates. PyG never calls `download()`; `process()` runs against a stub genome with `gene_set` and `alias_to_systematic` only (POX1 YGL205W, FAA1 YOR317W, FAA4 YMR246W, FKH1 YIL131C, GCN5 YGR252W, MED4 YOR174W, OPI1 YHL020C, RFX1 YLR176C, RGR1 YLR071C, RPD3 YNL330C, SPT3 YDR392W, YAP6 YDR259C, TFC7 YOR110W; the stub is the authority).

Rows, in sheet order with the WT second so the reference is found by label: `+ve Ctrl` [100,110,120] [200,200,200] [50,51,52] [8,10,12] [300,310,320]; `wt BY4741` [10,12,14] [20,20,20] [5,6,7] [1,2,3] [30,40,50]; `F-G 5d` with the third replicate blank everywhere, [1,3] [10,14] [5,5] [2,4] [7,9]; `P-S-Y 6dΔ` (delta glyph) all-constant triples; `T 4d` a single replicate 3, 4, 5, 6, 7. Four records: the positive control is the chassis triple sorted by ORF (YGL205W POX1, YMR246W FAA4, YOR317W FAA1) with means 110, 200, 51, 10, 310 and SE 10/sqrt(3), 0, 1/sqrt(3), 2/sqrt(3), 10/sqrt(3) (sample SD, n 3); `F-G 5d` decodes to the triple plus GCN5 and FKH1 with n 2 and SE = |a - b| / 2 = 1, 2, 0, 1, 1; `P-S-Y 6dΔ` to six deletions (YAP6, SPT3, POX1, FAA4, RPD3, FAA1 in ORF order); `T 4d` to four with n 1 and every SE NaN. Every record shares the WT reference (means 12, 20, 6, 2, 40; SE 2/sqrt(3), 0, 1/sqrt(3), 1/sqrt(3), 10/sqrt(3)), one index entry [0, 1, 2, 3]. `preprocess/data.csv` holds label, deletion count and `;`-joined sorted ORFs per record; `gene_set.json` is the nine ORFs sorted; the manifest names slug `ffa_xue2025`, the loader and this host. Environment `SC (FFA production)` liquid synthetic, 30 C, aerobic; PMID 23899824; `titer_mg_per_l`; `target_metabolite_ids` None.

Unit pins on the built instance: `_resolve_systematic` returns a `gene_set` member as is (whitespace stripped), upper-cases a common name for the alias map (`pox1` -> YGL205W), raises on `NOPE`; `_decode_genotype` maps `wt BY4741` and `BY4741` to [], `+ve Ctrl` to the triple, `G-O-T 6d` to the triple plus GCN5, OPI1, TFC7, strips the other delta code point (`M 4d∆`), raises `TF letters (2) + baseline (3) != declared deletion count 6` on `F-G 6d`, and raises `unparseable genotype string 'weird' (core 'weir')` because the strip set includes `d`. Build-time errors: no WT row (`found 0`), both WT labels present (`found 2: ['wt BY4741', 'BY4741']`), an Abbreviations sheet missing the TFC7 row (message carries the nine-entry map read), an `M 4d` row with all three C18:1 cells blank (`FFA C18:1 has no replicate values in row 'M 4d'`), `genome=None`, and the two `download()` failures (missing mirror file; `b"not the real workbook"` copied then rejected with its sha256 55b4751d... against the pinned 023de80e...).

Finding (pinned as the code behaves): the `gene_set` membership check in `_resolve_systematic` is case-sensitive while only the alias lookup upper-cases, so a lowercase systematic name (`ygl205w`) is unresolvable and raises. No real record is affected because the sheet carries common names only.

Not covered, with reason: `main()` (needs the real genome and `DATA_ROOT`), the `download()` success log (needs the real sha256-pinned workbook), and the `preprocess_raw` / `create_experiment` stubs. `coverage run` over this file: 27.9% -> 94% of `xue2025.py` (uncovered: 199->214, 219, 418, 422, 427-439). Twelve tests; Phase 5 of [[plan.test-suite-buildout.2026.09.25]].

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.
