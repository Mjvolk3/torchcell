---
id: 9tjqh9nd9nhf2rorfhq88go
title: Test_lopez2024
desc: ''
updated: 1790481903073
created: 1790481903073
---

## 2026.09.26 - Hermetic build of both Montaño López 2024 loaders over a synthetic workbook

No mirror and no real genome: `supplementary_tables.xlsx` is written with openpyxl into `<root>/raw/` (Table S2 header on 0-based row 1, Table S3 header on row 2 with the UP block in columns A-C and the DOWN block in E-G, so pandas suffixes the repeated names with `.1`), PyG therefore never calls `download()`, and `process()` runs against a stub genome that carries only `gene_attribute_table` (ID/gene) and `alias_to_systematic`, the two attributes `_resolver` and `_standard_name_map` read. Stub: current ORFs YAL001C (TFC3), YBR001C (NTH2), YCR001W (unnamed), YDR001C (NTH1); aliases YAL002W -> YAL001C, YBR002C -> YBR001C, YDR002W -> YDR001C, YER001W -> YER999W (not current).

Screen (Table S2): nine rows `yal001c` 1.5, `YAL001C` 2.5, `YBR001C` 0.5, `YCR001W` 4.0, `YBR002C` 0.8, `YDR002W` 0.25, `YER001W` 9.0, `YFR001W` 1.1, `ABC1` 1.2 give four records in first-sighting ORF order: YAL001C mean 2.0, n 2, SE = stdev([1.5, 2.5]) / sqrt(2) = 0.5 exactly; YBR001C 0.5; YCR001W 4.0 with the ORF as its perturbed name; YDR001C (NTH1) 0.25, the last three n 1 with SE None. Dropped by rule: YBR002C (alias target directly present, so it is not merged and YBR001C keeps n 1), YER001W (target not a current ORF), YFR001W (no alias entry), ABC1 (not systematic). Full `model_dump()` equality on record 0 (SC liquid synthetic, 30 C, aerobic, `biosensor_gfp_fluorescence_fold_change`, PMID 35022416), `gene_set.json` the four ORFs sorted, `build_manifest.json` with slug `isobutanol_screen_lopez2024`, loader class/module, this hostname and the worktree HEAD, closure containing `MetaboliteExperiment` and `KanMxDeletionPerturbation`.

Validated (Table S3): UP `YAL001C` 3.0/0.3, `YBL071W-A` 3.469/0.1, `YDR002W` 2.2/0.6 and DOWN `ybr001c` 0.25/0.06, `YBL071W-A` 0.0757/0.01, `YCR001W` 0.4/0.09, `YER001W` 0.3/0.05 give four triplicate records with SE = STD / sqrt(3) (0.3 / sqrt(3) = 0.1732 for YAL001C), UP before DOWN; YBL071W-A is dropped from both blocks before resolution and YER001W is unresolved. One shared reference with n 3, index entry [0, 1, 2, 3]. A workbook with `YAL001C` in both blocks raises `ORF YAL001C (from YAL001C) appears twice after resolution`.

Error paths: `genome=None` raises naming the loader class for both datasets; with no raw file and `DATA_ROOT` pointed at a tmp dir, `download()` raises `library mirror data file not found: <path>` and, given a mirror file holding `b"not the real workbook"`, copies it into `raw/` and raises `sha256 mismatch: got 55b4751d... expected f97cf13c...`.

Finding (pinned, not a contradiction of the docstring): the screen rebuilds the WT reference per record with `n_replicates` equal to that record's row count, so the dataset holds one reference per distinct n. With one aggregated gene and three singletons `experiment_reference_index.json` has two entries, member indices [0] and [1, 2, 3], both FC 1.0 with SE None.

Not covered, with reason: `main()` (needs the real genome and `DATA_ROOT`), the `download()` success log (needs the real sha256-pinned workbook), and the `preprocess_raw` / `create_experiment` stubs. `coverage run` over this file: 26.0% -> 94% of `lopez2024.py` (uncovered: 161->176, 181, 299, 303, 512-532). Seven tests; Phase 5 of [[plan.test-suite-buildout.2026.09.25]].

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.
