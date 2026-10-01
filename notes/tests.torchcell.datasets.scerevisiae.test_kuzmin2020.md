---
id: vxe9okxo4d2q096fikdmshf
title: Test_kuzmin2020
desc: ''
updated: 1790769107170
created: 1790769107170
---

## 2026.09.30 - Phase 15: the S5-over-S1/S3 rule and the rest, no file-level skip

Four to thirteen tests, 0 to 95 percent alone (100 with the siblings); the import-time gate is gone. Smf allele single with a blank SD; Dmf allele query times unknown array stored as `SgaAllelePerturbation`, tm801 in S1 and S3 giving one record, Table S5's 0.5 / 0.006 winning over S1/S3's 0.55 labeled `bootstrap_se` n = 12 with the exact INFO and WARNING lines including "(max |diff| 0.0500)"; Tmf, Dmi and Tmi records with reference SD 0.025; `subset_n`; `download` for five; `transform_item` for five; `main` with exact stdout.

Findings: a blank SD becomes NaN in Smf (line 339) and Dmf (627); a duplicated Table S5 "Double mutant" row doubles that strain's record (merge at 173); `main` builds only Tmi at a root relative to the working directory (1384).

## 2026.10.01 - Issue #533 Findings Retired

- Retired: blank SD stored as NaN on Smf and Dmf, the duplicated Table S5 "Double mutant" row doubling a strain's record, `main` building only Tmi at a relative root.
- Now asserted: blank SDs stored as None; the fixture's digenic row uses the ts array `YCR002C_tsa1` and `YCR002C_sn1` is refused by Dmf, Dmi, Tmf and Tmi with the exact message; a repeated S5 tm number is refused; `main` builds all five under `$DATA_ROOT/data/torchcell` with nothing in the working directory, and refuses an unset `DATA_ROOT`.
- Review follow-up: added `test_tmf_blank_sd_is_stored_as_none` and `test_array_strain_type_is_matched_case_sensitively` (both years; `YBR001C_TSA100` and `YAL048C_DMA5203` refused with the exact message); the repeated S5 row now differs in fitness; the unknown-array test asserts no LMDB is written; `test_main_refuses_an_unset_data_root` runs in `tmp_path` with `download_url` stubbed and asserts nothing is downloaded or created; renamed `test_dmf_allele_query_ts_array_and_s5_disagreement`.
