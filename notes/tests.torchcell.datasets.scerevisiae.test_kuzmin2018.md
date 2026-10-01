---
id: 55as547z8iesri6hlxxgdi8
title: Test_kuzmin2018
desc: ''
updated: 1790769099565
created: 1790769099565
---

## 2026.09.30 - Phase 15: the five loaders on a four-row table, no file-level skip

Three to twelve tests; the file used to skip whole at import (the import-time `load_dotenv()` and module skip are gone, the data gate sits on the three mirror tests), so alone it goes from 0 to 98 percent. Fixture: an allele query (`cdc28-4`), a blank SD, the allele in slot 1 and slot 2, ts and KanMX arrays. Smf records with the NaN-skipping reference SD (0.05 + 0.02 + 0.04)/3 and se half; Dmf allele digenic crosses and the tm801/tm802 query pairs with reference SD 0.05, `sample_sd` n = 4, se 0.025; Tmf se 0.01 and 0.02 with reference 0.03; Dmi and Tmi edge and hyperedge records against zero references; the unknown-array exception for all five loaders; a repeated digenic row stored twice by Dmf and Dmi and deduplicated by Smf; `subset_n` under seed 42 keeping positions [1, 3] and [1] with the reference SD from the full frame; `download` for all five; `transform_item` round trips.

Findings: Smf raises `UnboundLocalError` on an unknown array type (lines 336-370; 350 is dead) while the other four stop at the genotype assertion and the 2020 loaders store that strain as an allele; a blank SD is stored as `fitness_std` NaN (638); Dmf and Dmi store a repeated row twice.

## 2026.10.01 - Issue #533 Findings Retired

- Retired: Smf `UnboundLocalError` on an unknown array type, blank Dmf SD stored as NaN, repeated digenic cross stored twice by Dmf and Dmi.
- Now asserted: all five loaders refuse `YCR002C_sn77` with the exact `ValueError` and write no LMDB; a blank SD is stored as `fitness_std` None; Dmf and Dmi refuse a repeated cross naming the pair; Smf still keeps one single per allele.
