---
id: aiy2xr8zwzaubf788ha8ans
title: Natural_isolate_genotype
desc: ''
updated: 1791808466224
created: 1791808466224
---

## 2026.10.12 - The isolate-genotype rules every natural-isolate store shares

`IsolateGenotypeAudit` takes the builder of [[torchcell.sequence.natural_isolate]] (over the store's own isolates, reading the store's allele-diff cache) and the library-mirror root. Each verifier feeds it every DISTINCT stored genotype once and adds its four results:

| level | rule | what it checks |
|---|---|---|
| L1 | `isolate_genotype_identity` | every genotype carries `natural_isolate`; every perturbation that names a strain names that isolate; one genotype per isolate; every builder isolate stored |
| L2 | `isolate_genotype_round_trip` | the stored dump equals the builder's `genotype(isolate).model_dump()` |
| L4 | `isolate_descriptor_matches_table_s1` | manipulation, ploidy, zygosity and copy changes against Table S1 re-read from the library mirror's `si/si3.xls` (sha256 `b11de13b...`), and the HO deletion present exactly where Table S1 says yes |
| L4 | `loss_of_function_calls_match_release` | per isolate and cause, the stored calls are exactly the rows with a 1 in the release matrix, row for row by the matrix's own key (`NAME` in the loss-of-function matrix, `Gene` in the frameshift matrix), re-read with pandas and not through the builder's row logic |

Callers: `natural_isolate_response.verify_natural_isolate_response_streaming` (Peter 2018), `protein.verify_natural_isolate_proteome` / `verify_natural_isolate_turnover` (Muenzner 2024), `dechiara2022.verify_build`, `runners.verify_rnaseq_store` for an entry marked `natural_isolates` (Caudal 2024). Tests: `tests/torchcell/verification/test_natural_isolate_genotype.py` (the synthetic five-isolate panel of `test_natural_isolate.py`: all pass; a missing descriptor, a foreign strain, a repeat; a dropped call and a call re-keyed to an S288C name, each caught by the round trip and by the release keys; a library Table S1 that disagrees, and one off its pin).
