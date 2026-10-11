---
id: ujpubwp7cfxrl6uug1t1rrv
title: Natural_isolate_response
desc: ''
updated: 1791716761654
created: 1791716761654
---

## 2026.10.11 - A verifier family for natural-isolate genotypes

Module: `torchcell/verification/natural_isolate_response.py`, runner `runners.run_natural_isolate_response` over `NATURAL_ISOLATE_RESPONSE_DATASETS` (one entry, `isolate_growth_peter2018`), per-dataset entry `runners.verify_natural_isolate_dataset` (what the loader's `verify_build` calls). Results and the measured table: [[torchcell.datasets.scerevisiae.peter2018]].

A natural isolate's genotype is ~5,000 perturbations off S288C shared by every condition it was grown on. The environment-response verifier assumes a screened deletion: its strain key omits `strain_id` and its L4 rules require every systematic name to be a current S288C gene, which an accessory pangenome ORF is not. This module reuses that verifier's per-record pieces (`_condition_signature`, `_value_problem`, `_ratio_reference_accumulator`, `_environment_perturbed_result`) and `SharedRecordRules`, and replaces the genotype rules:

- L0 validates each interned genotype ONCE (records of one isolate share one resolved object) and splices the instance into every record's validation.
- L1 keys uniqueness on (isolate, condition) and checks one genotype and one `strain_id` per isolate.
- L2 `release_fidelity` re-reads the release (round-trip floats) and compares every cell; `genotype_round_trip` re-derives present and absent ORFs and copy numbers from Peter's matrices and checks every variant pointer.
- L3 `conditions_documented` re-reads Supplementary Table 2 from the mirrored workbook; `sourced_values` audits every quote.
- L4 splits containment: S288C-keyed names against R64-4-1 (floor `MIN_PETER_S288C_CONTAINMENT = 0.999`), accessory ORFs against the pangenome, and the release md5 against the host's `md5.txt`.

`SharedRecordRules` sees the full genotype on the first record of each isolate and an empty perturbation list on the other 34, so the census and spelling rules run once per isolate instead of 35 times. Tests: `tests/torchcell/datasets/scerevisiae/test_peter2018_synthetic.py` (a two-isolate store, every row passing, then a changed release cell, a wrong tarball pin and a missing gene each caught), `tests/torchcell/verification/test_runners.py` (runner wiring).
