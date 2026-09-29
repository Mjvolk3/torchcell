---
id: f5ojvpnzaiqu665d034flhq
title: Test_rnaseq
desc: ''
updated: 1790648010750
created: 1790648010750
---

## 2026.09.28 - The RNA-seq verifier on hand-built records (Phase 9)

10 tests. A TPM dataset emits seven results in order and passes; a pseudobulk dataset skips the count check and allows negatives; strain uniqueness keys on strain and condition (full details dict, including `n_missing` for a record without a strain); value fidelity indexes the flattened value list in the dumped dict's insertion order, which the schema's SortedDict makes sorted-gene order; count fidelity rejects bool, negative and float counts; an empty dataset defaults to the TPM family and passes a zero oracle (`"single measurement_type: None"`). Finding pinned: the record family is decided by `records[0]` alone (rnaseq.py line 168), so a TPM-first dataset mixed with a pseudobulk record raises `KeyError('expression_count')` at line 197 before the L3 `measurement_type_consistent` check at line 215 can report the mix; only pseudobulk-first mixing is reported. Coverage 0% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]].
