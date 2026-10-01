---
id: y81x3i4rwanjs6tgngd8tbv
title: Test_fitness
desc: ''
updated: 1790648019653
created: 1790648019653
---

## 2026.09.28 - The fitness verifier on hand-built records (Phase 9)

10 tests. A good dataset emits twelve results in order and passes; SGD genes add the L4 containment rows and the floor is forwarded; the resolver is forwarded to the canonical-name rule, whose failing message and `unresolved_common_names == ["YJR155W (retired; stored YJR155W)"]` are pinned exactly; pair uniqueness keys on strain id and environment; value fidelity skips `None` and then indexes the filtered list (fitness.py lines 147 to 151); the SE rule skips NaN and None and flags a negative; the reference-one rule reports the worst deviation (`0.02`); the uncertainty-sanity message is pinned verbatim from `common.py`, including its spelling "labelled". Coverage 19.7% to 99%; the one partial branch (line 190) is a perturbation without a systematic name, unreachable for schema-valid records. Phase 9 of [[plan.test-suite-buildout.2026.09.25]].

## 2026.10.01 - American spelling (issue #541)

The `uncertainty_sanity` messages are asserted with "labeled".

## 2026.10.01 - Review: no gene perturbations fail the report

`test_a_dataset_with_no_gene_perturbations_fails_for_no_measured_genes` runs `verify_fitness_dataset` on three wild-type records at three temperatures and on one wild-type record: `measured_genes_present` is the only failing result and `report.passed` is False.
