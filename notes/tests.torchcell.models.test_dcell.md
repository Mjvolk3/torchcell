---
id: z1nx20s3189k4kytu3cur7e
title: Test_dcell
desc: ''
updated: 1790413687880
created: 1790413687880
---

## 2026.09.26 - DCell on the three-term conftest ontology

Replaces the docstring-only placeholder. The fixture is the root term with two stratum-1 children over four genes from `tests/torchcell/conftest.py`; with `min_subsystem_size=2, subsystem_ratio=0.5` the paper formula gives every subsystem 2 units, input dims {leaf: 2, root: 2 + 2 + 1} and exactly 45 parameters (36 in the three Linear+BatchNorm subsystems, 9 in the three heads), all pinned. Forward returns one prediction per sample as the root head's output (shape `[batch]`, not `[batch, 1]` as the plan sketched), `GO:ROOT` and `GO:0` are the same tensor, and knocking out gene 0 changes only that sample's prediction in eval mode. Two findings recorded as pinned behavior: the prediction trains every subsystem and the root head but no leaf head (the leaf heads feed only `DCellLoss`'s auxiliary term), and a one-sample batch cannot pass a training-mode `BatchNorm1d`. Phase 2 of [[plan.test-suite-buildout.2026.09.25]]; `#321` is open against this model.

## 2026.09.27 - Seven more tests on the remaining branches

Seeded determinism, a gradient reaching every parameter, the remaining `ValueError` paths with their exact messages, and the gene-less childless term. Module coverage from this file 48%; the rest is `main()` (hydra plus a real LMDB, lines 420 to 834) and the "Root term was not processed" guard (line 343), which cannot fire because every term in stratum 0 is processed first. Finding: the "no children and no genes" `ValueError` (line 386) is unreachable; such a term gets a `[batch, 1]` zero placeholder and runs (pinned at 37 parameters with activation `subsystem_0(zeros)`). Phase 7 of [[plan.test-suite-buildout.2026.09.25]].
