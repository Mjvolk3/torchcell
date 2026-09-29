---
id: 4x2u3vgeiwndebqxilc3c79
title: Test_cosine_annealing_warmup
desc: ''
updated: 1790648927517
created: 1790648927517
---

## 2026.09.28 - The warmup cosine schedule as a closed-form table (Phase 9)

11 tests. Fourteen learning rates over three cycles are asserted to 1e-12 (cycle 0: 0.055, 0.1, 0.0775, 0.0325; cycle 1 of length 8 with max 0.05: 0.01, 0.03, 0.05, 0.047320..., 0.04, 0.03, 0.02, 0.012679...; cycle 2 of length 14 with max 0.025: 0.01, 0.0175), plus restart bookkeeping, every parameter group, the explicit-epoch path inside the first cycle and with `cycle_mult=1`, a negative explicit epoch, and the warmup-shorter-than-first-cycle check. Findings, recomputed by the audit: the explicit-epoch path (lines 99 to 124) computes the cycle length as `first * mult ** n`, ignoring warmup, while implicit stepping uses `int((cur - warmup) * mult) + warmup`, so reaching epoch 8 the two ways gives 0.047320508 against 0.048477591; `step(epoch=3)` after `step(epoch=8)` keeps `cycle=1`, so the maximum stays decayed (0.04, not 0.0775); `step` never sets `_last_lr`, so `get_last_lr()` raises `AttributeError` (line 86, torch 2.11). Coverage 65.7% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]].
