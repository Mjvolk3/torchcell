---
id: 412xeepxmr8khx9v1qud4m7
title: Test_losses_dcell
desc: ''
updated: 1790818903312
created: 1790818903312
---

## 2026.09.30 - Sum, declared root key and shape refusal asserted (issue #554)

The mean-pinning assertions are retired. Now asserted on predictions [1, 2], target [0, 0], GO:1 = [0, 0], GO:2 = [2, 2], root key GO:0: `"sum"` gives 3.7 (auxiliary 4.0, weighted 1.2); `"mean"` gives 3.1 (2.0, 0.6), named for what it pins, the mean's closed form, not a reproduction of old runs; alpha 1.0 gives 6.5; `aux_reduction` is required and keyword-only; an unknown reduction raises the exact `ValueError`; a declared root that is a distinct object is skipped while GO:7, equal in value, is counted (3.25); `linear_outputs` without `root_key` raises; a `[2, 1]` head or prediction against a `[2]` target raises; GO:2's gradient is 0.6 under the sum and 0.3 under the mean.
