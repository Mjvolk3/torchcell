---
id: 412xeepxmr8khx9v1qud4m7
title: Test_losses_dcell
desc: ''
updated: 1790818903312
created: 1790818903312
---

## 2026.09.30 - Sum reduction and key-only root skip asserted (issue #554)

The mean-pinning assertions are retired. Now asserted on predictions [1, 2], target [0, 0], GO:1 = [0, 0], GO:2 = [2, 2]: the default is alpha 0.3 with `aux_reduction="sum"`, total 3.7 (auxiliary 4.0, weighted 1.2); `"mean"` gives 3.1 (2.0, 0.6); alpha 1.0 gives 6.5; an unknown reduction raises the exact `ValueError`; GO:ROOT and its object alias GO:9 are skipped while GO:7, a distinct tensor equal to the root, is counted (3.25, where the old `torch.equal` test gave 2.5); GO:2's gradient is 0.6 under the sum and 0.3 under the mean.
