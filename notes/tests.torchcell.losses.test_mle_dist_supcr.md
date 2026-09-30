---
id: rlzvmv7vue4iaxvg4izp57e
title: Test_mle_dist_supcr
desc: ''
updated: 1790765123641
created: 1790765123641
---

## 2026.09.30 - Phase 14: every term on targets [0, 1, 2] and predictions [2, 0.5, 1]

Six to twenty tests, 84 to 99 percent (branch 639 -> 638 can never be taken); five existing tests that checked keys, shapes or ranges now pin exact values. MSE 1.75; the distribution term 1.25/3 on three samples (whose target labels all collapse to 1) and 0.5/6 on the doubled batch with labels [0, 0, 1, 1, 2, 2]; SupCR (log(1 + e^(-1/T)) + log 2)/3; total 1.7920021; gradient exactly plus or minus 1.1; the buffer wrap-around contents, the seeded buffer subsample, the gathered-batch path, a `torch.distributed` stand-in at world size 2 recording each `all_gather`.

Findings: an unknown temperature schedule silently holds the initial temperature (line 72); a switched-off term logs `zeros(2)` whatever the real width (562-634); at buffer weight 1 the current batch is also read back from the buffer and counted twice (185, 220); the buffered SupCR factor 1 - 0.5w scales the whole loss, not only the buffer rows (373-375).
