---
id: ljv9yyix6cmm641gwgqa015
title: Test_isomorphic_cell_loss
desc: ''
updated: 1791269879455
created: 1791269879455
---

## 2026.10.06 - Phase 21: ICLoss zero totals and ICLossStd

Fixture as in the test file docstring (predictions [[1, 0], [0, 1], [2, 2]], targets [[0, 1], [1, 0], [2, 3]], MSE dims [2/3, 1]). ICLossStd's own SupCR call cannot run, so the closed-form tests swap `supcr_fn` for a stub returning 0.3 and per-dimension [0.2, 0.4] and set lambda_dist 0, lambda_supcr 0.5: base task losses [2/3 + 0.1, 1.2].

- ICLoss: all-NaN targets make every total 0 and every `norm_*` entry the int 0; with predictions equal to targets only the weighted total is 0.
- ICLossStd closed form at sigma 2, task weights [1, 3], lambda_reg 0.01: weighted fitness 0.788981, gi 1.143148, reg 0.005, total 1.937129; every logged key and value pinned; prediction std skips NaN targets and reports 0.0 for an empty column.

Findings:

- `ICLossStd.forward` always raises `TypeError: WeightedSupCRCell.forward() takes 3 positional arguments but 4 were given` (isomorphic_cell_loss.py:181); its one caller is commented out (hetero_cell_pma.py:1068).
- Even with the call fixed, `torch.tensor([...])` (line 190) cuts the graph: predictions get no gradient and only log_sigma trains (gradient [0.803333, 0.095]).
