---
id: 5213bb91x9wg943fg4kkqpn
title: Test_linear
desc: ''
updated: 1791269262769
created: 1791269262769
---

## 2026.10.06 - Phase 21 tests

- Weight `[[1, 2]]`, bias 0, `x = [[1, 2], [3, 4], [5, 6]]`, `batch = [0, 0, 1]`: add gives `[16, 17]`, mean `[8, 17]`.
- Finding (linear.py:37): `scatter="max"` passes the `(values, argmax)` tuple to `nn.Linear` and raises `TypeError`.
- Finding (linear.py:32-37): an unknown mode skips aggregation, output `[5, 11, 17]` per node.
- `main` is a demo and left uncovered.
- Reach (audit 1): both findings are latent. Audit 1 revision: the default-scatter test now pins the summed output `[[16], [17]]` instead of the attribute.
