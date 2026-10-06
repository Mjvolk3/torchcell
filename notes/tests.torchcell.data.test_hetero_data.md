---
id: hkbqcrsy20v6f7op46schax
title: Test_hetero_data
desc: ''
updated: 1791269287967
created: 1791269287967
---

## 2026.10.06 - Phase 21 tests

- Every `custom_size_repr` branch as a literal string; `SparseTensor` / `TensorFrame` driven by stand-ins (torch_sparse and torch_frame are not installed).
- `hetero_repr`: global, node, then edge stores at indent 2.
- Finding (hetero_data.py:26, 28): the `dict(len=n)` early return keeps tuple-key quotes, `('a', 'b')=dict(len=1)`, while other paths print `(a, b)`.
