---
id: duzf50kqcpaj5dsu9t14fs3
title: ci-green
desc: ''
updated: 1789849385690
created: 1789849385690
---

## 2026.09.19

- [x] CI red since mid-July, not since today: pytest failed to collect 17 adapter test files because `torchcell/graph/sgd.py` raised at import without `DATA_ROOT` (runner has no `.env`); ruff format flagged `tests/torchcell/verification/test_sourced.py`. Data root read at use time now; file formatted. [[torchcell.graph.sgd]]
