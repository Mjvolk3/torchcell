---
id: 6rruvqgm0ptt1sbik2wnkz5
title: Test_mlp
desc: ''
updated: 1791269255181
created: 1791269255181
---

## 2026.10.06 - Phase 21 tests

- Layer stacks compared as descriptors; `Mlp(4, 3, 2, 3, 0.25, batch, relu, sigmoid)` has 15 + 6 + 12 + 6 + 8 = 47 parameters.
- Closed form: `W1 = [[1, -1], [2, 0]]`, `b1 = [0, -1]`, `W2 = [[1, 1]]`, `b2 = 0.5` gives `[1.5, 7.5]` for `x = [[1, 2], [3, 1]]`.
- Finding (mlp.py:71-72): the dropout follows the final Linear (docstring: before it); with p = 1 every training output is 0.
- Finding (mlp.py:62-65): `num_layers=1` ignores `dropout_prob`.
- `output_activation` is not validated (bare `KeyError`).
- Reach (audit 1): the `num_layers=1` finding REACHES the smf-dmf-tmf-001 deep_set sweeps, which pair `dropout_prob` [0.0, 0.2] with `num_layers` [1] (deep_set.py:303-312; e.g. sweep_1e03_17 and _1e04_one_hot_gene_00 to_06): the 0.2 arm was a no-op in the head. The dropout-placement finding is latent.
