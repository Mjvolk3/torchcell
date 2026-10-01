---
id: dngx2g3207ne0nafj943syh
title: Test_dcell_opt
desc: ''
updated: 1790820773223
created: 1790820773223
---

## 2026.09.30 - DCellOpt root key and paper loss (issue #554)

New. `DCellOpt` on the conftest hierarchy (seed 0, B = 3) declares `root_key` GO:0; its prediction and GO:0 are equal but distinct tensors. The head values are pinned to 1e-6 and the loss is the paper value 0.4857438 + 0.3 * (0.3546567 + 0.3064786) = 0.6840844, not the 0.8298076 an identity-based root skip gave (issue #554 review).
