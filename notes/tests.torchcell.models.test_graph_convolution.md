---
id: pzfmhf12j1glnne07e7yy91
title: Test_graph_convolution
desc: ''
updated: 1790780596681
created: 1790780596681
---

## 2026.09.30 - Phase 18: a class that cannot be built as shipped

New file, eleven functions (12 cases), 14 to 72 percent alone (`main()` stays). Finding first: `DeepSet(input_dim=...)` at line 33 raises `TypeError: DeepSet.__init__() got an unexpected keyword argument 'input_dim'`, so the class is dead code whose only caller sits under `experiments/DEPRECATED_...`; the test pins the refusal, and the rest of the file swaps the module's `DeepSet` for a legacy stand-in with the old keywords so the module's own message passing can be pinned. Also: the keywords passed to the encoder with extras dropped, parameter shapes (`o(i+1)` giving 75), widths with `out_channels=None` and an empty `node_layers`, the `set_layers` mutation (`insert(0, in_dim)` mutates the caller's list, line 59), the star-graph closed form (node 1 = 1/3 W x1 + (W x0 + W x2)/sqrt(3) + b, checked against PyG), `skip_mp` only where widths match, `skip_set` by a zero-init identity, node permutation equivariance with set invariance, a gradient reaching every parameter, seeded determinism.
