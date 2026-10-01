---
id: huh471sibkxr6shwcb4mc6a
title: Int_dcell
desc: ''
updated: 1790820765604
created: 1790820765604
---

## 2026.09.30 - DCellLoss receives [B] shapes (issues #554, #578)

Issue #554. `_shared_step` reshaped the root prediction and the target to `[B, 1]` and passed them to `DCellLoss`, whose heads are `[B]`, so every auxiliary MSE broadcast over a `[B, B]` grid; the root head was also counted because the old `torch.equal` test compared `[B]` with `[B, 1]`. The step now passes `predictions.squeeze(1)` and `gene_interaction_vals.squeeze(1)` to the loss and keeps the `[B, 1]` tensors for the metrics and plots, which flatten them anyway. `DCellLoss` now raises on any shape mismatch. Every auxiliary-on 005/006 run trained on the broadcast loss; issue #578 records them. Test: [[tests.torchcell.trainers.test_int_dcell]] (1.7 under "sum", 1.225 under "mean"; the old path gave 1.075).
