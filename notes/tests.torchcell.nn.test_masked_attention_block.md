---
id: ava0bxex1e0aop1hv3z7g8f
title: Test_masked_attention_block
desc: ''
updated: 1791269856346
created: 1791269856346
---

## 2026.10.06 - Phase 21: CPU-path closed forms and two findings

Fixture: hidden_dim 2 (one head, or two heads of width 1) or 4 (two heads), dropout 0, eval mode, CUDA hidden. Hand-set weights: q_proj = k_proj = 0 (every allowed score is 0, so the softmax is uniform over allowed keys), v_proj = out_proj = identity, last MLP layer zero, so the block output is x + (attention-weighted mean of LayerNorm(x)). With hidden 2 every LayerNorm row is [s, -s] or [-s, s], s = 1 / sqrt(1 + 1e-5). The edge MLP of head h is set to proj_h(a) = C[h] a + D[h] with C = (1, -2), D = (0, 0.5) by passing an identity activation. Seeded random weights are checked against `_numpy_block`, an independent numpy implementation (pre-LN masked multi-head attention, -1e9 fill, GELU MLP).

- `MaskedAttentionBlock`: the numpy reference matches for two heads and two samples with different masks (head split, sqrt(head_dim) scaling, per-sample masks). Hand-set rows: row 0 of mask [[1,1,0],[0,1,0],[0,0,0]] gives x0 + mean(LN0, LN1) = [1 + s, -1 - s]; row 1 gives [3 + s, 1 - s]. A float mask is cast with `.bool()`. q and k biases of 1e30 make every allowed score +inf, the softmax NaN, and the NaN guard (lines 163-166) zeroes those rows, so the output is x + out_proj.bias.
- `NodeSelfAttention` edge bias, tensor attributes: edges (0,1) a = 2, (1,0) a = -1, (2,0) a = 3 (masked, no bias), (5,0) (src >= seq_len, dropped). out[1, 0] = 3 + s e^-1 / (e^-1 + 2). 2-D attributes use the row mean; duplicate edges keep the LAST attribute (assignment, not a sum); fewer attributes than edges truncate to min(E, numel) silently; `edge_index` is shared by the batch, so the bias lands in every sample whose mask allows the pair.
- `_prepare_edge_projections` (the GPU-path cache, called directly) returns {(h, src, dst): C[h] a + D[h]} with the same out-of-range filter for tensor and dict inputs.
- 2-D input round-trips the batch axis; a [1, 3, 2] mask is padded with a False column and a [1, 4, 4] mask is cropped (unit-level form of the Phase 20 model finding).

Findings (reproduced in `$OUT/repro1.py`):

- A query with no allowed key attends uniformly to EVERY key, masked ones included: the fill is the finite -1e9 (masked_attention_block.py:156 and :499), so the row's scores are all equal. Row 2 of the mask above gives [s/3, 2 - s/3].
- A dict `edge_attr` is ignored unless an (unused) `edge_index` is also passed (line 502); the docstring says edge_index is required only for tensors. `NSAEncoder` calls exactly the ignored form.
- The per-head edge MLPs never receive a gradient: they are evaluated under `torch.no_grad()` and read with `.item()` (lines 537-538). After backward all 8 edge-MLP tensors have `grad is None` and the other 16 have nonzero gradients.

Not covered: the GPU (FlexAttention) branches, unreachable with CUDA hidden.

## 2026.10.06 - Phase 21 audit 2: reach and reference strength

- Reach: the fully-masked-row finding is LIVE through `NodeSelfAttention` (line 499) in 006 `hetero_cell_nsa_retry.yaml` / `scripts/hetero_cell_nsa_retry.py` via `HeteroNSA`; the `MaskedAttentionBlock` class itself is latent (no live caller).
- Both numpy-reference tests now give norm1 and norm2 seeded random affine params (weight uniform in [0.5, 1.5], bias normal), carried into the reference; a mutant applying norm1 where norm2 belongs now fails (killed for both classes).
- The NaN-guard test docstring lost its last, muddled sentence.
