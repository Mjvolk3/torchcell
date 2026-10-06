---
id: ethropapjhavqxjbmth4rm3
title: Test_nsa_encoder
desc: ''
updated: 1791269871633
created: 1791269871633
---

## 2026.10.06 - Phase 21: adjacency sources, padding and edge attributes

Fixture: a 3-node directed cycle 0->1->2->0, input_dim 2, hidden_dim 4, two heads, dropout 0, eval mode, seeded weights; structural identities only (equal outputs across equivalent inputs, the layer list, dependence on a batch companion).

- pattern None builds [NSA, SAB, NSA, SAB]; an invalid block type raises "Invalid block type 'Q'."
- An edge_index tensor, `Data(edge_index)`, `Data.adj_mask` [1, 3, 3] and `Data.adj` [1, 3, 3] give bit-identical outputs; an object with none of them raises "Cannot extract adjacency information from provided data."
- In a batch, the largest graph's rows equal that graph run alone under pattern ['M'].

Findings:

- A 2-D `data.adj` works only for S-only patterns (the existing tests use ['S']); an 'M' block fails expanding the mask over heads with the exact RuntimeError pinned in the test.
- Edge attributes never change the output: the encoder passes the dict without edge_index (nsa_encoder.py:140) and NodeSelfAttention drops it (masked_attention_block.py:502).
- S blocks attend to padding rows: a 2-node graph's outputs change when it is batched with a 4-node graph under ['S'] and stay the same (1e-6) under ['M'].

## 2026.10.06 - Phase 21 audit 2: an independent identity for node order

The batched-unpad test now also checks enc(xb, CYCLE) against the layer applied by hand, layers[0](input_proj[xb](None), to_dense_adj(CYCLE).bool(), None)[0] (with no edge attributes the encoder passes None, nsa_encoder.py:140). A padding mutant writing x in reversed node order is now killed.
