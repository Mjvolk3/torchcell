---
id: c7y7i7igwfg7g7aezu2a59u
title: Test_hetero_nsa
desc: ''
updated: 1791269863983
created: 1791269863983
---

## 2026.10.06 - Phase 21: routing contracts and three findings

Fixture: 3 genes and 2 reactions, hidden_dim 4, two heads, dropout 0, eval mode, seeded weights. Expected outputs are the block's own `NodeSelfAttention` / `SelfAttentionBlock` applied by hand to the mask the routing should pass; a spy on `NodeSelfAttention.forward` records the mask, edge_attr and edge_index of every call.

- adj_mask same-type relation: one call with mask [1, 3, 3], the store's edge_attr and edge_index; untouched types keep their input object.
- adj_mask bipartite relation: genes get the [3, 2] mask, reactions its transpose, through the same NodeSelfAttention (padded and cropped, Phase 20).
- `rmr` relations pass `stoichiometry` as edge_attr; inc_mask rmr stores prefer `hyperedge_index` and fall back to `edge_index`.
- 'sum' and 'mean' aggregation of two relations equal o1 + o2 and (o1 + o2) / 2; 'attention' without an aggregator for a type falls back to the mean.
- A relation absent from the data, or with a missing source embedding, is skipped; a declared type absent from x_dict maps to None.
- `_process_with_mask` expands a 2-D mask over a batch; a non-NSA block is called without the mask; 'S' blocks handle 2-D and 3-D input and pass unknown types through.
- `HeteroNSAEncoder` with pattern ['S']: h1 = LayerNorm_0(SAB(h0) + h0), graph vector = final_projection(mean of graph_projections(h1)); an absent type contributes zeros(1, 4) in sorted-name order; unknown aggregation is refused with the exact message by both the encoder and HeteroNSA.

Findings:

- inc_mask relations never pass their edge_attr (hetero_nsa.py:160-161), unlike the adj_mask branch (line 136).
- An edge_index-only relation builds an identity mask (lines 181-204), so its edges cannot change the output: two different edge sets give bit-identical outputs equal to the no-edge block.
- 'attention' aggregation returns one row per RELATION, not per node: `arange(len(outs)).repeat_interleave(num_nodes)` (line 224) groups all nodes of relation j into group j, giving [2, 4] for 3 genes; the row order follows the edge-type set's hash order.
- `HeteroNSAEncoder` pools a whole batch into one [1, H] graph vector; the batch vector is collected (lines 393-394) and never read.
- With an 'M' block, a declared node type absent from the data crashes the encoder with `AttributeError: 'NoneType' object has no attribute 'dim'` (line 412); the zeros fallback (line 423) is reachable only with S-only patterns.

The outputs were checked under PYTHONHASHSEED 1, 2 and 3.

## 2026.10.06 - Phase 21 audit 2: reach corrections

- inc_mask edge_attr finding: latent; the 006 retry model names its metabolic relation "reaction", not "rmr", and uses adj_mask stores.
- attention-aggregation finding: latent; the 006 retry config uses "sum".
- Line 233 (the final mean fallback) is reachable, but only outside `HeteroNSA`: `_HeteroNSA_Block(..., aggregation="max")` does not validate, and two relations are averaged; now pinned.
- The skip test's docstring now says the missing embedding is the destination (reaction) one.
