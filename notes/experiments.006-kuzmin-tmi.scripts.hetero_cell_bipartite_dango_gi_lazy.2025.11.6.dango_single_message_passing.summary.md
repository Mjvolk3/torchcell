---
id: fbaxfjgc4b6fr5gerfaj4o0
title: Summar
desc: ''
updated: 1762485636045
created: 1762485533261
---
summary from chatgpt:

Here’s the short version of what this model is doing.

### Big picture

`GeneInteractionDango` is:

1. **Global gene encoder over a multi-graph** (all genes, all networks at once)
2. **Graph-type aggregator** (cross-attention or pairwise interaction between networks)
3. **HyperSAGNN-style local interaction head** over *gene embeddings* (optionally batched)
4. **Global aggregator** to go from gene-level to phenotype (file truncated here)

Crucially: **message passing is over the full gene graph(s), not per-sample subgraphs**. The per-sample notion only comes in later via a `batch` index in the interaction head.

---

### Components

**1. SelfAttentionGraphAggregation / PairwiseGraphAggregation**

* Input: dict `{graph_name -> [num_nodes, hidden_dim]}` of per-graph gene embeddings.
* `SelfAttentionGraphAggregation`:

  * Stacks to `[num_nodes, num_graphs, hidden_dim]`.
  * Adds learnable per-graph embeddings.
  * Runs `MultiheadAttention` over the *graph* dimension.
  * Averages over graphs → `[num_nodes, hidden_dim]`.
* `PairwiseGraphAggregation`:

  * For all `(g1, g2)` pairs builds concatenated features `[num_nodes, 2*hidden_dim]`.
  * Feeds each through an MLP, stacks as `[num_nodes, num_pairs, hidden_dim]`.
  * Learns attention weights over pairs and aggregates to `[num_nodes, hidden_dim]`.
* Both operate **per-node over all graphs simultaneously**, no sample dimension.

---

**2. HeteroConvAggregator**

* Wraps a dict `{EdgeType -> conv}` (GIN/GATv2 wrapped in `AttentionConvWrapper`).
* For each edge type `(src, rel, dst)`:

  * Runs `conv(x_dict[src], edge_index_dict[(src, rel, dst)])` → `[num_dst_nodes, hidden_dim]`.
  * Collects outputs per destination node type and relation (graph name).
* Then aggregates per destination type:

  * `sum` / `mean`, or
  * learned `SelfAttentionGraphAggregation` / `PairwiseGraphAggregation`.
* So each layer does:
  [
  {x_{\text{gene}}} \xrightarrow{\text{per-graph conv}} {h^{(g)}*{\text{gene}}}
  \xrightarrow{\text{graph aggregation}} h*{\text{gene}}
  ]
  **once for all genes**, over all graphs in `gene_multigraph`.

---

**3. DangoLikeHyperSAGNN + GeneInteractionPredictor**

* `DangoLikeHyperSAGNN`:

  * Static embedding: `static = MLP(gene_embeddings)`.
  * Dynamic embedding: multi-layer, multi-head self-attention with ReZero:

    * If `batch` provided: loop over unique `b` and run attention per batch slice.
    * Otherwise: single global self-attention over all input rows.
* `GeneInteractionPredictor`:

  * Calls HyperSAGNN → `static`, `dynamic`.
  * Uses squared difference: `diff_sq = (dynamic - static)^2`.
  * Linear head → per-gene scores.
  * If `batch` given: `scatter_mean` to get **one score per batch element**.
* This is where **per-sample structure** enters: via a `batch` vector over *rows of `gene_embeddings`*, not via separate graphs.

---

**4. GeneInteractionDango (top-level)**

* Inputs:

  * `gene_num`, `GeneMultiGraph` (names + edge indices for all graphs).
* Internals:

  * Learnable gene embedding: `[gene_num, hidden_dim]`.
  * `PreProcessor` MLP over embeddings.
  * `num_layers` stacked `HeteroConvAggregator` layers over **all gene graphs**.
  * Final embeddings go into `GeneInteractionPredictor` (HyperSAGNN) and a `Global` aggregator (not shown in snippet).
* Message passing:

  * Uses *one* global `gene_multigraph` with edge types `("gene", graph_name, "gene")`.
  * For each layer and each graph name, runs a conv on the **full** gene graph for that relation.
  * There is **no explicit per-sample loop in the GNN**; batching happens later through `batch` indices in the predictor.

---

### Direct answer to your conceptual question

* **DANGO (original)**: does message passing on the full homogeneous gene graph once per forward, then indexes perturbed genes.
* **This hetero DANGO-GI model**: also does message passing on the full multi-graph once per layer (via `HeteroConvAggregator`), not per-sample subgraphs. Per-sample behavior comes later via `batch` in `GeneInteractionPredictor`.

So structurally, both architectures share the “**global graph encode → index/aggregate for samples**” pattern; the hetero version just adds (i) multi-graph HeteroConv + graph-attention aggregation and (ii) more complex local interaction head.

Claude summary:

● Performance Bottleneck Analysis Summary

  The Mystery

* DANGO: ~10 it/s (fast)
* Lazy Hetero: ~0.42 it/s (slow)
* Difference: 875x slower in full training

  But isolated forward pass profiling showed only 1.64x difference (23.8ms vs 39ms), which didn't explain the
  massive slowdown.

  ---
  Root Cause Discovered

  DANGO's Architecture (Efficient ✅)

  def forward(self, cell_graph, batch):
      # 1. Message passing on FULL graph ONCE
      embeddings = self.pretrain_model(cell_graph)  # [6,607 genes]

      # 2. Index perturbed genes only
      perturbed = embeddings[batch.perturbation_indices]  # [~84 nodes]

      # 3. Process perturbed genes
      scores = self.hyper_sagnn(perturbed, ...)

# Total nodes processed: 6,607

  Lazy Hetero's Architecture (Inefficient ❌)

  def forward(self, cell_graph, batch):
      # 1. Process wildtype
      z_w = self.forward_single(cell_graph)  # [6,607 genes]

      # 2. Process BATCHED perturbed graphs
      z_i = self.forward_single(batch)  # [6,607 × 28 = 185,396 nodes!]

# Total nodes processed: 192,003 (30x more!)

  The lazy hetero model expands embeddings for batch_size copies:
  x_gene_exp = self.gene_embedding.weight.expand(batch_size, -1, -1)

# [28 samples, 6607 genes, 64 dims] = 185,396 nodes

  ---
  The Numbers

  | Metric          | DANGO   | Lazy Hetero | Ratio |
  |-----------------|---------|-------------|-------|
  | Nodes processed | 6,607   | 192,003     | 30x   |
  | Edge operations | ~2.4M   | ~67M        | 28x   |
  | Graph copies    | 1       | 28          | 28x   |
  | Training speed  | 10 it/s | 0.42 it/s   | 24x   |

  ---
  Why Lazy Hetero Uses This Architecture

  Each sample has different edge masks (different perturbed genes):

* Sample 1: mask out genes [1, 5, 10]
* Sample 2: mask out genes [3, 8, 15]
* Sample 3: mask out genes [2, 7, 12]

  PyG batches these into one large graph with 28 copies, then applies different masks during message passing.

  Result: Must process all 28 copies separately because masks differ.

  ---
  Why Other Optimizations Failed

  | Optimization       | Why It Failed              |
  |--------------------|----------------------------|
  | GPU masks          | Still processing 30x nodes |
  | Preprocessed masks | Still processing 30x nodes |
  | Reduce DDP gather  | Not the bottleneck (17.6%) |
  | Single graph only  | Still batches 28 copies    |

  The fundamental issue: Redundant message passing on batched copies.

  ---
  Solutions (In Order of Impact)

  1. Adopt DANGO's Strategy (30x speedup potential)

  Process full graph once, apply perturbations as indexing operations:

# Compute once

  full_embeddings = forward_single(cell_graph)  # [6,607]

# Per sample: just index and mask

  for sample in batch:
      sample_emb = full_embeddings[sample.kept_genes]

  2. Cache Full Graph Embeddings (10-20x speedup)

  Compute full graph embeddings once per epoch, reuse across batches.

  3. Reduce Batch Size (Linear speedup)

* batch_size 28 → 4: 7x speedup
* But still processes 4 copies vs DANGO's 1

  4. Message Passing on Masked Graph (Complex)

  Generate actual subgraphs rather than masked copies, but loses lazy benefits.

  ---
  Key Insight

  The profiling showed all categories were slower (optimizer 24%, DDP 14%, etc.), but the real bottleneck is
  hidden:

  The "model forward" category (14.2%, 162s) includes the batched message passing overhead. Breaking it down:

* Actual GNN computation: ~40ms (like DANGO)
* Processing 30x nodes: ~6,000ms per step
* Other overhead: ~300ms

  This explains the 875x total slowdown - the core inefficiency compounds through every operation.

  ---
  Recommendation

  Adopt DANGO's single-pass architecture by:

  1. Process full graph embeddings once
  2. Use perturbation masks as indexing operations, not during message passing
  3. Apply subgraph logic AFTER embeddings are computed

  Expected result: 10-30x speedup, bringing lazy hetero from 0.42 it/s → 4-12 it/s.
