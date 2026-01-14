---
id: 6w2uayyrcxn4rmxlypoloty
title: Graph_recovery
desc: ''
updated: 1765217760443
created: 1765213423206
---

## Overview

The `GraphRecoveryVisualization` class provides visualization tools for graph regularization and edge recovery metrics in cell graph transformer models. These metrics measure how well learned attention patterns align with known biological graph structures (e.g., physical interactions, regulatory relationships).

**Location:** `torchcell/viz/graph_recovery.py`

**Test file:** `tests/torchcell/viz/test_graph_recovery.py`

## Metric Definitions

### Recall@Degree

**What it measures:** For each gene, does the attention head attend to at least as many neighbors as the gene has in the reference graph?

**Intuition:** If gene A has degree 5 in the physical interaction graph (5 known neighbors), and the attention head's top-5 attended genes include all 5 true neighbors → recall = 1.0

**Formula:** For each node $i$ with true neighbors $N(i) = \{j \mid A_{ij}^{\text{true}} = 1\}$ and degree $d_i = |N(i)|$:

$$
\text{recall}_i = \frac{|N(i) \cap T_{d_i}(i)|}{d_i}
$$

where $T_{d_i}(i)$ = top-$d_i$ attention-ranked nodes from $i$.

**Aggregate:**

$$
\text{Recall@Degree}(g) = \frac{1}{|\{i : d_i > 0\}|} \sum_{i: d_i>0} \text{recall}_i
$$

**Interpretation:**

- High recall (→ 1.0): Attention heads are recovering the local neighborhood structure
- Low recall (→ 0): Attention patterns don't align with known graph edges
- **Node-wise fairness:** Degree-3 and degree-300 nodes evaluated equally
- **Upper bound = 1.0** when model perfectly ranks true neighbors highest
- **No dense matrix reconstruction** needed (memory efficient)

### Precision@k

**What it measures:** Among the top-k attended gene pairs, what fraction are true edges in the reference graph?

**Intuition:** If you look at the 32 gene pairs with highest attention weights, and 24 of them are actual edges → precision@32 = 0.75

**Formula:** For each node $i$, let $T_k(i)$ = top-$k$ attention-ranked nodes:

$$
\text{prec}_k(i) = \frac{|N(i) \cap T_k(i)|}{k}
$$

**Aggregate:**

$$
\text{Precision@k}(g) = \frac{\sum_{i: d_i > 0} \text{prec}_k(i)}{|\{i \mid d_i > 0\}|}
$$

**Interpretation:**

- Evaluated at multiple k values (e.g., k=8, 32, 128, 320)
- Higher precision at small k: Model confidently identifies true edges
- Precision decay as k increases: Expected, since you're including weaker signals
- **Max precision for node $i$ is** $\min(d_i/k, 1)$

### Edge-Mass Alignment

**What it measures:** What fraction of total attention weight falls on known graph edges?

**Intuition:** If an attention head puts 80% of its attention mass on gene pairs that are connected in the reference graph → edge-mass = 0.80

**Formula:**

$$
\text{EdgeMass}(g) = \frac{\sum_{(i,j) \in E_g} A_{ij}^{\text{attn}}}{\sum_{i,j} A_{ij}^{\text{attn}}}
$$

where $E_g$ is the edge set of graph $g$ and $A^{\text{attn}}$ is the attention matrix.

**Interpretation:**

- High (→ 1.0): Most attention is focused on known interactions
- ~0.5 (random baseline): Attention is roughly uniformly distributed
- Low (< 0.5): Attention avoids known edges (unusual)
- **Random baseline ≈ edge_density** of the graph
- **Values > baseline** indicate attention is learning graph structure

### Summary

| Metric | What it tells you |
|--------|-------------------|
| Recall@Degree | Are attention heads learning the *topology* of biological networks? |
| Edge-Mass | Is attention *concentrated* on known interactions vs scattered? |
| Precision@k | Can you trust the highest-attention pairs as real biological relationships? |

## Visualization Methods

### `plot_graph_info_summary`

Creates a comprehensive summary visualization of graph statistics at model initialization.

**Inputs:**

- `graph_info`: Dict mapping `graph_name → {num_edges, num_nodes, avg_degree, reg_layer, reg_head}`
- `save_path`: Optional path to save figure to disk

**Output:** Two bar charts (edge count, average degree) plus a summary table showing regularization layer/head mappings.

**Wandb Key:** `graph_regularization_info/summary`

### `plot_edge_recovery_recall`

Bar chart showing Recall@Degree for each graph type, colored by graph name.

**Inputs:**

- `recall_metrics`: Dict mapping `metric_key (e.g., "physical_L0_H1") → recall@degree value`
- `num_epochs`: Current epoch number
- `stage`: Stage name (e.g., "val")

**Wandb Key:** `{stage}_edge_recovery_summary/recall`

### `plot_edge_recovery_precision`

Line plot showing Precision@k across different k values, with separate lines per graph/layer combination.

**Inputs:**

- `precision_metrics`: Dict mapping `metric_key → {k → precision value}`
- `k_values`: List of k values (e.g., `[8, 32, 128, 320]`)
- `num_epochs`: Current epoch number
- `stage`: Stage name

**Visual encoding:**

- **Color:** Graph type
- **Line style:** Regularization layer (solid=L0, dashed=L1, etc.)

**Wandb Key:** `{stage}_edge_recovery_summary/precision`

### `plot_edge_recovery_per_graph`

Individual plots for each graph showing both Recall@Degree (horizontal line) and Precision@k (bars).

**Wandb Key:** `{stage}_edge_recovery/per_graph/{graph_name}`

### `plot_edge_mass_alignment`

Bar chart showing edge-mass fraction for each graph/layer/head combination.

**Wandb Key:** `{stage}_edge_recovery_summary/edge_mass`

## Color Scheme

Colors are loaded from `torchcell/torchcell.mplstyle`. Key colors:

| Usage | Hex Code | Description |
|-------|----------|-------------|
| Edge count bars | `#7191A9` | Steel blue |
| Avg degree bars | `#CC8250` | Warm orange |
| Recall line | `#B73C39` | Warm red |
| Table header | `#4A4A4A` | Dark gray |

## Usage Example

```python
from torchcell.viz.graph_recovery import GraphRecoveryVisualization

vis = GraphRecoveryVisualization(base_dir="/path/to/run")

# Plot graph info at startup
vis.plot_graph_info_summary(graph_info, save_path="graph_summary.png")

# Plot edge recovery metrics during training
vis.plot_edge_recovery_recall(recall_metrics, num_epochs=10, stage="val")
vis.plot_edge_recovery_precision(precision_metrics, k_values=[8, 32, 128, 320], num_epochs=10)
vis.plot_edge_mass_alignment(edge_mass_metrics, num_epochs=10, stage="val")
```

## Why These Metrics?

**Recall@Degree vs Recall@Edges:**

- Recall@Edges would require global reconstruction of 6607×6607 attention matrix (≈350M entries/layer/head)
- **Recall@Degree is node-wise**, memory-efficient, and biologically interpretable

**Complementary Information:**

- **Recall@Degree:** Local ranking quality (can the model recover neighbors?)
- **Precision@k:** False-positive control (is attention specific or diffuse?)
- **Edge-Mass:** Global alignment (is attention mass concentrated on real edges?)

Together: Complete picture of how well learned attention approximates biological graph structure.
