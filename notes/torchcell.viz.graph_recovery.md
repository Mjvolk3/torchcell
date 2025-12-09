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

**Definition:** For each node $i$ with true neighbors $N(i) = \{j \mid A_{ij}^{\text{true}} = 1\}$ and degree $d_i = |N(i)|$:

$$
\text{recall}_i = \frac{|N(i) \cap T_{d_i}(i)|}{d_i}
$$

where $T_{d_i}(i)$ = top-$d_i$ attention-ranked nodes from $i$.

**Aggregate:**

$$
\text{Recall@Degree}(g) = \frac{1}{|\{i : d_i > 0\}|} \sum_{i: d_i>0} \text{recall}_i
$$

**Intuition:** "If I predict as many neighbors as each node *actually* has, what fraction do I recover?"

- **Node-wise fairness:** Degree-3 and degree-300 nodes evaluated equally
- **Upper bound = 1.0** when model perfectly ranks true neighbors highest
- **No dense matrix reconstruction** needed (memory efficient)

### Precision@k

**Definition:** For each node $i$, let $T_k(i)$ = top-$k$ attention-ranked nodes:

$$
\text{prec}_k(i) = \frac{|N(i) \cap T_k(i)|}{k}
$$

**Aggregate:**

$$
\text{Precision@k}(g) = \frac{\sum_{i: d_i > 0} \text{prec}_k(i)}{|\{i \mid d_i > 0\}|}
$$

**Intuition:** "Among the top-$k$ attention-ranked edges, what fraction are real edges?"

- **Measures false-positive rate** among strongest attention weights
- **Not degree-dependent:** Even low-degree nodes can have high precision
- **Max precision for node $i$ is** $\min(d_i/k, 1)$

### Edge-Mass Alignment

**Definition:** The fraction of total attention mass placed on known graph edges:

$$
\text{EdgeMass}(g) = \frac{\sum_{(i,j) \in E_g} A_{ij}^{\text{attn}}}{\sum_{i,j} A_{ij}^{\text{attn}}}
$$

where $E_g$ is the edge set of graph $g$ and $A^{\text{attn}}$ is the attention matrix.

**Intuition:** "What fraction of attention is concentrated on real biological edges?"

- **Random baseline ≈ edge_density** of the graph
- **Values > baseline** indicate attention is learning graph structure

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
