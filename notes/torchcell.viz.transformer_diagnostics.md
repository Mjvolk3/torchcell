---
id: 7eass27v5rggcnno06e6wky
title: Transformer_diagnostics
desc: ''
updated: 1765213621064
created: 1765213621064
---

## Overview

The `TransformerDiagnostics` class provides visualization tools for monitoring transformer attention health and detecting pathological behaviors during training. These metrics help diagnose issues like attention collapse, one-hot attention, and sink token formation.

**Location:** `torchcell/viz/transformer_diagnostics.py`

**Test file:** `tests/torchcell/viz/test_transformer_diagnostics.py`

## Metric Definitions

### Attention Entropy

**Definition:** Shannon entropy of the attention distribution for each row (query), averaged across all queries:

$$
H(\mathbf{a}_i) = -\sum_{j=1}^{N} a_{ij} \log(a_{ij})
$$

**Aggregate (per layer):**

$$
\text{Entropy}(l) = \frac{1}{N} \sum_{i=1}^{N} H(\mathbf{a}_i)
$$

**Interpretation:**
- **High entropy (→ log N):** Uniform attention (no focus)
- **Low entropy (→ 0):** Concentrated attention (potentially one-hot collapse)
- **Healthy range:** Moderate entropy indicating selective but distributed attention

### Effective Rank

**Definition:** Exponential of the entropy, measuring how many positions effectively receive attention:

$$
\text{EffRank}(\mathbf{a}_i) = \exp\left(H(\mathbf{a}_i)\right) = \exp\left(-\sum_{j=1}^{N} a_{ij} \log(a_{ij})\right)
$$

**Aggregate (per layer):**

$$
\text{EffRank}(l) = \frac{1}{N} \sum_{i=1}^{N} \text{EffRank}(\mathbf{a}_i)
$$

**Interpretation:**
- **EffRank = 1:** One-hot attention (collapsed)
- **EffRank = N:** Uniform attention (no selectivity)
- **Healthy range:** Typically 5-50 depending on task

### Top-K Concentration

**Definition:** Fraction of attention mass in the top-$k$ positions:

$$
\text{Top-}k(\mathbf{a}_i) = \sum_{j \in T_k(i)} a_{ij}
$$

where $T_k(i)$ = indices of top-$k$ attention weights for query $i$.

**Variants tracked:**
- **Top-5:** $k=5$
- **Top-10:** $k=10$
- **Top-50:** $k=50$

**Interpretation:**
- **Top-5 ≈ 1.0:** Extremely concentrated (potential collapse)
- **Top-50 ≈ 0.5:** Attention spread across many positions
- **Useful for:** Detecting if attention is too sharp or too diffuse

### Max Row Weight (One-Hot Detection)

**Definition:** Maximum attention weight in each row, then averaged:

$$
\text{MaxRowWeight}(l) = \frac{1}{N} \sum_{i=1}^{N} \max_{j} a_{ij}
$$

**Interpretation:**
- **MaxRowWeight → 1.0:** One-hot collapse (attention puts all weight on single token)
- **MaxRowWeight → 1/N:** Uniform attention
- **Healthy range:** 0.1-0.5 depending on sequence length

### Column Entropy (Sink Collapse Detection)

**Definition:** Entropy of the column-wise attention distribution (how attention is *received*):

$$
c_j = \sum_{i=1}^{N} a_{ij} \quad \text{(column sum)}
$$

$$
\tilde{c}_j = \frac{c_j}{\sum_{k} c_k} \quad \text{(normalized)}
$$

$$
H_{\text{col}} = -\sum_{j=1}^{N} \tilde{c}_j \log(\tilde{c}_j)
$$

**Interpretation:**
- **High col entropy:** Attention received uniformly across positions
- **Low col entropy:** Sink token formation (few tokens receive most attention)
- **Sink collapse:** Single token becomes attention sink for all queries

### Max Column Sum

**Definition:** Maximum total attention received by any single position:

$$
\text{MaxColSum}(l) = \max_{j} \sum_{i=1}^{N} a_{ij}
$$

**Interpretation:**
- **High value (→ N):** Sink token present (one position receives attention from all queries)
- **Healthy value (→ 1):** Each position receives roughly equal total attention
- **Complements column entropy:** Direct measure of sink severity

### Residual Update Ratio

**Definition:** Ratio of the residual update magnitude to the input magnitude:

$$
\text{ResidualRatio}(l) = \frac{\|\Delta \mathbf{x}^{(l)}\|}{\|\mathbf{x}^{(l)}\|}
$$

where $\Delta \mathbf{x}^{(l)} = \text{TransformerBlock}^{(l)}(\mathbf{x}^{(l)}) - \mathbf{x}^{(l)}$ is the residual update.

**Interpretation:**
- **Ratio ≈ 0:** Layer has no effect (vanishing gradients/dead layer)
- **Ratio >> 1:** Layer dominates the residual stream (unstable)
- **Healthy range:** 0.1-1.0 (layer contributes meaningfully but doesn't dominate)

## Visualization Method

### `plot_attention_diagnostics`

Creates a 3×2 grid of diagnostic plots for all metrics.

**Inputs:**
- `attention_stats`: Dict mapping `layer_idx → {entropy, effective_rank, top5, top10, top50, max_row_weight, col_entropy, max_col_sum}`
- `residual_ratios`: Optional dict mapping `layer_idx → residual_update_ratio`
- `qk_logit_stats`: Optional dict mapping `layer_idx → {logit_mean, logit_std, saturation_ratio}` (not currently plotted)
- `gradient_norms`: Optional dict mapping `layer_idx → gradient_norm` (not currently plotted)
- `num_epochs`: Current epoch number
- `stage`: Stage name (e.g., "val")

**Output:** 6-panel figure showing:
1. Attention Entropy per Layer
2. Effective Rank per Layer (log scale)
3. Top-K Concentration (5, 10, 50)
4. Max Row Weight (One-Hot Detection)
5. Column Concentration (dual y-axis: col entropy + max col sum)
6. Residual Update Ratio (log scale with healthy bounds)

**Wandb Key:** `{stage}_transformer_diagnostics/summary`

## Pathology Detection

| Pathology | Symptoms | Metrics to Watch |
|-----------|----------|------------------|
| **One-hot collapse** | Attention becomes delta function | Low entropy, EffRank → 1, MaxRowWeight → 1 |
| **Sink collapse** | All queries attend to same token | Low col entropy, MaxColSum → N |
| **Vanishing layers** | Layer has no effect | ResidualRatio → 0 |
| **Exploding updates** | Layer dominates residual | ResidualRatio >> 1 |
| **Uniform attention** | No selectivity | High entropy, EffRank → N, Top-5 → 5/N |

## Color Scheme

Colors are loaded from `torchcell/torchcell.mplstyle`:

| Plot Element | Hex Code | Description |
|--------------|----------|-------------|
| Entropy line | `#7191A9` | Steel blue |
| Effective rank | `#B73C39` | Warm red |
| Top-5 mass | `#000000` | Black |
| Top-10 mass | `#CC8250` | Warm orange |
| Top-50 mass | `#6B8D3A` | Olive green |
| Max row weight | `#34699D` | Deep blue |
| Column entropy | `#3D796E` | Teal |
| Max column sum | `#4A9C60` | Green |
| Residual ratio | `#3978B5` | Purple |

## Usage Example

```python
from torchcell.viz.transformer_diagnostics import TransformerDiagnostics

vis = TransformerDiagnostics(base_dir="/path/to/run")

# Collect stats during validation
attention_stats = {
    0: {"entropy": 2.5, "effective_rank": 12.0, "top5": 0.4, "top10": 0.6, "top50": 0.9,
        "max_row_weight": 0.15, "col_entropy": 5.0, "max_col_sum": 10.0},
    1: {"entropy": 2.2, "effective_rank": 9.0, ...},
    # ... more layers
}

residual_ratios = {0: 0.3, 1: 0.25, 2: 0.2, ...}

vis.plot_attention_diagnostics(
    attention_stats=attention_stats,
    residual_ratios=residual_ratios,
    num_epochs=10,
    stage="val"
)
```

## Related Modules

- [[torchcell.viz.graph_recovery]] - Edge recovery metrics for graph regularization
- [[torchcell.trainers.int_hetero_cell]] - Trainer that computes and logs these metrics
