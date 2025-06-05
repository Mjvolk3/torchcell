---
id: af71cbruacqjjbj9y5brjeq
title: 182738 Dango V Hetero_cell_bipartite_dango_gi
desc: ''
updated: 1749079671591
created: 1749079671591
---

# Comparison of HyperSAGNN Prediction Heads: Dango vs HeteroCellBipartiteDangoGI

## Overview

Both models aim to predict genetic interactions but use different approaches for their prediction heads. The original Dango model uses a HyperSAGNN (Hypergraph Self-Attention Graph Neural Network) while HeteroCellBipartiteDangoGI uses a modified approach with separate local and global predictors.

## Dango Model - HyperSAGNN

### Architecture

The Dango model's HyperSAGNN (lines 232-397 in dango.py) consists of:

1. **Static Embeddings**:
   ```python
   self.static_embedding = nn.Sequential(
       nn.Linear(hidden_channels, hidden_channels), nn.ReLU()
   )
   ```

2. **Two-Layer Self-Attention with ReZero**:
   - Layer 1: Q1, K1, V1, O1 projections with beta1 parameter
   - Layer 2: Q2, K2, V2, O2 projections with beta2 parameter
   - Multi-head attention (default 4 heads)
   - ReZero connections: `d_i = β * d̂_i + E_i`

3. **Prediction Mechanism**:
   ```python
   # Compute element-wise squared differences
   squared_diff = (dynamic_embeddings - static_embeddings) ** 2
   
   # Compute node scores
   node_scores = self.prediction_layer(squared_diff).squeeze(-1)
   
   # Aggregate scores for each set using scatter_mean
   interaction_scores = scatter_mean(node_scores, batch_indices, dim=0, dim_size=num_batches)
   ```

### Key Features:
- **Global Attention**: All nodes in a batch can attend to each other within their set
- **Masked Attention**: Prevents self-attention and cross-set attention
- **Single Prediction Path**: One unified prediction for the entire set
- **Vectorized Processing**: Processes all nodes simultaneously

### Mathematical Formulation:
For each node i in the set {i,j,k}:
- Static: `s_i = σ(W_s * E_i + b_s)`
- Dynamic: Multi-head self-attention with ReZero
- Score: `y_i = FC[(d_i - s_i)²]`
- Final: `τ̂_ijk = (y_i + y_j + y_k) / 3`

## HeteroCellBipartiteDangoGI - Modified Approach

### Architecture

The HeteroCellBipartiteDangoGI model (lines 48-172, 258-341) uses:

1. **GeneInteractionAttention Module** (lines 48-126):
   ```python
   class GeneInteractionAttention(nn.Module):
       def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
           # Q, K, V projections
           self.q_proj = nn.Linear(hidden_dim, hidden_dim)
           self.k_proj = nn.Linear(hidden_dim, hidden_dim)
           self.v_proj = nn.Linear(hidden_dim, hidden_dim)
           self.out_proj = nn.Linear(hidden_dim, hidden_dim)
           
           # ReZero parameter (initialized to small value)
           self.beta = nn.Parameter(torch.ones(1) * 0.1)
   ```

2. **GeneInteractionPredictor Module** (lines 128-172):
   - Uses GeneInteractionAttention for dynamic embeddings
   - Computes difference between dynamic and static embeddings
   - Squares the difference
   - Predicts scores per gene, then averages per batch

3. **Dual Prediction Paths**:
   - **Local Predictor**: `self.gene_interaction_predictor` - focuses on perturbed genes
   - **Global Predictor**: `self.global_interaction_predictor` - processes global embeddings
   
4. **Gating Mechanism**:
   ```python
   # MLP for gating weights
   self.gate_mlp = nn.Sequential(
       nn.Linear(2, hidden_channels),
       nn.ReLU(),
       nn.Dropout(dropout),
       nn.Linear(hidden_channels, 2),
   )
   
   # Apply gating
   gate_weights = F.softmax(gate_logits, dim=1)
   weighted_preds = pred_stack * gate_weights
   gene_interaction = weighted_preds.sum(dim=1, keepdim=True)
   ```

### Key Differences from Dango:

1. **Dual Prediction Architecture**:
   - Local predictor focuses on perturbed gene embeddings only
   - Global predictor uses aggregated whole-graph embeddings
   - Learned gating mechanism balances both predictions

2. **Attention Mechanism**:
   - Single attention layer (vs. two in Dango)
   - Different initialization (beta = 0.1 vs 0.01)
   - No multi-head attention structure (processes full hidden_dim)

3. **Embedding Processing**:
   - Uses `AttentionalGraphAggregation` for global embeddings
   - Separate handling of wildtype (z_w) and perturbed (z_i) states
   - Explicit perturbation difference: `z_p = z_w - z_i`

4. **Score Computation**:
   - Local: Based on perturbed gene embeddings only
   - Global: Based on perturbation difference at graph level
   - Final: Weighted combination of both

### Mathematical Formulation:
- Local: `score_local = mean(FC[(dynamic_pert - static_pert)²])`
- Global: `score_global = FC(z_p_global)`
- Gating: `weights = softmax(MLP([score_global, score_local]))`
- Final: `score_final = Σ(weights * [score_global, score_local])`

## Key Architectural Differences

| Feature | Dango HyperSAGNN | HeteroCellBipartiteDangoGI |
|---------|------------------|----------------------------|
| Prediction Paths | Single unified path | Dual (local + global) |
| Attention Layers | 2 layers | 1 layer |
| Multi-head | Yes (4 heads) | No (full dimension) |
| ReZero Init | β = 0.01 | β = 0.1 |
| Aggregation | scatter_mean | AttentionalGraphAggregation |
| Gating | None | Learned MLP gating |
| Focus | All genes in set | Perturbed genes + global |

## Advantages and Trade-offs

### Dango HyperSAGNN:
**Advantages**:
- Unified architecture following original paper
- Multi-head attention for diverse representations
- Two-layer depth for complex interactions
- Proven effectiveness on trigenic interactions

**Trade-offs**:
- Single prediction path may miss local/global patterns
- Requires all genes in set to be processed together

### HeteroCellBipartiteDangoGI:
**Advantages**:
- Dual prediction captures both local and global effects
- Learned gating adapts to different interaction types
- Explicit modeling of perturbation effects
- More flexible for varying perturbation sizes

**Trade-offs**:
- More complex architecture with more parameters
- Potential for overfitting with gating mechanism
- Deviation from original Dango design

## Implementation Notes

1. **NaN Handling**: HeteroCellBipartiteDangoGI includes extensive NaN checks throughout forward pass
2. **Batch Processing**: Both handle batched data but with different approaches
3. **Parameter Initialization**: Different strategies for weight initialization
4. **Loss Functions**: Both compatible with LogCosh and other regression losses

## Conclusion

The HeteroCellBipartiteDangoGI model represents a significant architectural departure from the original Dango HyperSAGNN. While Dango uses a unified self-attention approach processing all genes equally, HeteroCellBipartiteDangoGI introduces a dual-path architecture that separately models local (perturbed gene) and global (whole graph) effects, combining them through a learned gating mechanism. This design choice reflects different assumptions about how genetic interactions manifest - whether through uniform hypergraph relationships (Dango) or through a combination of local perturbation effects and global network changes (HeteroCellBipartiteDangoGI).