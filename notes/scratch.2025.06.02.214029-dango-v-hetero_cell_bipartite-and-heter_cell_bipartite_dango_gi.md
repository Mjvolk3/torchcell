---
id: z2x6zfsa4239dhafy3g8blw
title: 214029 Dango V Hetero_cell_bipartite and Heter_cell_bipartite_dango_gi
desc: ''
updated: 1748919072113
created: 1748918475075
---

# DANGO vs HeteroCellBipartite Models Analysis

## Model Comparison Summary

### 1. HeteroCellBipartite with DANGO (`hetero_cell_bipartite_dango.py`)

- **Purpose**: Multi-task model predicting both fitness and gene interactions
- **Architecture**: Includes metabolism (reaction/metabolite embeddings)
- **DANGO Implementation**: Simplified version, NOT full HyperSAGNN
- **Output**: Currently returns only interaction component (fitness computation commented out)

### 2. HeteroCellBipartiteDangoGI (`hetero_cell_bipartite_dango_gi.py`)

- **Purpose**: Specialized model for gene interaction prediction only
- **Architecture**: No metabolism components, dual-pathway prediction
- **DANGO Implementation**: Also simplified, NOT full HyperSAGNN
- **Output**: Gated combination of local and global predictions

### 3. Original DANGO (`dango.py`)

- **Purpose**: Pure gene interaction prediction
- **Architecture**: Full HyperSAGNN with pre-training on PPI networks
- **Key Components**: DangoPreTrain, MetaEmbedding, HyperSAGNN

## Detailed Architecture Comparison

### DANGO HyperSAGNN (Original)

The true HyperSAGNN implementation has:

1. **Two attention layers** with separate Q, K, V projections
2. **Multi-head attention** with proper head dimension splitting
3. **Global masked attention** that processes all nodes at once
4. **Vectorized batch processing** with scatter_mean aggregation
5. **ReZero connections** (beta parameters) for both layers
6. **Efficient masked attention** for same-set nodes only

### Simplified Attention in HeteroCellBipartite Models

Both HeteroCellBipartite variants use:

1. **Single attention layer** instead of two
2. **Iterative batch processing** rather than vectorized
3. **Simpler attention mechanism** without multi-head splitting
4. **Basic ReZero** with single beta parameter
5. **No sophisticated masked global attention**

## Key Implementation Differences

### HeteroCellBipartite (with DANGO)

```python
# Simplified attention mechanism
class GeneInteractionAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        # Single layer Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Parameter(torch.ones(1) * 0.1)
```

### Original DANGO HyperSAGNN

```python
class HyperSAGNN(nn.Module):
    def __init__(self, hidden_channels: int, num_heads: int = 4):
        # Layer 1
        self.Q1 = nn.Linear(hidden_channels, hidden_channels)
        self.K1 = nn.Linear(hidden_channels, hidden_channels)
        self.V1 = nn.Linear(hidden_channels, hidden_channels)
        self.O1 = nn.Linear(hidden_channels, hidden_channels)
        self.beta1 = nn.Parameter(torch.zeros(1))
        
        # Layer 2
        self.Q2 = nn.Linear(hidden_channels, hidden_channels)
        self.K2 = nn.Linear(hidden_channels, hidden_channels)
        self.V2 = nn.Linear(hidden_channels, hidden_channels)
        self.O2 = nn.Linear(hidden_channels, hidden_channels)
        self.beta2 = nn.Parameter(torch.zeros(1))
```

## Unique Features by Model

### HeteroCellBipartite

- Metabolism integration (reaction/metabolite nodes)
- Fitness prediction capability
- Transition mechanism between fitness and interaction predictions
- Uses SubgraphRepresentation processor

### HeteroCellBipartiteDangoGI

- Dual prediction pathway:
  - Local: DANGO-style on perturbed genes
  - Global: MLP on z_p_global
- Learned gating mechanism for combining predictions
- Extensive NaN checking
- More robust weight initialization
- Uses `cell_graph_idx_pert` field

### Original DANGO

- Pre-training on 6 STRING PPI networks
- Meta-embedding integration across networks
- Full vectorized HyperSAGNN implementation
- Lambda-weighted reconstruction loss

## Conclusion

**Neither HeteroCellBipartite model implements the full DANGO HyperSAGNN architecture.** They both use simplified attention mechanisms that capture the core concept (dynamic vs static embeddings → squared differences → prediction) but lack:

- Multi-layer attention architecture
- Proper multi-head attention with dimension splitting
- Vectorized global masked attention
- Full pre-training pipeline

The HeteroCellBipartite models are practical adaptations that integrate DANGO-inspired ideas into a broader framework (with metabolism, fitness prediction) while making architectural compromises for simplicity and integration purposes.
