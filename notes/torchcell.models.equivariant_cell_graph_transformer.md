---
id: dodckzudna75nmc7zdm8wxi
title: Equivariant_cell_graph_transformer
desc: ''
updated: 1765213721280
created: 1765213721280
---

## Overview

The `EquivariantCellGraphTransformer` implements a two-stage virtual instrument architecture for modeling cellular phenotypes as $y = f(G, S)$, where $G$ is the wildtype multi-graph and $S \subseteq V(G)$ is the set of perturbed genes. Unlike the non-equivariant variant, this model preserves per-gene structure through the perturbation transformation, enabling multi-task learning.

**Location:** `torchcell/models/equivariant_cell_graph_transformer.py`

**Architecture diagrams:** [[torchcell.models.equivariant_cell_graph_transformer.mermaid]]

## Mathematical Framework

### Core Reparametrization

Both transformer variants implement the **reparametrized form**:

$$
f(G,S) \approx g_\psi\big(F_\theta(G), S\big)
$$

This exploits the fact that $(G, S)$ uniquely determines $(G \setminus S)$, so any function of $(G \setminus S)$ can be reparametrized via $(F_\theta(G), S)$.

**Computational Cost Reduction:**

$$
\text{cost} \sim \mathcal{O}(B L) \quad \text{vs} \quad \mathcal{O}(B L |E|) \text{ (original)}
$$

The $|E|$ dependence is eliminated because we only encode the wildtype graph $G$ once, not per-sample perturbed graphs.

### Gene Sequence with CLS Token

Input sequence including CLS token:

$$
X = \big(x_{\mathrm{CLS}}, x_1,\dots,x_N\big) \in \mathbb{R}^{(N+1) \times d_{\text{in}}}
$$

where:
- $x_{\mathrm{CLS}}$ is a learnable whole-cell token
- $x_i = \text{Embed}(g_i)$ for gene $g_i \in \mathcal{G}$ with $|\mathcal{G}| = N = 6607$

### Transformer Encoder

$$
H = F_\theta(X) = \big(h_{\mathrm{CLS}}, h_1,\dots,h_N\big) \in \mathbb{R}^{(N+1) \times d}
$$

**Interpretation:**
- $h_{\mathrm{CLS}} \in \mathbb{R}^d$ is the **whole-cell representation**
- $h_i \in \mathbb{R}^d$ are **gene embeddings**

### Graph-Regularized Attention

Given $K$ biological graphs $G^{(k)} = (V, E^{(k)})$ with adjacencies $A^{(k)} \in \{0,1\}^{N \times N}$:

**Row-Normalized Adjacency:**

$$
\tilde{A}_{i,:}^{(k)} = \frac{A_{i,:}^{(k)}}{\sum_j A_{ij}^{(k)} + \varepsilon} = \frac{A_{i,:}^{(k)}}{d_i^{(k)} + \varepsilon}
$$

**Graph Regularization Loss** for graph $k$ assigned to layer $\ell_k$, head $h_k$:

$$
\mathcal{L}_{\text{graph}}^{(k)} = \sum_{i \in \mathcal{I}_k} \text{KL}\left(\tilde{A}_{i,:}^{(k)} \| \alpha_{i,:}^{(\ell_k, h_k)}\right)
$$

where $\mathcal{I}_k$ is a sampled row set (e.g., 50% of nodes with positive degree).

**Total Loss:**

$$
\mathcal{L} = \mathcal{L}_{\text{phenotype}} + \sum_{k=1}^K \lambda_k \mathcal{L}_{\text{graph}}^{(k)}
$$

## Type I / Type II Virtual Instrument Architecture

The key architectural innovation is the **separation of perturbation transformation (Type I) from task-specific readout (Type II)**.

### Type I: Equivariant Perturbation Transform

**Mathematical Formulation:**

$$
H_{\text{pert}} = \mathcal{T}_\psi(H^{(L)}, S) \in \mathbb{R}^{B \times N \times d}
$$

**Cross-Attention Implementation:**

For each sample $b \in \{1, \dots, B\}$ with perturbation set $S_b$:

1. **Cross-attention from all genes to perturbation context:**

$$
\begin{align}
\text{Query} &: H_{\text{genes}} \in \mathbb{R}^{N \times d} \quad \text{(all genes)} \\
\text{Key, Value} &: H_{\text{genes}}[S_b] \in \mathbb{R}^{|S_b| \times d} \quad \text{(perturbed genes)}
\end{align}
$$

$$
H_{\text{attn}}^{(b)} = \text{CrossAttn}\big(H_{\text{genes}}, H_{\text{genes}}[S_b], H_{\text{genes}}[S_b]\big) \in \mathbb{R}^{N \times d}
$$

2. **Residual connection + LayerNorm:**

$$
H_{\text{res}}^{(b)} = \text{LayerNorm}\big(H_{\text{genes}} + \text{Dropout}(H_{\text{attn}}^{(b)})\big)
$$

3. **Feed-forward refinement:**

$$
H_{\text{pert}}^{(b)} = \text{LayerNorm}\big(H_{\text{res}}^{(b)} + \text{Dropout}(\text{FFN}(H_{\text{res}}^{(b)}))\big) \in \mathbb{R}^{N \times d}
$$

4. **Stack across batch:**

$$
H_{\text{pert}} = \text{stack}([H_{\text{pert}}^{(1)}, \dots, H_{\text{pert}}^{(B)}]) \in \mathbb{R}^{B \times N \times d}
$$

**Key Property:** Output is **EQUIVARIANT** - preserves per-gene structure for all samples.

### Type II: Task-Specific Readout

**Gene Interaction Readout:**

1. **Aggregate perturbed genes per sample:**

$$
z_S^{(b)} = \frac{1}{|S_b|} \sum_{i \in S_b} H_{\text{pert}}^{(b)}[i] \in \mathbb{R}^d
$$

2. **Concatenate with CLS token:**

$$
\text{input}^{(b)} = [h_{\mathrm{CLS}} \,\|\, z_S^{(b)}] \in \mathbb{R}^{2d}
$$

3. **Predict gene interaction:**

$$
\hat{y}_{\text{GI}}^{(b)} = \text{MLP}(\text{input}^{(b)}) \in \mathbb{R}
$$

**Future Type II Instruments:**

| Task | Formula | Output Shape |
|------|---------|--------------|
| **Fitness** (invariant) | $\text{MLP}(\text{GlobalPool}(H_{\text{pert}}))$ | $\mathbb{R}^B$ |
| **Expression** (equivariant) | $\text{MLP}_{\text{gene}}(H_{\text{pert}})$ | $\mathbb{R}^{B \times N}$ |
| **Morphology** (gene-set) | $\text{MLP}(\text{GeneSetPool}(H_{\text{pert}}))$ | $\mathbb{R}^{B \times m}$ |

## Comparison: Non-Equivariant vs Equivariant

### Mathematical Difference

**Non-Equivariant:**

$$
\hat{y} = g_\psi(h_{\mathrm{CLS}}, H_{\text{genes}}, S) \quad \text{→ Immediate collapse to scalar}
$$

**Equivariant:**

$$
\hat{y} = \mathcal{R}_\phi(\mathcal{T}_\psi(H, S)) \quad \text{→ Preserves } H_{\text{pert}} \in \mathbb{R}^{B \times N \times d}
$$

### Summary Table

| Aspect | Non-Equivariant | Equivariant |
|--------|-----------------|-------------|
| **Output Structure** | Invariant ($[B, 1]$) | Equivariant ($[B, N, d]$ available) |
| **Multi-Task Support** | Single task | Multiple tasks |
| **Architecture** | Single fused module | Two-stage (Type I + Type II) |
| **Parameter Count** | ~748K | ~781K (+4.5%) |
| **Max Batch Size (4×A100)** | 512 | 128 (memory constrained) |
| **Modularity** | Fixed architecture | Swappable Type II heads |

## Memory Constraints

### Why No Propagation Layers?

The architecture originally proposed additional **propagation layers** after Type I:

$$
H_{\text{pert}}^{(\ell+1)} = \text{TransformerLayer}(H_{\text{pert}}^{(\ell)})
$$

However, this is **memory-prohibitive**:

$$
\text{Attention memory} = B \times H \times N^2 \times 4 \text{ bytes}
$$

For $B=256$, $H=8$, $N=6607$:

$$
256 \times 8 \times 6607^2 \times 4 \approx 357 \text{ GB per layer}
$$

**Solution:** Use single cross-attention transform (Type I) without propagation.

### Batch Size Trade-offs

| Batch Size | Equivariant Transform | With 1 Prop Layer | Status |
|------------|----------------------|-------------------|--------|
| 256 | ~17 GB | ~357 GB | ✓ Safe / ❌ OOM |
| 64 | ~17 GB | ~89 GB | ✓ Safe / ❌ OOM |
| 16 | ~16 GB | ~22 GB | ✓ Safe / ⚠️ Tight |

## Parameter Count Breakdown

**Configuration:** $d=64$, $L=6$, $H=8$

| Component | Non-Equivariant | Equivariant | Change |
|-----------|-----------------|-------------|--------|
| gene_embedding | 422,848 | 422,848 | 0 |
| cls_token | 64 | 64 | 0 |
| transformer_layers | 299,904 | 299,904 | 0 |
| **perturbation_transform** | — | **49,984** | **+49,984** |
| perturbation_head | 24,961 | 8,321 | -16,640 |
| **TOTAL** | **747,777** | **781,121** | **+4.5%** |

**New Parameters from Type I Transform:**

- Cross-attention (Q, K, V, O): 16,640 params
- FFN ($d \to 4d \to d$): 33,088 params
- LayerNorm (2×): 256 params

## Implementation Details

**Default Configuration:**

- $N = 6607$ genes
- $d = 64$ or $96$ hidden dimensions
- $L = 6$ or $8$ transformer layers
- $H = 8$ or $12$ attention heads per layer
- $K = 9$ biological graphs (physical, regulatory, tflink, 6× STRING layers)
- Graph regularization at mid-depth layers (e.g., layer 4)

## When to Use Which Model?

### Use Non-Equivariant if:

- Single task (gene interaction prediction)
- Speed is critical
- Limited GPU memory
- Simple deployment requirements

### Use Equivariant if:

- Multi-task learning (fitness, expression, morphology)
- Biological interpretability of perturbed cell states is important
- Future extensibility to new readout tasks
- Sufficient GPU memory (reduce batch size as needed)

## Future Directions

### Enabling Propagation Layers

1. **Flash Attention 2**: 2-4× memory reduction
2. **Sparse Attention**: Only attend to k-nearest neighbors ($O(B \times N \times k)$)
3. **Gradient Checkpointing**: Trade compute for memory
4. **Biological Sparsity**: Use graph structure to define attention neighborhoods

### Multi-Task Type II Instruments

**Expression Prediction** (equivariant):

$$
\hat{y}_{\text{expr}}^{(b)} = \text{MLP}_{\text{gene}}(H_{\text{pert}}^{(b)}) \in \mathbb{R}^N
$$

**Morphology Prediction** (gene-set specific):

$$
\hat{y}_{\text{morph}}^{(b)} = \text{MLP}\big(\text{GeneSetPool}(H_{\text{pert}}^{(b)}, \mathcal{G}_{\text{cytoskeleton}})\big) \in \mathbb{R}^m
$$

### Sparse Multi-Task Loss

With PyG-style pointers for missing labels:

$$
\mathcal{L} = \sum_{t \in \{\text{fit, GI, morph, expr}\}} w_t \sum_{b : \text{label}_t^{(b)} \text{ available}} \ell_t\big(\hat{y}_t^{(b)}, y_t^{(b)}\big)
$$

## Related Modules

- [[torchcell.models.cell_graph_transformer]] - Non-equivariant variant
- [[torchcell.viz.graph_recovery]] - Edge recovery metrics for graph regularization
- [[torchcell.viz.transformer_diagnostics]] - Attention health diagnostics
- [[torchcell.trainers.int_hetero_cell]] - Trainer for both model variants
