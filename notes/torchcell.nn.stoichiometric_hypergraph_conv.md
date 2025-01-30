---
id: dxrsq9k73rbu63vfo76tqyf
title: Met_hypergraph_conv
desc: ''
updated: 1738019137523
created: 1736144453668
---
## 2025.01.27

# StoichiometricHypergraphConv Class Documentation

This document explains how the **`StoichiometricHypergraphConv`** operator implements a hypergraph-like convolution using stoichiometric coefficients and optional attention.

---

## Overview of the Mathematical Form

The traditional hypergraph convolution layer can be summarized by:

$$
\mathbf{X}^{\prime} \;=\; \mathbf{D}^{-1} \,\mathbf{H}\,\mathbf{W}\,\mathbf{B}^{-1}\,\mathbf{H}^{\top}\,\mathbf{X}\,\Theta
$$

where:

- $\mathbf{H} \in \{0,1\}^{N \times M}$ is the incidence matrix (mapping nodes to edges).
- $\mathbf{W} \in \mathbb{R}^M$ is the (diagonal) hyperedge weight matrix.
- $\mathbf{D}$ and $\mathbf{B}$ are diagonal degree matrices for the nodes and hyperedges, respectively.
- $\mathbf{X} \in \mathbb{R}^{N \times F_{\mathrm{in}}}$ is the node feature matrix.
- $\Theta \in \mathbb{R}^{F_{\mathrm{in}} \times F_{\mathrm{out}}}$ is the learnable weight matrix.

**In this code**, we adapt the formula to incorporate **stoichiometric coefficients** (signed weights) and an **attention mechanism** (optional). The stoichiometric values replace or augment the role of $\mathbf{W}$ in weighting node-edge incidences, while $\mathbf{D}^{-1}$ and $\mathbf{B}^{-1}$ are computed from the absolute values of those coefficients.

---

## Class Signature

```python
class StoichiometricHypergraphConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        attention_mode: str = "node",
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs,
    ):
        ...
```

### Key Constructor Arguments

1. **`in_channels`** / **`out_channels`**: Dimensionalities of the input and output features.
2. **`use_attention`** *(bool)*: If `True`, attention coefficients $\alpha$ are computed and applied during message passing.
3. **`attention_mode`** *(str)*: Either `"node"` or `"edge"`, controlling how the softmax normalization of attention scores is performed.
4. **`heads`** *(int)*: Number of attention heads (only relevant if `use_attention=True`).
5. **`concat`** *(bool)*: If `True`, multi-head outputs are concatenated; if `False`, they are averaged.
6. **`negative_slope`**, **`dropout`**: LeakyReLU slope for attention scores, and dropout probability for attention.
7. **`bias`** *(bool)*: Whether a bias vector is learned after the linear transforms.
8. **`**kwargs`**: Additional arguments passed to PyG’s `MessagePassing` (like `aggr="add"`).

---

## Forward Method

```python
def forward(
    self,
    x: Tensor,
    edge_index: Tensor,
    stoich: Tensor,
    hyperedge_attr: Optional[Tensor] = None,
    num_edges: Optional[int] = None,
) -> Tensor:
    ...
```

### Forward Inputs

- **`x`** $\in \mathbb{R}^{N \times F_{\mathrm{in}}}$: Node features.
- **`edge_index`** $\in \mathbb{R}^{2 \times E}$:
  - Defines the mapping between nodes and “hyperedges” (reactions).  
  - For standard hypergraph notation, this effectively encodes $\mathbf{H}$.
- **`stoich`** $\in \mathbb{R}^{E}$:
  - **Stoichiometric coefficients** used as signed weights.  
  - Negative coefficients typically indicate reactants, positive indicate products.
- **`hyperedge_attr`** $\in \mathbb{R}^{M \times \_}$ *(optional)*:
  - Extra features for each hyperedge. Only needed if `use_attention=True`.
- **`num_edges`** *(int, optional)*:
  - The number of hyperedges $M$. Used for size checks and indexing.

### Internal Steps

1. **Linear Transform**:

   ```python
   x = self.lin(x)
   ```

   - Learns a transform $\Theta$ from $\mathbb{R}^{F_{\mathrm{in}}}\to \mathbb{R}^{F_{\mathrm{out}}}$.
   - If `use_attention=True`, `x` is reshaped for multiple heads.

2. **Attention (Optional)**:

   ```python
   if self.use_attention:
       hyperedge_attr = self.lin(hyperedge_attr)
       ...
       alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
       alpha = F.leaky_relu(alpha, self.negative_slope)
       alpha = softmax(alpha, ...)  
       alpha = F.dropout(alpha, p=self.dropout, ...)
   ```

   - Merges node and hyperedge features ($x_i$ and $x_j$) into a single attention score $\alpha$.
   - Normalized via softmax over nodes or edges, depending on `attention_mode`.

3. **Degree Computation** ($\mathbf{D}^{-1}, \mathbf{B}^{-1}$):

   ```python
   # D: node-based degrees
   D = scatter(torch.abs(stoich), edge_index[0], ... reduce="sum")
   D = 1.0 / D

   # B: hyperedge-based degrees
   B = scatter(torch.abs(stoich), edge_index[1], ... reduce="sum")
   B = 1.0 / B
   ```

   - `edge_index[0]` enumerates nodes, `edge_index[1]` enumerates hyperedges.
   - $\mathbf{D}[i] = \sum_{e \in \mathcal{E}_i} |stoich_e|$
   - $\mathbf{B}[j] = \sum_{v \in \mathcal{V}*j} |stoich*{v,j}|$
   - Inverse degrees used to approximate $\mathbf{D}^{-1}$ and $\mathbf{B}^{-1}$.

4. **Message Passing**:

   ```python
   out = self.propagate(
       edge_index, x=x, norm=B, alpha=alpha, stoich=stoich, ...
   )
   out = self.propagate(
       edge_index.flip([0]), x=out, norm=D, alpha=alpha, stoich=stoich, ...
   )
   ```

   - First pass: node $\to$ hyperedge direction (applying $\mathbf{B}^{-1}$).
   - Second pass: hyperedge $\to$ node direction (applying $\mathbf{D}^{-1}$).
   - The sign and magnitude of `stoich` are used in `message(...)`.

5. **Message Function**:

   ```python
   def message(
       self, x_j, norm_i, alpha, stoich
   ) -> Tensor:
       magnitude = torch.abs(stoich)
       sign = torch.sign(stoich)
       out = (
           norm_i.view(-1, 1, 1)
           * magnitude.view(-1, 1, 1)
           * sign.view(-1, 1, 1)
           * x_j.view(-1, self.heads, self.out_channels)
       )
       if alpha is not None:
           out = alpha.view(-1, self.heads, 1) * out
       return out
   ```

   - **`magnitude = abs(stoich)`**: The scale of the stoichiometric coefficient.
   - **`sign = sign(stoich)`**: Indicates reactant $(-)$ or product $(+)$.
   - **`norm_i`**: The corresponding $\mathbf{B}^{-1}$ or $\mathbf{D}^{-1}$.
   - **`alpha`** (if attention is on): A multiplicative factor from the attention scores.

6. **Concatenation / Averaging Heads**:

   ```python
   if self.concat:
       out = out.view(-1, self.heads * self.out_channels)
   else:
       out = out.mean(dim=1)
   ```

   - Final shape: $[N, \text{heads} \times F_{\mathrm{out}}]$ or $[N, F_{\mathrm{out}}]$.

7. **Adding Bias**:

   ```python
   if self.bias is not None:
       out = out + self.bias
   ```

---

## Variable Mapping to the Code

| **Mathematical Symbol**     | **Code Variable**              | **Meaning**                                                                  |
|-----------------------------|--------------------------------|------------------------------------------------------------------------------|
| $\mathbf{H}$             | `edge_index`                   | Incidence structure between nodes and hyperedges.                            |
| $\mathbf{W}$             | `stoich`                       | Stoichiometric coefficients (signed).                                        |
| $\mathbf{D}^{-1},\mathbf{B}^{-1}$| `D = 1/degree_nodes`, `B = 1/degree_edges`| Degrees computed from absolute stoichiometric values, then inverted.         |
| $\mathbf{X}$             | `x`                            | Node features $[N, F_{\mathrm{in}}]$.                                      |
| $\Theta$                 | `self.lin(...)`                | Linear transformation weights (learned).                                     |
| $\alpha$                 | `alpha`                        | Attention coefficients (if `use_attention=True`).                            |
| $\mathbf{X}'$            | `out`                          | Output node features $[N, F_{\mathrm{out}}]$.                              |

---

## Summary

**`StoichiometricHypergraphConv`** extends a hypergraph convolution to leverage **signed stoichiometric coefficients** for metabolic or reaction-based hypergraph data. It optionally **incorporates an attention mechanism**, where each node-edge interaction is reweighted via learned attention scores.

- **Double Propagation** (node $\to$ edge, then edge $\to$ node) mimics the $\mathbf{H} \mathbf{B}^{-1} \mathbf{H}^\top$ structure.
- **Stoichiometric Coefficients** serve as the edge weights $\mathbf{W}$, providing domain-specific weighting (reactants vs. products).
- **Attention** modulates these base weights if `use_attention=True`.

This operator is especially useful for modeling reaction networks where **directionality** (sign) and **relative magnitude** (stoichiometric ratio) are critical.
