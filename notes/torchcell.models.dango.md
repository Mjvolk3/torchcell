---
id: xry94ndvkugglyit6fof4re
title: Dango
desc: ''
updated: 1746326205159
created: 1746161964281
---

**Consolidating algorithm details**

The user wants a very detailed description of the algorithm from the paper "DANGO Predicting higher-order genetic interaction," specifically focusing on functions, layers, loss functions, parameters, and more. They will use this information to write the model in their framework, so I need to be thorough. I'll extract details from the Methods section, which includes pre-training GNN, meta embedding, hypergraph representation, ReZero improvements, and loss definitions. I'll organize this with LaTeX equations for clarity.

Thought for 5 seconds

Below is a detailed, equation‐by‐equation spec of DANGO suitable for implementation. You can paste this into your markdown document.

---

## 1. Definitions

* **Genes**: indexed by $i\in\{1,\dots,G\}$.

* **Trigenic interaction score** $\tau_{ijk}$ for genes $i,j,k$ defined as in Kuzmin *et al.* (2018):

  $$
    \tau_{ijk}
    = f_{ijk}
    - f_i f_j f_k
    - \varepsilon_{jk} f_i
    - \varepsilon_{ik} f_j
    - \varepsilon_{ij} f_k
  $$

  where $\varepsilon_{ij}=f_{ij}-f_i f_j$ and $f_{\cdot}$ are growth‐fitness measurements .

* **Hypergraph** $\mathcal{G}=(V,E)$:
  $V=\{1,\dots,G\}$, and each 3‐gene combination $(i,j,k)$ is a hyperedge $e\in E$.

## 2. Pre‐training GNNs on Pairwise Networks

For each of six STRING PPI networks ($k=1,\dots,6$), learn a 2‑layer GNN mapping node $i$ to embedding $E_i^{(k)}\in\mathbb{R}^D$:

1. **Initial node features**: one–hot embedding

   $$
     H_i^{(0)}\in\mathbb{R}^D
     \quad\text{via shared lookup of size }G\times D.
   $$

2. **Layer $n$ update** ($n=1,2$) for each node $i$:

   $$
     H_{\,\mathcal{N}(i)}^{(n)}
     = \mathop{\mathrm{Average}}\bigl\{H_u^{(n-1)}:u\in\mathcal{N}(i)\bigr\},
   $$

   $$
     H_i^{(n)}
     = \sigma\!\Bigl(W_{\mathrm{GNN}}^{(n)}\,
       \bigl[\,H_i^{(n-1)} \,\|\, H_{\mathcal{N}(i)}^{(n)}\bigr]\Bigr),
   $$

   where “$\|$” is concatenation, $\sigma$ a nonlinearity .

3. **Reconstruction head**: predict row $i$ of adjacency $w^{(k)}_{i\bullet}$ via

   $$
     \widehat w^{(k)}_{i\bullet}
     = \mathrm{FC}\bigl(H_i^{(2)}\bigr),
   $$

   trained with **weighted MSE**:

   $$
     L_i^{(k)}
     = \frac1G\sum_{n=1}^G
       \bigl[\,(x_n-z_n)^2\,\mathbb{I}(z_n\neq0)
         +\lambda\,(x_n-z_n)^2\,\mathbb{I}(z_n=0)\bigr],
   $$

   where $\lambda$ set to 0.1 if >1% zeros vanish in STRING v11 vs v9.1, else 1.0 .

Collect node embeddings $E_i^{(k)}=H_i^{(2)}$.

## 3. Meta‐Embedding Integration

Combine $\{E_i^{(k)}\}_{k=1}^6$ into a single $E_i\in\mathbb{R}^D$ via attention‐weighted sum:

$$
  \alpha_i^{(k)}
  = \exp\!\bigl(\mathrm{MLP}(E_i^{(k)})\bigr)
  \quad,\quad
  E_i
  = \frac{\sum_{k=1}^6 \alpha_i^{(k)}\,E_i^{(k)}}{\sum_{k=1}^6 \alpha_i^{(k)}},
$$

where MLP = two FC layers .

## 4. Hyper‐SAGNN for Trigenic Regression

Given a triplet $(i,j,k)$, retrieve $E_i,E_j,E_k$. Pass each through the same FFN to get **static** embeddings:

$$
  s_i = \sigma\bigl(W_s\,E_i + b_s\bigr),
  \quad
  s_j,s_k\ \text{similarly}.
$$

### 4.1. Self‐Attention with ReZero

Stack two multi‐head self‐attention layers (with $h$ heads), computing **dynamic** embeddings $\tilde d_i$, then apply ReZero:

$$
  \tilde\alpha_{i\gets j}
  = \bigl(W_Q E_i\bigr)^\intercal\bigl(W_K E_j\bigr),
  \quad
  \hat d_i
  = \sigma\!\Bigl(\sum_{j\neq i}
    \mathrm{softmax}_j(\tilde\alpha_{i\gets j})
    W_V E_j\Bigr),
$$

$$
  d_i = \beta\,\hat d_i + E_i
  \quad,\quad
  \beta\ \text{initialized }0,\ \text{learned.}
$$

Repeat for second layer .

### 4.2. Edge Score Computation

For each node $i\in\{i,j,k\}$, compute a scalar

$$
  y_i = \mathrm{FC}\bigl((d_i - s_i)\circ(d_i - s_i)\bigr),
$$

then average:

$$
  \widehat\tau_{ijk}
  = \frac{1}{3}\bigl(y_i + y_j + y_k\bigr).
$$

## 5. Training Objective

Train end‐to‐end (GNNs, MLP, SAGNN) with **log‑cosh** regression loss:

$$
  L = \frac{1}{|E|}\sum_{(i,j,k)\in E}
      \log\!\bigl(\cosh(\widehat\tau_{ijk}-\tau_{ijk})\bigr).
$$

**Notes on hyperparameters:**

* GNN hidden dim $D$: as desired (e.g.\ 128).
* Self‐attention heads $h$, inner dims $d_k=d_v=D$.
* $\beta$ per‐layer initialized 0.
* All FC layers followed by $\sigma=\mathrm{ReLU}$ (or as specified).

---

This spec covers all components—definitions, pre‐training, meta embedding, hypergraph model, and loss—in full mathematical detail.

## Data

This is what batched data looks like.

```python
cell_graph
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
  },
  (gene, string9_1_coexpression, gene)={
    edge_index=[2, 320295],
    num_edges=320295,
  },
  (gene, string9_1_cooccurence, gene)={
    edge_index=[2, 9266],
    num_edges=9266,
  },
  (gene, string9_1_database, gene)={
    edge_index=[2, 40093],
    num_edges=40093,
  },
  (gene, string9_1_experimental, gene)={
    edge_index=[2, 226057],
    num_edges=226057,
  },
  (gene, string9_1_fusion, gene)={
    edge_index=[2, 7965],
    num_edges=7965,
  },
  (gene, string9_1_neighborhood, gene)={
    edge_index=[2, 52208],
    num_edges=52208,
  }
)
dataset[0]
HeteroData(
  gene={
    num_nodes=6607,
    perturbed_genes=[3],
    perturbation_indices=[3],
    pert_mask=[6607],
    mask=[6607],
    phenotype_values=[1],
    phenotype_type_indices=[1],
    phenotype_sample_indices=[1],
    phenotype_types=[1],
    phenotype_stat_values=[1],
    phenotype_stat_type_indices=[1],
    phenotype_stat_sample_indices=[1],
    phenotype_stat_types=[1],
  }
)
batch
HeteroDataBatch(
  gene={
    num_nodes=13214,
    perturbed_genes=[2],
    perturbation_indices=[6],
    pert_mask=[13214],
    mask=[13214],
    phenotype_values=[2],
    phenotype_type_indices=[2],
    phenotype_sample_indices=[2],
    phenotype_types=[2],
    phenotype_stat_values=[2],
    phenotype_stat_type_indices=[2],
    phenotype_stat_sample_indices=[2],
    phenotype_stat_types=[2],
    batch=[13214],
    ptr=[3],
  }
)
```
