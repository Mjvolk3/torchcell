---
id: s37xid4mixmz8z9o1gdbe39
title: Root-TorchCell
desc: ''
updated: 1724621537582
created: 1690296505376
---
# Mathematical Formulation of Heterogeneous Graph SAGPooling

Let's define our heterogeneous graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where $\mathcal{V}$ is the set of nodes and $\mathcal{E}$ is the set of edges. In our case, we have one node type (protein) and two edge types (PPI and regulatory).

## Notation

- $\mathbf{X} \in \mathbb{R}^{N \times F}$: Node feature matrix, where $N$ is the number of nodes and $F$ is the number of features
- $\mathbf{A}_{ppi}, \mathbf{A}_{reg} \in \mathbb{R}^{N \times N}$: Adjacency matrices for PPI and regulatory edges
- $\mathbf{H}^{(l)} \in \mathbb{R}^{N_l \times D}$: Hidden representation at layer $l$, where $N_l$ is the number of nodes at layer $l$ and $D$ is the hidden dimension

## Heterogeneous Graph Convolution

For each layer $l$, we perform heterogeneous graph convolution:

$$
\mathbf{H}^{(l+1)} = \sigma\left(\mathbf{W}_{ppi}^{(l)}\tilde{\mathbf{A}}_{ppi}\mathbf{H}^{(l)} + \mathbf{W}_{reg}^{(l)}\tilde{\mathbf{A}}_{reg}\mathbf{H}^{(l)} + \mathbf{b}^{(l)}\right)
$$

where:
- $\tilde{\mathbf{A}}_{ppi}, \tilde{\mathbf{A}}_{reg}$ are the normalized adjacency matrices
- $\mathbf{W}_{ppi}^{(l)}, \mathbf{W}_{reg}^{(l)} \in \mathbb{R}^{D \times D}$ are learnable weight matrices
- $\mathbf{b}^{(l)} \in \mathbb{R}^D$ is a learnable bias vector
- $\sigma$ is a non-linear activation function (ReLU in our case)

## SAGPooling

After each convolution, we apply SAGPooling:

1. Compute attention scores:

   $$
   \mathbf{s}^{(l)} = \text{GNN}(\mathbf{H}^{(l)}, \tilde{\mathbf{A}}^{(l)})
   $$

   where $\text{GNN}$ is a single layer GNN.

2. Select top-k nodes:

   $$
   \text{idx} = \text{topk}(\mathbf{s}^{(l)}, \lfloor k \cdot N_l \rfloor)
   $$

   where $k$ is the pooling ratio (0.8 in our case).

3. Pool nodes and edges:

   $$
   \mathbf{H}^{(l+1)} = (\mathbf{H}^{(l)} \odot \tanh(\mathbf{s}^{(l)}))_{\text{idx}}
   $$

   $$
   \tilde{\mathbf{A}}^{(l+1)} = \tilde{\mathbf{A}}^{(l)}_{\text{idx}, \text{idx}}
   $$

   where $\odot$ denotes element-wise multiplication.

## Overall Process

For each layer $l$ from 1 to 13:

1. Apply heterogeneous graph convolution:

   $$
   \mathbf{H}^{(l)} = \text{HeteroGNNConv}(\mathbf{H}^{(l-1)}, \tilde{\mathbf{A}}_{ppi}^{(l-1)}, \tilde{\mathbf{A}}_{reg}^{(l-1)})
   $$

2. Apply SAGPooling:

   $$
   \mathbf{H}^{(l)}, \tilde{\mathbf{A}}_{ppi}^{(l)}, \tilde{\mathbf{A}}_{reg}^{(l)} = \text{SAGPool}(\mathbf{H}^{(l)}, \tilde{\mathbf{A}}_{ppi}^{(l-1)}, \tilde{\mathbf{A}}_{reg}^{(l-1)})
   $$

3. If $N_l \leq 1$, break the loop

## Final Classification

After all pooling layers:

1. Global mean pooling:

   $$
   \mathbf{h} = \frac{1}{N_{13}}\sum_{i=1}^{N_{13}} \mathbf{H}^{(13)}_i
   $$

2. Classification:

   $$
   \mathbf{y} = \text{softmax}(\mathbf{W}\mathbf{h} + \mathbf{b})
   $$

   where $\mathbf{W} \in \mathbb{R}^{C \times D}$ and $\mathbf{b} \in \mathbb{R}^C$ are learnable parameters, and $C$ is the number of classes.

This formulation describes the flow of data through our heterogeneous graph neural network with 13 rounds of SAGPooling, from the initial input to the final classification output.