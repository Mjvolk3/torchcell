---
id: kc0f661z51mm7gcsvmksxkp
title: Hetero_cell
desc: ''
updated: 1739621742725
created: 1739611656728
---
## 2025.02.12 - Data Masking

```python
cell_graph
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 64],
  },
  metabolite={
    num_nodes=2534,
    node_ids=[2534],
  },
  reaction={
    num_nodes=4881,
    node_ids=[4881],
  },
  (gene, physical_interaction, gene)={ # regular graph 
    edge_index=[2, 144211],
    num_edges=144211,
  },
  (gene, regulatory_interaction, gene)={  # regular graph
    edge_index=[2, 16095],
    num_edges=16095,
  },
  (metabolite, reaction, metabolite)={  # hypergraph
    hyperedge_index=[2, 20960],
    stoichiometry=[20960],
    num_edges=4882,
    reaction_to_genes=dict(len=4881),
    reaction_to_genes_indices=dict(len=4881),
  },
  (gene, gpr, reaction)={ # bipartite
    hyperedge_index=[2, 5450],
    num_edges=4881,
  }
)
```

```python
dataset[0]
HeteroData(
  gene={
    node_ids=[6605],
    num_nodes=6605,
    ids_pert=[2],
    cell_graph_idx_pert=[2],
    x=[6605, 64],
    x_pert=[2, 64],
    fitness=[1],
    fitness_std=[1],
    gene_interaction=[1],
    gene_interaction_p_value=[1],
    pert_mask=[6607],
  },
  reaction={
    num_nodes=4881,
    node_ids=[4881],
    pert_mask=[4881],
  },
  metabolite={
    node_ids=[2534],
    num_nodes=2534,
    pert_mask=[2534],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 144102],
    num_edges=144102,
    pert_mask=[144211],
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 16090],
    num_edges=16090,
    pert_mask=[16095],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5450],
    num_edges=5450,
    pert_mask=[5450],
  },
  (metabolite, reaction, metabolite)={
    hyperedge_index=[2, 20960],
    stoichiometry=[20960],
    num_edges=4882,
    pert_mask=[20960],
  }
)
```

```python
batch
HeteroDataBatch(
  gene={
    node_ids=[4],
    num_nodes=26417,
    ids_pert=[4],
    cell_graph_idx_pert=[11],
    x=[26417, 64],
    x_batch=[26417],
    x_ptr=[5],
    x_pert=[11, 64],
    x_pert_batch=[11],
    x_pert_ptr=[5],
    gene_interaction=[4],
    gene_interaction_p_value=[4],
    fitness=[4],
    fitness_std=[4],
    pert_mask=[26428],
    batch=[26417],
    ptr=[5],
  },
  reaction={
    num_nodes=19523,
    node_ids=[4],
    pert_mask=[19524],
    batch=[19523],
    ptr=[5],
  },
  metabolite={
    node_ids=[4],
    num_nodes=10136,
    pert_mask=[10136],
    batch=[10136],
    ptr=[5],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 576619],
    num_edges=[4],
    pert_mask=[576844],
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 64363],
    num_edges=[4],
    pert_mask=[64380],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 21799],
    num_edges=[4],
    pert_mask=[21800],
  },
  (metabolite, reaction, metabolite)={
    hyperedge_index=[2, 83835],
    stoichiometry=[83835],
    num_edges=[4],
    pert_mask=[83840],
  }
)
```

## 2025.02.15 - Algorithm

## HeteroCell Model Algorithm (Detailed)

This algorithm outlines the forward pass for the HeteroCell model, which is designed to predict two outputs:

- **Fitness** from the intact (perturbed batch) representation.
- **Gene Interaction** from the perturbed node aggregation.

---

### 1. Data Structures

- **Whole Graph (cell_graph):**  
  A HeteroData object representing the unperturbed (wildtype) cell, containing:
  - **Gene Nodes:** \( \mathcal{V}_g \) with features \( \mathbf{X}_g \in \mathbb{R}^{N_g \times h} \)
  - **Reaction Nodes:** \( \mathcal{V}_r \) with features \( \mathbf{X}_r \in \mathbb{R}^{N_r \times h} \)
  - **Metabolite Nodes:** \( \mathcal{V}_m \) with features \( \mathbf{X}_m \in \mathbb{R}^{N_m \times h} \)
  - **Edge Types:**  
    - Gene–gene (physical and regulatory)  
    - Gene→reaction (GPR)  
    - Metabolism hyperedges (with stoichiometry)

- **Batch of Perturbed Graphs (batch):**  
  A HeteroDataBatch object containing multiple perturbed instances. In addition to the regular node features, each batch item includes:
  - Perturbed gene features (`x_pert`)
  - Indices identifying the perturbed nodes (`cell_graph_idx_pert`, `x_pert_ptr`, etc.)

---

### 2. Node Embedding Initialization and Preprocessing

- **Gene Embeddings:**  
  A learnable embedding matrix:
  $$
  \mathbf{E}_g \in \mathbb{R}^{N_g \times h}
  $$
  which is then preprocessed by an MLP:
  $$
  \tilde{\mathbf{E}}_g = \mathrm{PreProcessor}(\mathbf{E}_g)
  $$

- **Reaction and Metabolite Embeddings:**  
  Initialized as:
  $$
  \mathbf{E}_r \in \mathbb{R}^{N_r \times h}, \quad \mathbf{E}_m \in \mathbb{R}^{N_m \times h}
  $$

---

### 3. Heterogeneous Message Passing

For each layer \( l = 1, \dots, L \), node features are updated via a set of heterogeneous convolution operations:

1. **Gene–Gene Interactions:**  
   Using GATv2Conv for both physical and regulatory edges:
   $$
   \mathbf{H}^{(l)}_{g \rightarrow g} = \mathrm{GATv2Conv}\Big(\tilde{\mathbf{E}}_g^{(l-1)}\Big)
   $$

2. **Gene→Reaction Interactions:**  
   Also processed via GATv2Conv:
   $$
   \mathbf{H}^{(l)}_{g \rightarrow r} = \mathrm{GATv2Conv}\Big(\tilde{\mathbf{E}}_g^{(l-1)}\Big)
   $$

3. **Metabolism Hyperedge Processing:**  
   Using StoichHypergraphConv, which incorporates stoichiometry \( S \):
   $$
   \mathbf{H}^{(l)}_{m \rightarrow m} = \mathrm{StoichHypergraphConv}\Big(\mathbf{E}_m^{(l-1)},\, S,\, \mathbf{E}_r^{(l-1)}\Big)
   $$

4. **Aggregation:**  
   For each node type \( i \in \{\text{gene}, \text{reaction}, \text{metabolite}\} \), messages from all edge types are summed:
   $$
   \mathbf{E}_i^{(l)} = \sum_{e \in \mathcal{E}_i} \mathbf{H}^{(l)}_{e}
   $$

The final output for gene nodes after the message passing layers is denoted as:
$$
\mathbf{Z}_g
$$

---

### 4. Global Aggregation

Two types of global pooling are applied using an attentional mechanism:

#### 4.1 Reference (Wildtype) Aggregation

- Process the whole graph to obtain gene features:
  $$
  \mathbf{Z}_g^{\text{cell}}
  $$
- Pool all gene features into a single reference vector:
  $$
  z_w = \mathrm{AttentionalGraphAggregation}\Big(\mathbf{Z}_g^{\text{cell}},\, \mathbf{0}\Big) \quad \in \mathbb{R}^{h}
  $$
  (A dummy index aggregates all nodes.)

#### 4.2 Batch (Intact) Aggregation

- Process the batch of perturbed graphs to obtain gene features:
  $$
  \mathbf{Z}_g^{\text{batch}}
  $$
- Pool per batch item using the provided batch indices:
  $$
  z_i = \mathrm{AttentionalGraphAggregation}\Big(\mathbf{Z}_g^{\text{batch}},\, \text{batch indices}\Big) \quad \in \mathbb{R}^{B \times h}
  $$
  where \( B \) is the batch size.

---

### 5. Perturbed Node Extraction and Aggregation

1. **Extract Perturbed Indices:**  
   For each batch item \( i \) (with \( i = 1,\dots,B \)), extract the set of perturbed gene indices:
   $$
   \mathcal{P}_i \subset \{1,\dots,N_g^{(i)}\}
   $$

2. **Replicate Reference Representation:**  
   Expand the reference vector \( z_w \) to match the batch dimension:
   $$
   z_w^{\text{exp}} \in \mathbb{R}^{B \times h}
   $$
   For each batch item \( i \), replicate \( z_w^{(i)} \) for every perturbed index in \( \mathcal{P}_i \).

3. **Aggregate Perturbed Representations:**  
   Pool the replicated vectors with a dedicated perturbed aggregator:
   $$
   z_p = \mathrm{AttentionalGraphAggregation}\Big(\{z_w^{\text{rep}}(i)\}_{i=1}^B, \text{perturbed batch mapping}\Big) \quad \in \mathbb{R}^{B \times h}
   $$

---

### 6. Prediction Heads

The model uses separate MLPs to predict the two outputs:

- **Fitness Prediction:**  
  The intact (batch) representation \( z_i \) is passed through the fitness head:
  $$
  \hat{y}_{\text{fitness}} = \mathrm{MLP}_{\text{fitness}}(z_i) \quad \in \mathbb{R}^{B \times 1}
  $$

- **Gene Interaction Prediction:**  
  The aggregated perturbed representation \( z_p \) is passed through the gene interaction head:
  $$
  \hat{y}_{\text{gene\_interaction}} = \mathrm{MLP}_{\text{interaction}}(z_p) \quad \in \mathbb{R}^{B \times 1}
  $$

- **Final Output:**  
  The final prediction is the concatenation of the two outputs:
  $$
  \hat{y} = \Big[ \hat{y}_{\text{fitness}} \, \| \, \hat{y}_{\text{gene\_interaction}} \Big] \quad \in \mathbb{R}^{B \times 2}
  $$

---

### 7. Loss and Training

- **Loss Function:**  
  The composite loss is defined as:
  $$
  \mathcal{L} = \mathcal{L}_{\text{MSE}}(y, \hat{y}) + \lambda_1\, \mathcal{L}_{\text{dist}} + \lambda_2\, \mathcal{L}_{\text{supcr}} + \lambda_3\, \mathcal{L}_{\text{cell}}
  $$
  where \( y \) contains the ground-truth fitness and gene interaction labels.

- **Training:**  
  The model is trained using an optimizer (e.g., Adam) to minimize the loss.

---

### 8. Summary of the Forward Pass

1. **Whole Graph Processing:**
   - Input the cell_graph to compute gene embeddings via the PreProcessor and HeteroConv layers.
   - Aggregate these features using the global aggregator to obtain a single reference vector \( z_w \).

2. **Batch Processing:**
   - Process the perturbed batch similarly to compute gene features.
   - Pool these features using the global aggregator (with batch indices) to obtain \( z_i \).

3. **Perturbed Node Aggregation:**
   - For each batch item, extract perturbed gene indices.
   - Replicate the reference vector \( z_w \) to match the number of perturbed nodes.
   - Use the perturbed aggregator to pool these replicated vectors into \( z_p \).

4. **Prediction:**
   - Predict fitness from \( z_i \) using the fitness head.
   - Predict gene interaction from \( z_p \) using the gene interaction head.
   - Concatenate the two outputs to form the final prediction \( \hat{y} \).

This detailed algorithm encapsulates the entire forward pass and prediction strategy of the HeteroCell model.
