---
id: y4o0nnp86xc58waqi5gu1u8
title: '183354'
desc: ''
updated: 1741308535123
created: 1741307635965
---
## Data

We have the option to add features `x`, but here we don't, and we plan to use learnable embeddings for nodes, that is `nn.embedding`.

These are the edge types.

`(gene, physical_interaction, gene)` - Graph (part of gene multigraph)
`(gene, regulatory_interaction, gene)` -  Graph (part of gene multigraph)
`(metabolite, reaction, metabolite)` - Hypergraph
`(gene, gpr, reaction)` - Bipartite Graph

**Hypergraph**:

For example, in the hypergraph scenario $\mathcal{G}=(\mathcal{V}, \mathcal{E})$ with $\mathcal{V}=\{0,1,2,3\}$ and $\mathcal{E}=\{\{0,1,2\},\{1,2,3\}\}$, the hyperedge_index is represented as:

```python
hyperedge_index = torch.tensor([
    [0, 1, 2, 1, 2, 3], # metabolite
    [0, 0, 0, 1, 1, 1], # reaction
```

**Bipartite Graph**:

```python
x_s = torch. randn(2, 16) #2 nodes. - gene
x_t = torch. randn(3, 16) # 3 nodes. - reaction
edge_index = torch.tensor([
    [0, 0, 1, 1], # gene
    [0, 1, 1, 2], # reaction
])
```

This is the reference data, or wildtype.

```python
cell_graph
HeteroData(
  gene={
    num_nodes=6607,
    node_ids=[6607],
    x=[6607, 0],
  },
  metabolite={
    num_nodes=2534,
    node_ids=[2534],
  },
  reaction={
    num_nodes=4881,
    node_ids=[4881],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 144211],
    num_edges=144211,
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 16095],
    num_edges=16095,
  },
  (metabolite, reaction, metabolite)={
    hyperedge_index=[2, 20960],
    stoichiometry=[20960],
    num_edges=4882,
    reaction_to_genes=dict(len=4881),
    reaction_to_genes_indices=dict(len=4881),
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 5450],
    num_edges=4881,
  }
)
```

Single instance that has perturbations on top of wildtype and associated labels.

```python
dataset[0]
HeteroData(
  gene={
    node_ids=[6605],
    num_nodes=6605,
    ids_pert=[2],
    cell_graph_idx_pert=[2],
    x=[6605, 0],
    x_pert=[2, 0],
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

Batch of data.

```python
batch
HeteroDataBatch(
  gene={
    node_ids=[32],
    num_nodes=211349,
    ids_pert=[32],
    cell_graph_idx_pert=[75],
    x=[211349, 0],
    x_batch=[211349],
    x_ptr=[33],
    x_pert=[75, 0],
    x_pert_batch=[75],
    x_pert_ptr=[33],
    fitness=[32],
    fitness_std=[32],
    gene_interaction=[32],
    gene_interaction_p_value=[32],
    pert_mask=[211424],
    batch=[211349],
    ptr=[33],
  },
  reaction={
    num_nodes=156101,
    node_ids=[32],
    pert_mask=[156192],
    batch=[156101],
    ptr=[33],
  },
  metabolite={
    node_ids=[32],
    num_nodes=81086,
    pert_mask=[81088],
    batch=[81086],
    ptr=[33],
  },
  (gene, physical_interaction, gene)={
    edge_index=[2, 4611926],
    num_edges=[32],
    pert_mask=[4614752],
  },
  (gene, regulatory_interaction, gene)={
    edge_index=[2, 514656],
    num_edges=[32],
    pert_mask=[515040],
  },
  (gene, gpr, reaction)={
    hyperedge_index=[2, 174271],
    num_edges=[32],
    pert_mask=[174400],
  },
  (metabolite, reaction, metabolite)={
    hyperedge_index=[2, 670346],
    stoichiometry=[670346],
    num_edges=[32],
    pert_mask=[670720],
  }
)
```

## Model Idea

```python
hetero_conv = HeteroConv({
    ('paper', 'cites', 'paper'): GCNConv(-1, 64),
    ('author', 'writes', 'paper'): SAGEConv((-1, -1), 64),
    ('paper', 'written_by', 'author'): GATConv((-1, -1), 64),
}, aggr='sum')
```

```python
  HeteroNSA(graphs = [(gene, physical_interaction, gene), "graph",
                      (gene, regulatory_interaction, gene), "graph",
                      (metabolite, reaction, metabolite), "hypergraph",
                      (gene, gpr, reaction), "bipartite"],
            pattern = ["S", "M", "M", "M", "S", "S"]
  )
```

## Hypergraph Stoich Caveat

Currently we include the stoichiometry by weighting the incidence matrix as we know that the incidence matrix has very valuable information about cell physiology. I am not sure how we will resolve this.

```python
class StoichHypergraphConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_stoich_gated: bool = False,
        use_attention: bool = False,
        attention_mode: str = "node",
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(flow="source_to_target", node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_stoich_gated = is_stoich_gated
        self.use_attention = use_attention
        self.attention_mode = attention_mode
        self.heads = heads if use_attention else 1
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Main transformation
        self.lin = Linear(
            in_channels,
            heads * out_channels if use_attention else out_channels,
            bias=False,
            weight_initializer="glorot",
        )

        # Gating network
        if is_stoich_gated:
            self.gate_lin = Linear(
                in_channels, 1, bias=True, weight_initializer="glorot"
            )

        # Attention
        if use_attention:
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.register_parameter("att", None)

        # Bias
        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.is_stoich_gated:
            self.gate_lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        if self.bias is not None:
            zeros(self.bias)

    @disable_dynamic_shapes(required_args=["num_edges"])
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        stoich: Tensor,
        hyperedge_attr: Optional[Tensor] = None,
        num_edges: Optional[int] = None,
    ) -> Tensor:
        num_nodes = x.size(0)
        num_edges = int(edge_index[1].max()) + 1 if num_edges is None else num_edges

        # Transform node features before splitting for attention
        x_transformed = self.lin(x)

        # Handle attention if enabled
        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x_transformed = x_transformed.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)
            x_i = x_transformed[edge_index[0]]
            x_j = hyperedge_attr[edge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(
                alpha,
                edge_index[1] if self.attention_mode == "node" else edge_index[0],
                num_nodes=num_edges if self.attention_mode == "node" else num_nodes,
            )
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Compute gating coefficients if enabled - using original features
        gate_values = None
        if self.is_stoich_gated:
            gate_values = torch.sigmoid(self.gate_lin(x))
            gate_values = gate_values[edge_index[0]]

        # Compute normalization coefficients
        D = scatter(
            torch.abs(stoich), edge_index[0], dim=0, dim_size=num_nodes, reduce="sum"
        )
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(
            torch.abs(stoich), edge_index[1], dim=0, dim_size=num_edges, reduce="sum"
        )
        B = 1.0 / B
        B[B == float("inf")] = 0

        # Message passing
        out = self.propagate(
            edge_index,
            x=x_transformed,
            norm=B,
            alpha=alpha,
            stoich=stoich,
            gate_values=gate_values,
            size=(num_nodes, num_edges),
        )

        # Second message passing step
        out = self.propagate(
            edge_index.flip([0]),
            x=out,
            norm=D,
            alpha=alpha,
            stoich=stoich,
            gate_values=gate_values,
            size=(num_edges, num_nodes),
        )

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(
        self,
        x_j: Tensor,
        norm_i: Tensor,
        alpha: Optional[Tensor],
        stoich: Tensor,
        gate_values: Optional[Tensor],
    ) -> Tensor:
        # Split into magnitude and sign
        magnitude = torch.abs(stoich)
        sign = torch.sign(stoich)

        # Apply gating if enabled
        if gate_values is not None:
            magnitude = magnitude * gate_values.view(-1)

        # Combine all components
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
