# HeteroCellBipartiteDangoGI Model Refactoring Plan

## Current Status Summary

### Context

- Working on updating `hetero_cell_bipartite_dango_gi.py` to support dynamic graph types
- The model needs to handle any kind of graph dynamically, not just physical and regulatory interactions
- Switching to COO-based format for future multimodal phenotype prediction
- Using `load_batch_005.py` for data loading with new batch format. We are specifically using "hetero_cell_bipartite" config.

### Key Files

1. **Main Model File**: `/Users/michaelvolk/Documents/projects/torchcell/torchcell/models/hetero_cell_bipartite_dango_gi.py`
2. **Data Loader**: `/Users/michaelvolk/Documents/projects/torchcell/torchcell/scratch/load_batch_005.py`
3. **Graph Module**: `/Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/graph.py`
4. **Graph Processor**: `/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/graph_processor.py`
5. **Config File**: `/Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/conf/hetero_cell_bipartite_dango_gi.yaml`
6. **Sample Data**: `/Users/michaelvolk/Documents/projects/torchcell/notes/torchcell.models.hetero_cell_bipartite_dango_gi.md`
7. **Dango Model**: `/Users/michaelvolk/Documents/projects/torchcell/notes/torchcell.models.dango.md`

### Completed Tasks

1. Added import for `GeneMultiGraph` from `torchcell.graph.graph`
2. Updated main function to use correct parameters for `load_sample_data_batch`
   - Changed from `metabolism_graph="metabolism_bipartite"` to `config="hetero_cell_bipartite"`
3. Started updating `GeneInteractionDango` class to accept `GeneMultiGraph` parameter

## Remaining Tasks

### 1. Complete GeneInteractionDango Class Update

- [ ] Finish updating `__init__` method to accept `gene_multigraph: GeneMultiGraph` parameter
- [ ] Extract graph names dynamically: `self.graph_names = list(gene_multigraph.keys())`
- [ ] Create GATv2Conv layers dynamically for each graph in the multigraph
- [ ] Update the convolution layer creation loop to use dynamic graph names

### 2. Update forward_single Method

- [ ] Modify edge processing to handle dynamic edge types based on graph names
- [ ] Replace hardcoded edge type checks with dynamic iteration over `self.graph_names`
- [ ] Update edge_index_dict construction to use dynamic graph names

### 3. Update Main Function

- [ ] Create a `GeneMultiGraph` object using `build_gene_multigraph`
- [ ] Pass the multigraph to the model initialization
- [ ] Ensure the graph names from config are used correctly

### 4. Handle Data Structure Updates

- [ ] Ensure compatibility with new COO format for phenotype data. There is only one phenotype per dat instance and is gene interactions.
- [ ] Verify the model works with the new batch structure from `load_batch_005`
- [ ] Test with different graph combinations (STRING networks, TFLink, etc.)... I will test this by updating the config manually, you don't need to run anything.

## Implementation Plan

### Step 1: Complete __init__ Method Update

```python
def __init__(
    self,
    gene_num: int,
    hidden_channels: int,
    num_layers: int,
    gene_multigraph: GeneMultiGraph,
    dropout: float = 0.1,
    norm: str = "layer",
    activation: str = "relu",
    gene_encoder_config: Optional[Dict[str, Any]] = None,
):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.gene_multigraph = gene_multigraph
    
    # Extract graph names from the multigraph
    self.graph_names = list(gene_multigraph.keys())
    
    # ... rest of init code ...
    
    # Graph convolution layers
    self.convs = nn.ModuleList()
    for _ in range(num_layers):
        conv_dict = {}
        
        # Create a GATv2Conv for each graph in the multigraph
        for graph_name in self.graph_names:
            edge_type = ("gene", graph_name, "gene")
            conv_dict[edge_type] = GATv2Conv(
                hidden_channels,
                hidden_channels // gene_encoder_config.get("heads", 1),
                heads=gene_encoder_config.get("heads", 1),
                concat=gene_encoder_config.get("concat", True),
                add_self_loops=gene_encoder_config.get("add_self_loops", False),
            )
        
        # Wrap convolutions...
```

### Step 2: Update forward_single Method

```python
def forward_single(self, data: HeteroData | Batch) -> torch.Tensor:
    # ... existing code ...
    
    # Process edge indices dynamically
    edge_index_dict = {}
    
    for graph_name in self.graph_names:
        edge_type = ("gene", graph_name, "gene")
        if edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index.to(device)
            edge_index_dict[edge_type] = edge_index
    
    # Apply convolution layers
    for conv in self.convs:
        x_dict = conv(x_dict, edge_index_dict)
    
    return x_dict["gene"]
```

### Step 3: Update Main Function

```python
def main(cfg: DictConfig) -> None:
    # ... existing setup code ...
    
    # Load data
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=cfg.data_module.batch_size,
        num_workers=cfg.data_module.num_workers,
        config="hetero_cell_bipartite",
        is_dense=False,
    )
    
    # Build gene multigraph
    from torchcell.graph.graph import build_gene_multigraph, SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    
    # Initialize genome and graph
    genome = SCerevisiaeGenome(...)  # Add proper paths
    sc_graph = SCerevisiaeGraph(genome=genome, ...)
    
    # Get graph names from config or use defaults
    graph_names = cfg.cell_dataset.get("graphs", ["physical", "regulatory"])
    gene_multigraph = build_gene_multigraph(sc_graph, graph_names)
    
    # Initialize model with multigraph
    model = GeneInteractionDango(
        gene_num=cfg.model.gene_num,
        hidden_channels=cfg.model.hidden_channels,
        num_layers=cfg.model.num_layers,
        gene_multigraph=gene_multigraph,
        dropout=cfg.model.dropout,
        norm=cfg.model.norm,
        activation=cfg.model.activation,
        gene_encoder_config=cfg.model.gene_encoder_config,
    ).to(device)
    
    # ... rest of training code ...
```

## Testing Strategy

I will run the tests manually for different graph types.

1. Test with default physical and regulatory graphs
2. Test with STRING network graphs
3. Test with TFLink regulatory network
4. Verify COO format phenotype data is handled correctly
5. Check memory usage and performance with different graph combinations

## Notes

- The model should be flexible enough to handle any valid graph type from `SCEREVISIAE_GENE_GRAPH_MAP`
- ~~Maintain backward compatibility where possible~~. We actually do not care about backwards compatibility.
- Focus on minimal changes to achieve dynamic graph support
- Ensure the model can handle missing graph types gracefully
- Ensure that we are still using tensor processing, in the forward method we DO NOT want to loop over instances in a batch. We need to be using pytorch tensor processing.
- We are using a Dango-like prediction head for predicting gene interactions. After we get things working we will make our current prediction head look more like HyperSAGNN with more parameterization for self-attention `num_layers`, `num_heads`, etc.
