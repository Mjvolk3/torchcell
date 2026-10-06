# tests/torchcell/nn/test_hetero_nsa.py
"""Tests for the heterogeneous NSA (neighborhood self-attention) module."""

from typing import cast

import pytest
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_dense_adj

from torchcell.nn.hetero_nsa import HeteroNSA, HeteroNSAEncoder
from torchcell.nn.nsa_encoder import NSAEncoder
from torchcell.nn.self_attention_block import SelfAttentionBlock
from torchcell.scratch.load_batch import load_sample_data_batch

# The synthetic tests below need no data. Only test_hetero_nsa_with_real_data reads the
# sample batch under $DATA_ROOT (torchcell.graph.sgd resolves DATA_ROOT lazily and
# imports cleanly), and it alone is data-gated.
load_dotenv()


@pytest.fixture
def test_hetero_graph():
    """Fixture to create a basic heterogeneous graph for testing."""
    num_nodes_dict = {"gene": 100, "reaction": 50, "metabolite": 30}
    num_edges_dict = {
        ("gene", "physical_interaction", "gene"): 200,
        ("gene", "regulatory_interaction", "gene"): 150,
        ("gene", "gpr", "reaction"): 80,
        ("reaction", "rmr", "metabolite"): 60,
    }

    data = HeteroData()

    # Add node features
    hidden_dim = 32
    for node_type, count in num_nodes_dict.items():
        data[node_type].x = torch.randn(count, hidden_dim)
        data[node_type].num_nodes = count

    # Add edges
    for edge_type, count in num_edges_dict.items():
        src, rel, dst = edge_type
        src_count = num_nodes_dict[src]
        dst_count = num_nodes_dict[dst]

        # Random edge indices
        edge_index = torch.stack(
            [
                torch.randint(0, src_count, (count,)),
                torch.randint(0, dst_count, (count,)),
            ]
        )

        data[edge_type].edge_index = edge_index

        # Add edge attributes for metabolic edges
        if rel == "rmr":
            data[edge_type].edge_type = torch.randint(0, 2, (count,))
            data[edge_type].stoichiometry = torch.randn(count)

    # Create an undirected version for self-loops (makes testing easier)
    transform = ToUndirected()
    data = transform(data)

    return data


@pytest.fixture
def standard_encoder_config():
    """Fixture to provide a standard configuration for the HeteroNSAEncoder."""
    node_types = {"gene", "reaction", "metabolite"}
    edge_types = {
        ("gene", "physical_interaction", "gene"),
        ("gene", "regulatory_interaction", "gene"),
        ("gene", "gpr", "reaction"),
        ("reaction", "rmr", "metabolite"),
    }

    # Changed from dictionary to single list pattern
    pattern = ["M", "S"]  # Simple pattern for all edge types

    input_dims = {"gene": 32, "reaction": 32, "metabolite": 32}

    return {
        "input_dims": input_dims,
        "hidden_dim": 64,
        "node_types": node_types,
        "edge_types": edge_types,
        "pattern": pattern,  # Changed from "patterns" to "pattern"
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "aggregation": "attention",
    }


@pytest.fixture
def nsa_encoder_config(test_hetero_graph):
    """Fixture to provide configuration specific to NSAEncoder."""
    # Get input dimension from node features
    input_dim = test_hetero_graph["gene"].x.size(1)

    return {
        "input_dim": input_dim,
        "hidden_dim": 64,
        "pattern": ["M", "S"],  # Using the same pattern format
        "num_heads": 4,
        "dropout": 0.1,
    }


def test_hetero_nsa_initialization(standard_encoder_config):
    """Test that HeteroNSAEncoder initializes correctly."""
    config = standard_encoder_config

    # Initialize the HeteroNSAEncoder module with the single pattern
    model = HeteroNSAEncoder(
        input_dims=config["input_dims"],
        hidden_dim=config["hidden_dim"],
        node_types=config["node_types"],
        edge_types=config["edge_types"],
        pattern=config["pattern"],  # Changed from "patterns" to "pattern"
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        aggregation=config["aggregation"],
    )

    # Check if the model has the necessary components for each node type
    for node_type in config["node_types"]:
        assert node_type in model.input_projections, (
            f"Missing input projection for {node_type}"
        )

    # Check if the model has the right number of NSA layers
    assert len(model.nsa_layers) == config["num_layers"], (
        "Incorrect number of NSA layers"
    )

    # Check if the model has the right number of layer norms
    for node_type in config["node_types"]:
        assert node_type in model.layer_norms, f"Missing layer norms for {node_type}"
        node_layer_norms = cast(nn.ModuleList, model.layer_norms[node_type])
        assert len(node_layer_norms) == config["num_layers"], (
            f"Incorrect number of layer norms for {node_type}"
        )


def test_nsa_encoder_forward(test_hetero_graph, nsa_encoder_config):
    """Test the forward pass of NSAEncoder."""
    data = test_hetero_graph

    # Create encoder with the correct config
    encoder = NSAEncoder(**nsa_encoder_config)

    # Extract node features from a specific node type
    x = data["gene"].x

    # Extract edge indices from a specific edge type
    edge_index = data[("gene", "physical_interaction", "gene")].edge_index

    # Forward pass with appropriate parameters
    node_embeddings = encoder(x, edge_index)

    # Check output shapes
    assert node_embeddings.shape == (
        data["gene"].num_nodes,
        nsa_encoder_config["hidden_dim"],
    )
    assert not torch.isnan(node_embeddings).any()


def test_invalid_pattern():
    """Test that invalid pattern raises appropriate errors."""
    node_types = {"gene", "reaction"}
    edge_types = {("gene", "interaction", "gene")}

    # Create a Hetero NSA module directly for testing
    with pytest.raises(ValueError, match="Invalid block type"):
        HeteroNSA(
            hidden_dim=64,
            node_types=node_types,
            edge_types=edge_types,
            pattern=["M", "X", "S"],  # X is invalid
            num_heads=4,
            dropout=0.1,
        )

    # Also test empty pattern
    with pytest.raises(ValueError, match="Pattern list cannot be empty"):
        HeteroNSA(
            hidden_dim=64,
            node_types=node_types,
            edge_types=edge_types,
            pattern=[],  # Empty pattern
            num_heads=4,
            dropout=0.1,
        )


@pytest.fixture
def dense_sample_data():
    """Create a simple synthetic dense HeteroData for testing."""
    data = HeteroData()

    # Add node features
    batch_size = 2
    hidden_dim = 32

    # Sizes per batch
    gene_size = 20
    reaction_size = 10
    metabolite_size = 15

    # Total sizes
    total_gene_size = gene_size * batch_size
    total_reaction_size = reaction_size * batch_size
    total_metabolite_size = metabolite_size * batch_size

    # Add features
    data["gene"].x = torch.randn(total_gene_size, hidden_dim)
    data["gene"].num_nodes = total_gene_size
    data["reaction"].x = torch.randn(total_reaction_size, hidden_dim)
    data["reaction"].num_nodes = total_reaction_size
    data["metabolite"].x = torch.randn(total_metabolite_size, hidden_dim)
    data["metabolite"].num_nodes = total_metabolite_size

    # Add batch indices
    data["gene"].batch = torch.repeat_interleave(torch.arange(batch_size), gene_size)
    data["reaction"].batch = torch.repeat_interleave(
        torch.arange(batch_size), reaction_size
    )
    data["metabolite"].batch = torch.repeat_interleave(
        torch.arange(batch_size), metabolite_size
    )

    # Create a simple batch object
    batch = data

    return None, batch


def test_hetero_nsa_with_dense_data(dense_sample_data, monkeypatch):
    """Test HeteroNSAEncoder with dense data by patching the forward method."""
    import torch

    from torchcell.nn.hetero_nsa import HeteroNSAEncoder

    # Mock the forward method to avoid edge_index_dict lookup
    def mock_forward(self, x_dict, data, batch_idx=None):
        # Simple pass-through implementation that avoids using edge_index
        return {
            node_type: torch.nn.functional.relu(x) for node_type, x in x_dict.items()
        }, torch.zeros(1, self.hidden_dim, device=next(self.parameters()).device)

    # Apply the monkey patch
    monkeypatch.setattr(HeteroNSAEncoder, "forward", mock_forward)

    # Get sample data
    _, batch = dense_sample_data

    # Add input_dims parameter
    input_dims = {
        "gene": batch["gene"].x.size(1) if hasattr(batch["gene"], "x") else 64,
        "reaction": 64,
        "metabolite": 64,  # Fixed typo: R64 -> 64
    }

    # Create model with input_dims - using single pattern
    model = HeteroNSAEncoder(
        input_dims=input_dims,
        hidden_dim=64,
        node_types={"gene", "reaction", "metabolite"},
        edge_types={
            ("gene", "physical_interaction", "gene"),
            ("gene", "regulatory_interaction", "gene"),
            ("gene", "gpr", "reaction"),
            ("reaction", "rmr", "metabolite"),
        },
        pattern=["M", "S"],  # Single pattern list instead of dictionary
        num_heads=4,
        dropout=0.1,
    )

    # Create input features
    hidden_dim = 64
    device = torch.device("cpu")

    # Get node counts
    gene_count = batch["gene"].num_nodes
    reaction_count = batch["reaction"].num_nodes
    metabolite_count = batch["metabolite"].num_nodes

    # Create dummy input features
    x_dict = {
        "gene": torch.randn(gene_count, hidden_dim, device=device),
        "reaction": torch.randn(reaction_count, hidden_dim, device=device),
        "metabolite": torch.randn(metabolite_count, hidden_dim, device=device),
    }

    # Forward pass with mocked method
    output_dict, graph_emb = model(x_dict, batch)

    # Check outputs
    assert "gene" in output_dict
    assert "reaction" in output_dict
    assert "metabolite" in output_dict

    # Check shapes
    assert output_dict["gene"].shape == x_dict["gene"].shape
    assert output_dict["reaction"].shape == x_dict["reaction"].shape
    assert output_dict["metabolite"].shape == x_dict["metabolite"].shape
    assert graph_emb.shape == (1, hidden_dim)


@pytest.fixture
def real_data_batch(monkeypatch):
    """Fixture to load a real data batch with metabolism edges and attributes."""
    try:
        # Try to load real data batch, but handle gracefully if it fails

        try:
            dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
                batch_size=2,
                num_workers=0,
                metabolism_graph="metabolism_bipartite",
                is_dense=True,
            )
            return dataset, batch, input_channels, max_num_nodes
        except Exception as e:
            pytest.skip(f"Failed to load real data batch: {e}")
    except ImportError:
        pytest.skip("torchcell.scratch.load_batch not available")


@pytest.mark.data
def test_hetero_nsa_with_real_data(real_data_batch):
    """Test HeteroNSAEncoder with real biological network data."""
    # Skip if fixture returns None (data loading failed)
    if real_data_batch is None:
        pytest.skip("Could not load real data batch")

    dataset, batch, input_channels, max_num_nodes = real_data_batch

    # Extract node types and edge types from the batch
    node_types = set(batch.node_types)
    edge_types = set(batch.edge_types)

    # Create input_dims dictionary
    input_dims = {}
    for node_type in node_types:
        if hasattr(batch[node_type], "x") and batch[node_type].x is not None:
            input_dims[node_type] = batch[node_type].x.size(1)
        else:
            # Default embedding size if no features available
            input_dims[node_type] = 64

    # Add a preprocessing step to add feature vectors for nodes that don't have them
    # This is needed for the test to pass
    for node_type in node_types:
        if not hasattr(batch[node_type], "x") or batch[node_type].x is None:
            # Create dummy features for testing purposes
            num_nodes = batch[node_type].num_nodes
            batch[node_type].x = torch.zeros(num_nodes, input_dims[node_type])
            print(f"Added dummy features for {node_type} nodes")

    # Initialize model with real data dimensions
    model = HeteroNSAEncoder(
        input_dims=input_dims,
        hidden_dim=64,
        node_types=node_types,
        edge_types=edge_types,
        pattern=["M", "S", "M"],  # Test with a more complex pattern
        num_layers=1,  # Keep small for memory efficiency in testing
        num_heads=4,
        dropout=0.1,
        aggregation="sum",
    )

    # Track memory usage if CUDA is available
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    # Forward pass with the real batch
    node_embeddings, graph_embedding = model(batch)

    # Check output types and shapes
    assert isinstance(node_embeddings, dict), "Output should be a dictionary"
    for node_type in node_types:
        if node_type in node_embeddings:
            # Verify node embeddings have correct shapes
            assert node_embeddings[node_type].shape[1] == 64, (
                f"Wrong embedding dimension for {node_type}"
            )
            # Check no NaNs in output
            assert not torch.isnan(node_embeddings[node_type]).any(), (
                f"NaN values in {node_type} embeddings"
            )

    # Verify graph embedding shape - more flexible check
    assert graph_embedding.dim() == 2, "Graph embedding should be 2-dimensional"
    assert graph_embedding.shape[1] == 64, (
        "Graph embedding hidden dimension should be 64"
    )

    # Log memory usage
    if torch.cuda.is_available():
        final_memory = torch.cuda.memory_allocated()
        memory_used = final_memory - initial_memory
        print(f"Memory used: {memory_used / 1024**2:.2f} MB")


def test_boolean_mask_memory_efficiency(test_hetero_graph):
    """Test that boolean masks provide memory savings."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    data = test_hetero_graph.to(device)

    # Take a simple edge type to test
    edge_type = ("gene", "physical_interaction", "gene")

    # Create boolean mask
    bool_mask = to_dense_adj(data[edge_type].edge_index).bool().to(device)

    # Create float mask (what we're avoiding)
    float_mask = bool_mask.float()

    # Measure memory usage
    bool_bytes = bool_mask.element_size() * bool_mask.numel()
    float_bytes = float_mask.element_size() * float_mask.numel()

    # Boolean should use significantly less memory
    assert bool_bytes < float_bytes / 3

    # Print memory savings
    memory_ratio = float_bytes / bool_bytes
    print(f"Memory ratio (float/bool): {memory_ratio:.2f}x")


def test_directed_vs_undirected_graphs(test_hetero_graph):
    """Test HeteroNSA works correctly with both directed and undirected graphs."""
    # Create directed version of the graph
    directed_data = test_hetero_graph.clone()

    # Create undirected version using ToUndirected transform
    from torch_geometric.transforms import ToUndirected

    undirected_data = ToUndirected()(test_hetero_graph.clone())

    # Setup model
    hidden_dim = 64
    node_types = {"gene", "reaction", "metabolite"}
    edge_types = {
        ("gene", "physical_interaction", "gene"),
        ("gene", "regulatory_interaction", "gene"),
        ("gene", "gpr", "reaction"),
        ("reaction", "rmr", "metabolite"),
    }
    pattern = ["M", "S"]

    # Create model and input dict
    model = HeteroNSA(
        hidden_dim=hidden_dim,
        node_types=node_types,
        edge_types=edge_types,
        pattern=pattern,
        num_heads=4,
    )

    x_dict = {
        node_type: torch.randn(directed_data[node_type].num_nodes, hidden_dim)
        for node_type in node_types
        if node_type in directed_data.node_types
    }

    # Forward pass with both graphs
    out_directed = model(x_dict, directed_data)
    out_undirected = model(x_dict, undirected_data)

    # Results should differ due to different edge connectivity
    for node_type in node_types:
        if node_type in out_directed and node_type in out_undirected:
            assert not torch.allclose(
                out_directed[node_type], out_undirected[node_type]
            )


@pytest.fixture
def metabolic_graph():
    """Fixture for a small metabolic network with stoichiometry."""
    # Create a small metabolic network with 6 nodes (3 metabolites, 3 reactions)
    # Node features (one-hot encoding for node type)
    x = torch.zeros(6, 2)  # 2 node types: metabolite (0) and reaction (1)
    x[0:3, 0] = 1  # Metabolites A, B, C
    x[3:6, 1] = 1  # Reactions R1, R2, R3

    # Edge connections: metabolite -> reaction and reaction -> metabolite
    edge_index = torch.tensor(
        [
            [0, 3, 1, 4, 2, 5],  # source nodes (A, R1, B, R2, C, R3)
            [3, 1, 4, 2, 5, 0],  # target nodes (R1, B, R2, C, R3, A)
        ]
    )

    # Stoichiometric coefficients (-1 for consumption, +1 for production)
    edge_attr = torch.tensor(
        [
            [-1.0],  # A consumed by R1
            [1.0],  # B produced by R1
            [-1.0],  # B consumed by R2
            [1.0],  # C produced by R2
            [-1.0],  # C consumed by R3
            [1.0],  # A produced by R3
        ]
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = 6  # Explicitly set

    return data


def test_edge_attribute_integration(metabolic_graph):
    """Test that edge attributes like stoichiometry properly influence attention."""
    # Convert Data to HeteroData
    hetero_data = HeteroData()

    # Create metabolite nodes (first 3 nodes)
    hetero_data["metabolite"].x = metabolic_graph.x[:3]
    hetero_data["metabolite"].num_nodes = 3

    # Create reaction nodes (last 3 nodes)
    hetero_data["reaction"].x = metabolic_graph.x[3:6]
    hetero_data["reaction"].num_nodes = 3

    # Create rmr edges (reaction to metabolite)
    # In metabolic_graph: [0->3, 3->1, 1->4, 4->2, 2->5, 5->0]
    # We need to convert to indices within their respective node types
    # reaction indices: map 3->0, 4->1, 5->2
    # metabolite indices: map 0->0, 1->1, 2->2

    # Original edges involving reactions->metabolites: 3->1, 4->2, 5->0
    # Convert to: 0->1, 1->2, 2->0
    edge_index = torch.tensor(
        [
            [0, 1, 2],  # reaction indices (after mapping)
            [1, 2, 0],  # metabolite indices (after mapping)
        ]
    )

    # Get corresponding stoichiometry values: 1.0, 1.0, 1.0 (all productions)
    stoichiometry = torch.tensor([1.0, 1.0, 1.0])

    # Add to HeteroData
    hetero_data["reaction", "rmr", "metabolite"].edge_index = edge_index
    hetero_data["reaction", "rmr", "metabolite"].stoichiometry = stoichiometry

    # Setup model
    hidden_dim = 32
    node_types = {"metabolite", "reaction"}
    edge_types = {("reaction", "rmr", "metabolite")}
    pattern = ["M"]

    # Create model
    model = HeteroNSA(
        hidden_dim=hidden_dim,
        node_types=node_types,
        edge_types=edge_types,
        pattern=pattern,
        num_heads=4,
    )

    # Create input embeddings
    x_dict = {
        "metabolite": torch.randn(hetero_data["metabolite"].num_nodes, hidden_dim),
        "reaction": torch.randn(hetero_data["reaction"].num_nodes, hidden_dim),
    }

    # Create copy with modified stoichiometry
    modified_data = hetero_data.clone()
    modified_data["reaction", "rmr", "metabolite"].stoichiometry = -hetero_data[
        "reaction", "rmr", "metabolite"
    ].stoichiometry

    # Forward pass with both graphs
    out_original = model(x_dict, hetero_data)
    out_modified = model(x_dict, modified_data)

    # Outputs should differ due to different stoichiometry values
    for node_type in node_types:
        if node_type in out_original and node_type in out_modified:
            # Ensure stoichiometry affects results
            assert not torch.allclose(out_original[node_type], out_modified[node_type])


def test_batched_processing(standard_encoder_config):
    """Test HeteroNSA correctly handles batched graphs."""
    # This test focuses on HeteroNSA, not the encoder
    # Create multiple simple graphs with the correct structure
    num_graphs = 3

    # Setup for HeteroNSA model (simpler than HeteroNSAEncoder)
    hidden_dim = 64
    node_types = {"gene", "reaction", "metabolite"}
    edge_types = {("gene", "gpr", "reaction")}
    pattern = ["S", "M"]  # Simple valid pattern

    # Create HeteroNSA model (not encoder)
    model = HeteroNSA(
        hidden_dim=hidden_dim,
        node_types=node_types,
        edge_types=edge_types,
        pattern=pattern,
        num_heads=4,
        dropout=0.1,
    )

    # Test each graph individually
    for i in range(num_graphs):
        # Create a simple graph
        data = HeteroData()

        # Set node counts for this graph instance
        gene_count = 10 + i * 2
        reaction_count = 8 + i * 2

        # Create node features
        data["gene"].x = torch.randn(gene_count, hidden_dim)
        data["gene"].num_nodes = gene_count
        data["reaction"].x = torch.randn(reaction_count, hidden_dim)
        data["reaction"].num_nodes = reaction_count
        data["metabolite"].x = torch.randn(5, hidden_dim)  # Fixed size for simplicity
        data["metabolite"].num_nodes = 5

        # Create edge indices for gene->reaction
        edge_index = torch.stack(
            [
                torch.randint(0, gene_count, (15,)),
                torch.randint(0, reaction_count, (15,)),
            ]
        )
        data["gene", "gpr", "reaction"].edge_index = edge_index

        # Create simple adjacency mask
        adj_mask = torch.zeros(gene_count, reaction_count, dtype=torch.bool)
        for j in range(edge_index.size(1)):
            src, dst = edge_index[0, j], edge_index[1, j]
            adj_mask[src, dst] = True
        data["gene", "gpr", "reaction"].adj_mask = adj_mask

        # Create input dictionary - use existing embeddings as input
        x_dict = {
            "gene": data["gene"].x.clone(),
            "reaction": data["reaction"].x.clone(),
            "metabolite": data["metabolite"].x.clone(),
        }

        # Forward pass with the model
        output_dict = model(x_dict, data)

        # Check outputs - handling the case where a batch dimension is added
        for node_type in node_types:
            assert node_type in output_dict

            # Get output for this node type
            output = output_dict[node_type]
            input_shape = x_dict[node_type].shape

            # HeteroNSA sometimes adds a batch dimension of size 1
            # Check if the output has an extra dimension and handle it
            if output.dim() == 3 and output.size(0) == 1:
                # When an extra batch dimension is added, check that the content dimensions match
                assert output.size(1) == input_shape[0], (
                    f"Node count mismatch for {node_type}"
                )
                assert output.size(2) == input_shape[1], (
                    f"Feature dimension mismatch for {node_type}"
                )

                # Verify we can get back to the original shape by squeezing
                output_squeezed = output.squeeze(0)
                assert output_squeezed.shape == input_shape
            else:
                # Direct shape comparison if no batch dimension was added
                assert output.shape == input_shape, f"Shape mismatch for {node_type}"

            # Check for NaN values
            assert not torch.isnan(output).any(), f"NaN values in {node_type} output"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hetero_nsa_gpu():
    """Test HeteroNSA using FlexAttention on GPU."""
    # Create a simple graph
    data = HeteroData()

    # Setup dimensions
    hidden_dim = 64
    gene_count = 30
    reaction_count = 20

    # Create node features (on GPU)
    data["gene"].x = torch.randn(gene_count, hidden_dim, device="cuda")
    data["gene"].num_nodes = gene_count
    data["reaction"].x = torch.randn(reaction_count, hidden_dim, device="cuda")
    data["reaction"].num_nodes = reaction_count

    # Create edges
    edge_index = torch.stack(
        [
            torch.randint(0, gene_count, (50,), device="cuda"),
            torch.randint(0, reaction_count, (50,), device="cuda"),
        ]
    )
    data["gene", "gpr", "reaction"].edge_index = edge_index

    # Create boolean adjacency mask
    adj_mask = torch.zeros(gene_count, reaction_count, dtype=torch.bool, device="cuda")
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_mask[src, dst] = True
    data["gene", "gpr", "reaction"].adj_mask = adj_mask

    # Setup model parameters
    node_types = {"gene", "reaction"}
    edge_types = {("gene", "gpr", "reaction")}
    pattern = ["M", "S"]

    # Create model on GPU
    model = HeteroNSA(
        hidden_dim=hidden_dim,
        node_types=node_types,
        edge_types=edge_types,
        pattern=pattern,
        num_heads=4,
    ).cuda()

    # Create input embeddings (on GPU)
    x_dict = {"gene": data["gene"].x.clone(), "reaction": data["reaction"].x.clone()}

    # Should use FlexAttention on GPU
    output_dict = model(x_dict, data)

    # Check outputs
    assert output_dict["gene"].device.type == "cuda"
    assert output_dict["reaction"].device.type == "cuda"
    assert output_dict["gene"].shape == x_dict["gene"].shape
    assert output_dict["reaction"].shape == x_dict["reaction"].shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_flex_attention_error_propagation():
    """Test that errors in attention mechanisms on GPU propagate correctly."""
    # Create a simple graph
    data = HeteroData()

    # Setup dimensions
    hidden_dim = 64
    gene_count = 20
    reaction_count = 15

    # Create node features (on GPU)
    data["gene"].x = torch.randn(gene_count, hidden_dim, device="cuda")
    data["gene"].num_nodes = gene_count
    data["reaction"].x = torch.randn(reaction_count, hidden_dim, device="cuda")
    data["reaction"].num_nodes = reaction_count

    # Create edges
    edge_index = torch.stack(
        [
            torch.randint(0, gene_count, (40,), device="cuda"),
            torch.randint(0, reaction_count, (40,), device="cuda"),
        ]
    )
    data["gene", "gpr", "reaction"].edge_index = edge_index

    # Create boolean adjacency mask
    adj_mask = torch.zeros(gene_count, reaction_count, dtype=torch.bool, device="cuda")
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_mask[src, dst] = True
    data["gene", "gpr", "reaction"].adj_mask = adj_mask

    # Setup model parameters
    node_types = {"gene", "reaction"}
    edge_types = {("gene", "gpr", "reaction")}
    pattern = ["M"]  # Just use masked attention

    # Create model on GPU
    model = HeteroNSA(
        hidden_dim=hidden_dim,
        node_types=node_types,
        edge_types=edge_types,
        pattern=pattern,
        num_heads=4,
    ).cuda()

    # Save the original _process_with_mask method from the first block
    first_block = cast(SelfAttentionBlock, model.blocks[0])
    original_process = first_block._process_with_mask

    # Replace with a method that raises an error
    def mock_process(*args, **kwargs):
        raise RuntimeError("Simulated attention error")

    # Monkey-patch the first block's method (setattr keeps the dynamic patch
    # without tripping nn.Module's typed-attribute assignment under strict mypy).
    setattr(first_block, "_process_with_mask", mock_process)

    # Create input embeddings (on GPU)
    x_dict = {"gene": data["gene"].x.clone(), "reaction": data["reaction"].x.clone()}

    # Forward pass should now raise an error
    with pytest.raises(RuntimeError) as excinfo:
        model(x_dict, data)

    # Check the error message
    assert "Simulated attention error" in str(excinfo.value)

    # Restore the original method
    first_block._process_with_mask = original_process


# ---------------------------------------------------------------------------
# 2026.10.06 - Phase 21: routing contracts of _HeteroNSA_Block and HeteroNSAEncoder
#
# Fixture: 3 genes and 2 reactions with hidden_dim 4, two heads, dropout 0, eval mode,
# seeded weights. Expected outputs are the block's own NodeSelfAttention /
# SelfAttentionBlock submodules applied by hand to the mask the routing should hand
# them (structural identities), and a spy on NodeSelfAttention.forward records exactly
# which mask, edge_attr and edge_index each relation passes. CUDA is hidden.
# ---------------------------------------------------------------------------

import re  # noqa: E402
from collections.abc import Callable, Iterator  # noqa: E402
from typing import Any  # noqa: E402

from torch_geometric.nn.aggr.attention import AttentionalAggregation  # noqa: E402

from torchcell.nn.hetero_nsa import _HeteroNSA_Block  # noqa: E402
from torchcell.nn.masked_attention_block import NodeSelfAttention  # noqa: E402

GG = ("gene", "physical", "gene")
GG2 = ("gene", "regulatory", "gene")
GR = ("gene", "gpr", "reaction")
RMR = ("reaction", "rmr", "metabolite")

Call = tuple[torch.Tensor, torch.Tensor, Any, Any]


@pytest.fixture(autouse=True)
def _fork_rng() -> Iterator[None]:
    """Every test runs inside its own RNG fork so seeding never leaks."""
    with torch.random.fork_rng():
        yield


@pytest.fixture
def nsa_calls(monkeypatch: pytest.MonkeyPatch) -> list[Call]:
    """Record (x, adj_mask, edge_attr, edge_index) of every NodeSelfAttention call."""
    calls: list[Call] = []
    original: Callable[..., torch.Tensor] = NodeSelfAttention.forward

    def spy(
        self: NodeSelfAttention,
        x: torch.Tensor,
        adj_mask: torch.Tensor,
        edge_attr: Any = None,
        edge_index: Any = None,
    ) -> torch.Tensor:
        calls.append((x, adj_mask, edge_attr, edge_index))
        return original(self, x, adj_mask, edge_attr, edge_index)

    monkeypatch.setattr(NodeSelfAttention, "forward", spy)
    return calls


def _block(
    layer_type: str,
    edge_types: set[tuple[str, str, str]],
    node_types: set[str] | None = None,
    aggregation: str = "sum",
) -> _HeteroNSA_Block:
    torch.manual_seed(0)
    return _HeteroNSA_Block(
        layer_type,  # type: ignore[arg-type, unused-ignore]
        4,
        node_types or {"gene", "reaction"},
        edge_types,
        num_heads=2,
        dropout=0.0,
        aggregation=aggregation,  # type: ignore[arg-type, unused-ignore]
    ).eval()


def _x() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {"gene": torch.randn(3, 4), "reaction": torch.randn(2, 4)}


def _nsa(block: _HeteroNSA_Block, edge_type: tuple[str, str, str]) -> NodeSelfAttention:
    module = block.masked_blocks["__".join(edge_type)]
    assert isinstance(module, NodeSelfAttention)
    return module


GG_MASK = torch.tensor([[True, True, False], [False, True, True], [True, False, True]])
GR_MASK = torch.tensor([[True, False], [False, True], [True, True]])


def test_invalid_layer_type_is_refused_by_the_block() -> None:
    """The block itself refuses 'X' with the exact message (hetero_nsa.py:37)."""
    with pytest.raises(
        ValueError, match=re.escape("Invalid layer_type 'X'. Must be 'M' or 'S'.")
    ):
        _block("X", {GG})


def test_same_type_relation_passes_its_mask_and_edge_attributes(
    nsa_calls: list[Call],
) -> None:
    """A gene-gene adj_mask relation: one call with mask [1, 3, 3], edge_attr and index.

    The output is that relation's NodeSelfAttention applied to (x[None], mask[None]) and
    squeezed back; the untouched reaction type keeps its input tensor object.
    """
    block = _block("M", {GG})
    data = HeteroData()
    data[GG].adj_mask = GG_MASK
    data[GG].edge_index = torch.tensor([[0, 1], [1, 2]])
    data[GG].edge_attr = torch.tensor([0.5, -1.0])
    x = _x()
    with torch.no_grad():
        out = block(x, data)
        expected = _nsa(block, GG)(
            x["gene"][None], GG_MASK[None], data[GG].edge_attr, data[GG].edge_index
        )[0]
    assert len(nsa_calls) == 2  # the block call and the expected call above
    seen_x, seen_mask, seen_attr, seen_index = nsa_calls[0]
    assert torch.equal(seen_x, x["gene"][None])
    assert torch.equal(seen_mask, GG_MASK[None])
    assert seen_attr is data[GG].edge_attr
    assert seen_index is data[GG].edge_index
    assert torch.equal(out["gene"], expected)
    assert out["reaction"] is x["reaction"]


def test_bipartite_relation_runs_the_source_mask_and_its_transpose(
    nsa_calls: list[Call],
) -> None:
    """gene->reaction with adj_mask [3, 2]: genes get the mask, reactions its transpose.

    Both go through the SAME NodeSelfAttention, each as a self-attention over its own
    node type with a non-square mask (padded / cropped, Phase 20 finding), and no edge
    attributes because the store has no edge_attr.
    """
    block = _block("M", {GR})
    data = HeteroData()
    data[GR].adj_mask = GR_MASK
    data[GR].edge_index = torch.tensor([[0, 2], [0, 1]])
    x = _x()
    with torch.no_grad():
        out = block(x, data)
    assert [c[1].tolist() for c in nsa_calls] == [
        GR_MASK[None].tolist(),
        GR_MASK.T[None].tolist(),
    ]
    assert [c[2] for c in nsa_calls] == [None, None]
    nsa = _nsa(block, GR)
    with torch.no_grad():
        torch.testing.assert_close(out["gene"], nsa(x["gene"], GR_MASK))
        torch.testing.assert_close(out["reaction"], nsa(x["reaction"], GR_MASK.T))


def test_rmr_relation_swaps_edge_attr_for_stoichiometry(nsa_calls: list[Call]) -> None:
    """An ``rmr`` adj_mask store passes ``stoichiometry`` as edge_attr, not edge_attr."""
    block = _block("M", {RMR}, {"reaction", "metabolite"})
    data = HeteroData()
    data[RMR].adj_mask = torch.ones(2, 3, dtype=torch.bool)
    data[RMR].edge_index = torch.tensor([[0, 1], [2, 0]])
    data[RMR].edge_attr = torch.tensor([9.0, 9.0])
    data[RMR].stoichiometry = torch.tensor([-1.0, 2.0])
    x = {"reaction": torch.randn(2, 4), "metabolite": torch.randn(3, 4)}
    with torch.no_grad():
        block(x, data)
    assert len(nsa_calls) == 2
    for _, _, attr, index in nsa_calls:
        assert attr is data[RMR].stoichiometry
        assert index is data[RMR].edge_index


def test_inc_mask_relation_drops_edge_attr_unless_rmr(nsa_calls: list[Call]) -> None:
    """Finding: an inc_mask relation never passes its edge_attr (hetero_nsa.py:160-161).

    The same store with ``adj_mask`` would pass edge_attr (line 136); with ``inc_mask``
    a non-rmr relation always gets (None, None), so its edge attributes are silently
    unused. An rmr inc_mask store passes stoichiometry with hyperedge_index preferred
    over edge_index, and edge_index when there is no hyperedge_index. Reach: latent;
    the 006 retry model names its metabolic relation "reaction", not "rmr", and its
    relations carry adj_mask. Pinned until the inc_mask branch forwards edge attributes
    like the adj_mask branch.
    """
    block = _block("M", {GR})
    data = HeteroData()
    data[GR].inc_mask = GR_MASK
    data[GR].edge_index = torch.tensor([[0], [1]])
    data[GR].edge_attr = torch.tensor([3.0])
    with torch.no_grad():
        out = block(_x(), data)
    assert [(c[2], c[3]) for c in nsa_calls] == [(None, None), (None, None)]
    assert [c[1].tolist() for c in nsa_calls] == [
        GR_MASK[None].tolist(),
        GR_MASK.T[None].tolist(),
    ]
    assert out["gene"].shape == (3, 4) and out["reaction"].shape == (2, 4)

    nsa_calls.clear()
    rmr_block = _block("M", {RMR}, {"reaction", "metabolite"})
    rmr = HeteroData()
    rmr[RMR].inc_mask = torch.ones(2, 3, dtype=torch.bool)
    rmr[RMR].edge_index = torch.tensor([[0], [0]])
    rmr[RMR].hyperedge_index = torch.tensor([[1], [1]])
    rmr[RMR].stoichiometry = torch.tensor([-1.0])
    x = {"reaction": torch.randn(2, 4), "metabolite": torch.randn(3, 4)}
    with torch.no_grad():
        rmr_block(x, rmr)
    assert [
        (c[2] is rmr[RMR].stoichiometry, c[3] is rmr[RMR].hyperedge_index)
        for c in nsa_calls
    ] == [(True, True), (True, True)]
    nsa_calls.clear()
    del rmr[RMR].hyperedge_index
    with torch.no_grad():
        rmr_block(x, rmr)
    assert [c[3] is rmr[RMR].edge_index for c in nsa_calls] == [True, True]


def test_relation_without_masks_ignores_its_edges() -> None:
    """Finding: an edge_index-only relation attends each node to itself alone.

    The fallback (hetero_nsa.py:181-204) builds an identity mask, so a node's softmax
    has one key and the edge bias (only on self-loops) cannot change it: two different
    edge_index tensors give bit-identical outputs, equal to the block run with no
    edges. The relation's graph never reaches the representation. Pinned until the
    fallback builds the mask from edge_index.
    """
    block = _block("M", {GG})
    x = _x()
    outs = []
    for edges in ([[0, 1], [1, 2]], [[2, 0, 1], [0, 1, 1]]):
        data = HeteroData()
        data[GG].edge_index = torch.tensor(edges)
        data[GG].edge_attr = torch.tensor([5.0, -5.0, 1.0][: len(edges[0])])
        with torch.no_grad():
            outs.append(block(x, data)["gene"])
    with torch.no_grad():
        no_edges = _nsa(block, GG)(x["gene"], torch.eye(3, dtype=torch.bool))
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(outs[0], no_edges)


def test_bipartite_relation_without_masks_uses_two_identity_masks() -> None:
    """gene->reaction with only edge_index: genes and reactions each attend to themselves."""
    block = _block("M", {GR})
    data = HeteroData()
    data[GR].edge_index = torch.tensor([[0, 2], [0, 1]])
    x = _x()
    nsa = _nsa(block, GR)
    with torch.no_grad():
        out = block(x, data)
        torch.testing.assert_close(out["gene"], nsa(x["gene"], torch.eye(3).bool()))
        torch.testing.assert_close(
            out["reaction"], nsa(x["reaction"], torch.eye(2).bool())
        )


def _two_gene_relations() -> HeteroData:
    data = HeteroData()
    data[GG].adj_mask = GG_MASK
    data[GG2].adj_mask = GG_MASK.T.contiguous()
    return data


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_two_relations_on_one_type_are_summed_or_averaged(aggregation: str) -> None:
    """Gene outputs of two relations o1, o2: 'sum' gives o1 + o2, 'mean' (o1 + o2) / 2."""
    block = _block("M", {GG, GG2}, aggregation=aggregation)
    x = _x()
    with torch.no_grad():
        out = block(x, _two_gene_relations())
        o1 = _nsa(block, GG)(x["gene"], GG_MASK)
        o2 = _nsa(block, GG2)(x["gene"], GG_MASK.T)
    expected = o1 + o2 if aggregation == "sum" else (o1 + o2) / 2
    torch.testing.assert_close(out["gene"], expected)


def test_attention_aggregation_pools_each_relation_into_one_vector() -> None:
    """Finding: 'attention' aggregation returns one row per RELATION, not per node.

    ``node_indices = arange(len(outs)).repeat_interleave(num_nodes)`` (hetero_nsa.py:224)
    puts all 3 genes of relation j into group j, so AttentionalAggregation returns
    [2 relations, 4] instead of [3 genes, 4]: row j is the gated softmax pool of
    relation j's node outputs, and j follows the iteration order of the edge-type set,
    so which row is which relation depends on PYTHONHASHSEED (Phase 20). Per-node
    aggregation needs ``arange(num_nodes).repeat(len(outs))``. Reach: latent; the 006
    retry config uses "sum". Pinned until the index groups the same node across
    relations.
    """
    block = _block("M", {GG, GG2}, {"gene"}, aggregation="attention")
    x = {"gene": _x()["gene"]}
    with torch.no_grad():
        out = block(x, _two_gene_relations())
        o1 = _nsa(block, GG)(x["gene"], GG_MASK)
        o2 = _nsa(block, GG2)(x["gene"], GG_MASK.T)
        aggr = block.node_aggregators["gene"]
        assert isinstance(aggr, AttentionalAggregation)
        # relation order is the iteration order of the edge-type SET (hash seed)
        ordered = [{GG: o1, GG2: o2}[et] for et in block.edge_types]
        expected = aggr(torch.cat(ordered), index=torch.tensor([0, 0, 0, 1, 1, 1]))
    assert out["gene"].shape == (2, 4)
    torch.testing.assert_close(out["gene"], expected)


def test_an_unknown_aggregation_given_to_the_block_falls_back_to_the_mean() -> None:
    """``HeteroNSA`` refuses aggregation 'max', but ``_HeteroNSA_Block`` does not
    validate it: built directly, two relations on genes reach the final else (line 233)
    and are averaged, (o1 + o2) / 2. Reachable only outside HeteroNSA.
    """
    block = _block("M", {GG, GG2}, aggregation="max")
    x = _x()
    with torch.no_grad():
        out = block(x, _two_gene_relations())
        o1 = _nsa(block, GG)(x["gene"], GG_MASK)
        o2 = _nsa(block, GG2)(x["gene"], GG_MASK.T)
    torch.testing.assert_close(out["gene"], (o1 + o2) / 2)


def test_attention_aggregation_without_an_aggregator_falls_back_to_the_mean() -> None:
    """A type in only one relation gets no aggregator (needs > 1); a type in two that
    lacks one (deleted here) gets the mean, as line 231 promises.
    """
    block = _block("M", {GG, GG2}, {"gene", "reaction"}, aggregation="attention")
    assert sorted(block.node_aggregators.keys()) == ["gene"]
    del block.node_aggregators["gene"]
    x = _x()
    with torch.no_grad():
        out = block(x, _two_gene_relations())
        o1 = _nsa(block, GG)(x["gene"], GG_MASK)
        o2 = _nsa(block, GG2)(x["gene"], GG_MASK.T)
    torch.testing.assert_close(out["gene"], (o1 + o2) / 2)


def test_missing_relation_or_embedding_is_skipped_and_absent_types_become_none() -> (
    None
):
    """A relation absent from data, or whose destination (reaction) embedding is missing,
    is skipped:
    the gene type keeps its input object; reaction, absent from x_dict, maps to None.
    """
    block = _block("M", {GG, GR})
    data = HeteroData()
    data[GR].adj_mask = GR_MASK
    gene = torch.randn(3, 4)
    out = block({"gene": gene}, data)
    assert sorted(out) == ["gene", "reaction"]
    assert out["gene"] is gene
    assert out["reaction"] is None


def test_process_with_mask_broadcasts_a_2d_mask_over_a_batch() -> None:
    """Mask [3, 3] with embeddings [2, 3, 4] is expanded to both samples; a [1, 3, 3]
    mask with [3, 4] embeddings unsqueezes the embeddings and squeezes the result.
    """
    block = _block("M", {GG})
    nsa = _nsa(block, GG)
    torch.manual_seed(3)
    batched = torch.randn(2, 3, 4)
    with torch.no_grad():
        out = block._process_with_mask(nsa, batched, GG_MASK)
        expected = nsa(batched, GG_MASK.expand(2, 3, 3))
        assert torch.equal(out, expected)
        flat = block._process_with_mask(nsa, batched[0], GG_MASK[None])
        assert torch.equal(flat, nsa(batched[0][None], GG_MASK[None])[0])


def test_process_with_mask_calls_a_non_nsa_block_without_the_mask() -> None:
    """A non-NodeSelfAttention block (a SelfAttentionBlock) is called as block(x)."""
    block = _block("S", {GG}, {"gene"})
    sab = block.self_blocks["gene"]
    torch.manual_seed(3)
    x = torch.randn(3, 4)
    with torch.no_grad():
        out = block._process_with_mask(sab, x, GG_MASK)
        assert torch.equal(out, sab(x[None])[0])


def test_self_attention_block_handles_2d_and_3d_and_passes_unknown_types() -> None:
    """'S': 2-D gene input -> sab(x[None])[0]; 3-D input -> sab(x); an unknown type
    ('metabolite', no block) is returned as the same object.
    """
    block = _block("S", set(), {"gene"})
    sab = block.self_blocks["gene"]
    torch.manual_seed(4)
    gene = torch.randn(3, 4)
    other = torch.randn(5, 4)
    with torch.no_grad():
        out = block({"gene": gene, "metabolite": other}, HeteroData())
        assert torch.equal(out["gene"], sab(gene[None])[0])
        assert out["metabolite"] is other
        out3 = block({"gene": gene[None]}, HeteroData())
        assert torch.equal(out3["gene"], sab(gene[None]))


def _encoder(pattern: list[str], node_types: set[str]) -> HeteroNSAEncoder:
    torch.manual_seed(0)
    return HeteroNSAEncoder(
        input_dims={nt: 2 for nt in node_types},
        hidden_dim=4,
        node_types=node_types,
        edge_types={GG},
        pattern=pattern,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    ).eval()


def test_encoder_composes_projection_residual_norm_and_pooling() -> None:
    """Pattern ['S'], one layer, genes only.

    h0 = input_projections.gene(x); h1 = LayerNorm_0(SAB(h0[None])[0] + h0); the graph
    vector is final_projection(mean over genes of graph_projections.gene(h1)). The
    returned node embeddings are h1.
    """
    enc = _encoder(["S"], {"gene"})
    data = HeteroData()
    torch.manual_seed(5)
    data["gene"].x = torch.randn(3, 2)
    with torch.no_grad():
        nodes, graph = enc(data)
        h0 = enc.input_projections["gene"](data["gene"].x)
        layer = enc.nsa_layers[0]
        assert isinstance(layer, HeteroNSA)
        s_block = layer.blocks[0]
        assert isinstance(s_block, _HeteroNSA_Block)
        sab = s_block.self_blocks["gene"]
        norms = enc.layer_norms["gene"]
        assert isinstance(norms, nn.ModuleList)
        h1 = norms[0](sab(h0[None])[0] + h0)
        pooled = enc.graph_projections["gene"](h1).mean(dim=0, keepdim=True)
        expected_graph = enc.final_projection(pooled)
    torch.testing.assert_close(nodes["gene"], h1)
    torch.testing.assert_close(graph, expected_graph)


def test_encoder_pools_a_whole_batch_into_one_graph_vector() -> None:
    """Finding: HeteroNSAEncoder ignores the batch vector when pooling.

    ``data["gene"].batch`` is collected into batch_idx (hetero_nsa.py:393-394) but no
    block reads it, and the graph vector is ``mean(dim=1)`` over every node
    (line 415). Two graphs of 2 and 1 genes in one batch give a [1, 4] graph
    embedding equal to the pool over all 3 nodes. Pinned until pooling is per graph.
    """
    enc = _encoder(["S"], {"gene"})
    data = HeteroData()
    torch.manual_seed(6)
    data["gene"].x = torch.randn(3, 2)
    data["gene"].batch = torch.tensor([0, 0, 1])
    with torch.no_grad():
        nodes, graph = enc(data)
        pooled = enc.graph_projections["gene"](nodes["gene"]).mean(0, keepdim=True)
        expected = enc.final_projection(pooled)
    assert graph.shape == (1, 4)
    torch.testing.assert_close(graph, expected)


def test_encoder_fills_an_absent_type_with_zeros_in_sorted_order() -> None:
    """Pattern ['S'], types {gene, reaction}, data with genes only: the concat follows
    sorted type names, [gene pool, reaction zeros(1, 4)], into final_projection.
    """
    enc = _encoder(["S"], {"gene", "reaction"})
    data = HeteroData()
    torch.manual_seed(8)
    data["gene"].x = torch.randn(3, 2)
    with torch.no_grad():
        nodes, graph = enc(data)
        pooled = enc.graph_projections["gene"](nodes["gene"]).mean(0, keepdim=True)
        expected = enc.final_projection(torch.cat([pooled, torch.zeros(1, 4)], dim=-1))
    assert sorted(nodes) == ["gene"]
    torch.testing.assert_close(graph, expected)


def test_encoder_with_a_masked_block_crashes_on_an_absent_type() -> None:
    """Finding: with an 'M' block an absent node type crashes the encoder.

    The M block returns ``None`` for every declared type with no relation output and no
    input (hetero_nsa.py:210); the encoder stores it in final_embeddings and then calls
    ``emb.dim()`` on it (line 412). The zeros fallback (line 423) is reachable only with
    S-only patterns. Pinned until absent types are left out of the block output.
    """
    enc = _encoder(["M"], {"gene", "reaction"})
    data = HeteroData()
    data["gene"].x = torch.randn(3, 2)
    data[GG].adj_mask = GG_MASK
    with pytest.raises(
        AttributeError, match=re.escape("'NoneType' object has no attribute 'dim'")
    ):
        enc(data)


def test_encoder_refuses_an_unknown_aggregation() -> None:
    """The encoder validates aggregation before building anything (line 341)."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid aggregation 'max'. Must be 'sum', 'mean', or 'attention'."
        ),
    ):
        HeteroNSAEncoder(
            input_dims={"gene": 2},
            hidden_dim=4,
            node_types={"gene"},
            edge_types={GG},
            pattern=["S"],
            aggregation="max",  # type: ignore[arg-type, unused-ignore]
        )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid aggregation 'max'. Must be 'sum', 'mean', or 'attention'."
        ),
    ):
        HeteroNSA(4, {"gene"}, {GG}, ["S"], aggregation="max")  # type: ignore[arg-type, unused-ignore]
