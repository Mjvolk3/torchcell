# tests/torchcell/nn/test_hetero_nsa.py

import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from torchcell.nn.hetero_nsa import HeteroNSA, NSAEncoder


@pytest.fixture
def test_hetero_graph():
    """Fixture to create a basic heterogeneous graph for testing."""
    num_nodes_dict = {"gene": 100, "reaction": 50, "metabolite": 30}
    num_edges_dict = {
        ("gene", "physical_interaction", "gene"): 200,
        ("gene", "regulatory_interaction", "gene"): 150,
        ("gene", "gpr", "reaction"): 80,  # Uncomment this line
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
    """Fixture to provide a standard configuration for the NSAEncoder."""
    node_types = {"gene", "reaction", "metabolite"}
    edge_types = {
        ("gene", "physical_interaction", "gene"),
        ("gene", "regulatory_interaction", "gene"),
        ("gene", "gpr", "reaction"),
        ("reaction", "rmr", "metabolite"),
    }

    patterns = {
        ("gene", "physical_interaction", "gene"): ["M", "S"],
        ("gene", "regulatory_interaction", "gene"): ["M", "S"],
        ("gene", "gpr", "reaction"): ["M"],
        ("reaction", "rmr", "metabolite"): ["M"],
        # Added by ToUndirected
        ("reaction", "rmr_rev", "gene"): ["M"],
        ("metabolite", "rmr_rev", "reaction"): ["M"],
    }

    input_dims = {"gene": 32, "reaction": 32, "metabolite": 32}

    return {
        "input_dims": input_dims,
        "hidden_dim": 64,
        "node_types": node_types,
        "edge_types": edge_types,
        "patterns": patterns,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "aggregation": "attention",
    }


def test_hetero_nsa_initialization(standard_encoder_config):
    """Test that HeteroNSA initializes correctly."""
    config = standard_encoder_config

    # Initialize just the HeteroNSA module
    model = HeteroNSA(
        hidden_dim=config["hidden_dim"],
        node_types=config["node_types"],
        edge_types=config["edge_types"],
        patterns=config["patterns"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        aggregation=config["aggregation"],
    )

    # Check that attention blocks were created for each edge type
    for edge_type in config["edge_types"]:
        src, rel, dst = edge_type
        key = f"{src}__{rel}__{dst}"
        assert key in model.attention_blocks

        # Check pattern lengths
        pattern = config["patterns"].get(edge_type, [])
        assert len(model.attention_blocks[key]) == len(pattern)


def test_nsa_encoder_forward(test_hetero_graph, standard_encoder_config):
    """Test the forward pass of NSAEncoder with a heterogeneous graph."""
    # Get test data
    data = test_hetero_graph
    config = standard_encoder_config

    # Create encoder
    encoder = NSAEncoder(**config)

    # Forward pass
    node_embeddings, graph_embedding = encoder(data)

    # Check output shapes
    assert "gene" in node_embeddings
    assert "reaction" in node_embeddings
    assert "metabolite" in node_embeddings

    assert node_embeddings["gene"].size(-1) == config["hidden_dim"]
    assert node_embeddings["reaction"].size(-1) == config["hidden_dim"]
    assert node_embeddings["metabolite"].size(-1) == config["hidden_dim"]

    assert graph_embedding.size() == (1, config["hidden_dim"])


def test_nsa_aggregation_methods(test_hetero_graph, standard_encoder_config):
    """Test different aggregation methods in NSAEncoder."""
    data = test_hetero_graph
    config = standard_encoder_config

    # Test each aggregation method
    for aggregation in ["sum", "mean", "max", "attention"]:
        config["aggregation"] = aggregation
        encoder = NSAEncoder(**config)

        # Forward pass should work with all aggregation methods
        node_embeddings, graph_embedding = encoder(data)

        # All should produce embeddings of the same shape
        assert node_embeddings["gene"].size(-1) == config["hidden_dim"]
        assert graph_embedding.size() == (1, config["hidden_dim"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hetero_nsa_cuda(test_hetero_graph, standard_encoder_config):
    """Test that HeteroNSA works on CUDA."""
    # Skip if no CUDA
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    data = test_hetero_graph
    config = standard_encoder_config

    # Move data to CUDA
    device = torch.device("cuda")
    for node_type in data.node_types:
        data[node_type].x = data[node_type].x.to(device)
    for edge_type in data.edge_types:
        data[edge_type].edge_index = data[edge_type].edge_index.to(device)
        if hasattr(data[edge_type], "edge_type"):
            data[edge_type].edge_type = data[edge_type].edge_type.to(device)
        if hasattr(data[edge_type], "stoichiometry"):
            data[edge_type].stoichiometry = data[edge_type].stoichiometry.to(device)

    # Create encoder and move to CUDA
    encoder = NSAEncoder(**config).to(device)

    # Forward pass
    node_embeddings, graph_embedding = encoder(data)

    # Check device
    assert node_embeddings["gene"].device.type == "cuda"
    assert graph_embedding.device.type == "cuda"


def test_invalid_patterns():
    """Test that invalid patterns raise appropriate errors."""
    node_types = {"gene", "reaction"}
    edge_types = {("gene", "interaction", "gene")}

    # Invalid block type
    invalid_patterns = {("gene", "interaction", "gene"): ["M", "X", "S"]}

    with pytest.raises(ValueError, match="Invalid block type"):
        HeteroNSA(
            hidden_dim=64,
            node_types=node_types,
            edge_types=edge_types,
            patterns=invalid_patterns,
        )

    # Missing pattern for an edge type
    with pytest.raises(ValueError, match="does not have a pattern defined"):
        HeteroNSA(
            hidden_dim=64,
            node_types=node_types,
            edge_types=edge_types,
            patterns={},  # Empty patterns dict
        )
