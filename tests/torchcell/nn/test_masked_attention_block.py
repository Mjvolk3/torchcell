# tests/torchcell/nn/test_masked_attention_block.py

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import StochasticBlockModelDataset

from torchcell.nn.masked_attention_block import (
    EdgeSetAttention,
    ESAEncoder,
    MaskedAttentionBlock,
    NodeSetAttention,
    NSAEncoder,
)
from torchcell.nn.self_attention_block import SelfAttentionBlock


def create_test_graph(num_nodes=100, num_features=16, device="cpu"):
    """Create a test graph for attention tests."""
    # Create a simple graph with community structure
    block_sizes = [25, 25, 25, 25]  # 4 blocks of 25 nodes each
    edge_probs = [
        [0.7, 0.05, 0.05, 0.05],
        [0.05, 0.7, 0.05, 0.05],
        [0.05, 0.05, 0.7, 0.05],
        [0.05, 0.05, 0.05, 0.7],
    ]  # Higher intra-cluster connection probability

    dataset = StochasticBlockModelDataset(
        root="/tmp/sbm", block_sizes=block_sizes, edge_probs=edge_probs
    )
    data = dataset[0]

    # Add node features if not present
    if not hasattr(data, "x") or data.x is None:
        data.x = torch.randn(data.num_nodes, num_features)

    # Add edge features
    num_edges = data.edge_index.size(1)
    data.edge_attr = torch.randn(num_edges, num_features)

    # Move to device
    return data.to(device)


def test_masked_attention_block_initialization():
    """Test that the MAB initializes with valid parameters."""
    hidden_dim = 64
    num_heads = 8

    # Test node mode
    node_mab = MaskedAttentionBlock(
        hidden_dim=hidden_dim, num_heads=num_heads, mode="node"
    )
    assert node_mab.hidden_dim == hidden_dim
    assert node_mab.num_heads == num_heads
    assert node_mab.head_dim == hidden_dim // num_heads
    assert node_mab.mode == "node"

    # Test edge mode
    edge_mab = MaskedAttentionBlock(
        hidden_dim=hidden_dim, num_heads=num_heads, mode="edge"
    )
    assert edge_mab.hidden_dim == hidden_dim
    assert edge_mab.num_heads == num_heads
    assert edge_mab.head_dim == hidden_dim // num_heads
    assert edge_mab.mode == "edge"

    # Test convenience classes
    nsa = NodeSetAttention(hidden_dim=hidden_dim, num_heads=num_heads)
    assert nsa.mode == "node"

    esa = EdgeSetAttention(hidden_dim=hidden_dim, num_heads=num_heads)
    assert esa.mode == "edge"


def test_node_set_attention_forward():
    """Test forward pass of NodeSetAttention."""
    hidden_dim = 64
    batch_size = 2
    num_nodes = 100

    # Create model and input tensor
    nsa = NodeSetAttention(hidden_dim=hidden_dim)
    x = torch.randn(batch_size, num_nodes, hidden_dim)

    # Create a simple adjacency matrix (fully connected within each graph)
    adj_matrix = torch.ones(batch_size, num_nodes, num_nodes, dtype=torch.bool)

    # Forward pass
    output = nsa(x, adj_matrix)

    # Check output shape
    assert output.shape == (batch_size, num_nodes, hidden_dim)


def test_edge_set_attention_forward():
    """Test forward pass of EdgeSetAttention."""
    hidden_dim = 64
    batch_size = 2
    num_edges = 50

    # Create model and input tensor
    esa = EdgeSetAttention(hidden_dim=hidden_dim)
    x = torch.randn(batch_size, num_edges, hidden_dim)

    # Create a simple edge-to-edge adjacency matrix
    edge_adj = torch.ones(batch_size, num_edges, num_edges, dtype=torch.bool)

    # Forward pass
    output = esa(x, edge_adj)

    # Check output shape
    assert output.shape == (batch_size, num_edges, hidden_dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nsa_encoder_with_graph():
    """Test NSAEncoder with a PyG graph."""
    # Create a test graph
    data = create_test_graph(device="cuda")

    # Setup model parameters
    input_dim = data.x.size(1)
    hidden_dim = 64
    num_heads = 4
    pattern = ["M", "S", "M", "S"]

    # Create encoder
    encoder = NSAEncoder(
        input_dim=input_dim, hidden_dim=hidden_dim, pattern=pattern, num_heads=num_heads
    ).cuda()

    # Forward pass
    node_embeddings = encoder(data.x, data.edge_index)

    # Check output shape
    assert node_embeddings.shape == (data.num_nodes, hidden_dim)

    # Check that output contains no NaN values
    assert not torch.isnan(node_embeddings).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_esa_encoder_with_graph():
    """Test ESAEncoder with a PyG graph."""
    # Create a test graph
    data = create_test_graph(device="cuda")

    # Setup model parameters
    input_dim = data.edge_attr.size(1)
    hidden_dim = 64
    num_heads = 4
    pattern = ["M", "S", "M", "S"]

    # Create encoder
    encoder = ESAEncoder(
        input_dim=input_dim, hidden_dim=hidden_dim, pattern=pattern, num_heads=num_heads
    ).cuda()

    # Forward pass
    edge_embeddings = encoder(data.edge_attr, data.edge_index)

    # Check output shape
    assert edge_embeddings.shape == (data.edge_index.size(1), hidden_dim)

    # Check that output contains no NaN values
    assert not torch.isnan(edge_embeddings).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nsa_encoder_with_batched_graphs():
    """Test NSAEncoder with batched PyG graphs."""
    # Create multiple test graphs
    data_list = [
        create_test_graph(num_nodes=50, device="cuda"),
        create_test_graph(num_nodes=80, device="cuda"),
        create_test_graph(num_nodes=60, device="cuda"),
    ]

    # Batch the graphs
    batch = Batch.from_data_list(data_list)

    # Setup model parameters
    input_dim = batch.x.size(1)
    hidden_dim = 64
    num_heads = 4
    pattern = ["M", "S", "M"]

    # Create encoder
    encoder = NSAEncoder(
        input_dim=input_dim, hidden_dim=hidden_dim, pattern=pattern, num_heads=num_heads
    ).cuda()

    # Forward pass
    node_embeddings = encoder(batch.x, batch.edge_index, batch.batch)

    # Check output shape
    assert node_embeddings.shape == (batch.num_nodes, hidden_dim)

    # Check that output contains no NaN values
    assert not torch.isnan(node_embeddings).any()


def test_pattern_validation():
    """Test that pattern validation works correctly."""
    input_dim = 16
    hidden_dim = 64

    # Valid patterns
    valid_patterns = [
        ["M", "S", "M", "S"],
        ["M", "M", "S"],
        ["S", "M", "S"],
        ["M"],
        ["S"],
    ]

    for pattern in valid_patterns:
        # These should not raise exceptions
        NSAEncoder(input_dim=input_dim, hidden_dim=hidden_dim, pattern=pattern)
        ESAEncoder(input_dim=input_dim, hidden_dim=hidden_dim, pattern=pattern)

    # Invalid patterns
    invalid_patterns = [["M", "X", "S"], ["Y"], ["m", "s"]]  # Case sensitive

    for pattern in invalid_patterns:
        with pytest.raises(ValueError):
            NSAEncoder(input_dim=input_dim, hidden_dim=hidden_dim, pattern=pattern)

        with pytest.raises(ValueError):
            ESAEncoder(input_dim=input_dim, hidden_dim=hidden_dim, pattern=pattern)


def test_differentiability():
    """Test that the attention blocks are differentiable and gradients flow properly."""
    hidden_dim = 64
    batch_size = 2
    seq_len = 20

    # Create models
    nsa = NodeSetAttention(hidden_dim=hidden_dim)
    esa = EdgeSetAttention(hidden_dim=hidden_dim)

    # Create input tensors
    x_node = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    x_edge = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

    # Create adjacency matrices
    node_adj = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
    edge_adj = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

    # Forward pass for NSA
    output_node = nsa(x_node, node_adj)
    loss_node = output_node.mean()

    # Backward pass for NSA
    loss_node.backward()

    # Check that gradients were computed for NSA
    assert x_node.grad is not None
    assert not torch.isnan(x_node.grad).any()

    # Forward pass for ESA
    output_edge = esa(x_edge, edge_adj)
    loss_edge = output_edge.mean()

    # Backward pass for ESA
    loss_edge.backward()

    # Check that gradients were computed for ESA
    assert x_edge.grad is not None
    assert not torch.isnan(x_edge.grad).any()


def test_nsa_with_metabolic_stoichiometry():
    """Test NSAEncoder with a metabolic network structure including stoichiometry."""
    # Create a small metabolic network with 6 nodes (3 metabolites, 3 reactions)
    # Metabolites: A, B, C
    # Reactions: R1 (A -> B), R2 (B -> C), R3 (C -> A)

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

    # Stoichiometric coefficients
    # -1 for consumption, +1 for production
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

    # Create an NSAEncoder with a simple pattern
    encoder = NSAEncoder(input_dim=2, hidden_dim=16, pattern=["M", "S"], num_heads=4)

    # Forward pass with stoichiometry
    node_embeddings = encoder(x, edge_index, edge_attr)

    # Verify output shape
    assert node_embeddings.shape == (6, 16)

    # Compare with output without stoichiometry
    node_embeddings_no_stoich = encoder(x, edge_index)

    # Outputs should be different when stoichiometry is considered
    assert not torch.allclose(
        node_embeddings, node_embeddings_no_stoich, rtol=1e-3, atol=1e-3
    )

    # Also test with a batch of identical graphs
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    x_batched = torch.cat([x, x], dim=0)

    # Create batched edge_index by offsetting the second graph's indices
    edge_index_batched = torch.cat(
        [
            edge_index,
            edge_index + torch.tensor([[6], [6]]),  # Add 6 to indices for second graph
        ],
        dim=1,
    )

    # Repeat edge attributes for the second graph
    edge_attr_batched = torch.cat([edge_attr, edge_attr], dim=0)

    # Forward pass with batched input
    node_embeddings_batched = encoder(
        x_batched, edge_index_batched, edge_attr_batched, batch
    )

    # Verify output shape
    assert node_embeddings_batched.shape == (12, 16)
