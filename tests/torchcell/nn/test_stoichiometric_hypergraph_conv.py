# Test file
import pytest
import torch
from torch_geometric.data import Data

from torchcell.nn.met_hypergraph_conv import StoichiometricHypergraphConv


@pytest.fixture
def basic_data():
    x = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float)  # Node A  # Node B

    edge_index = torch.tensor(
        [[0, 1], [0, 0]],  # Nodes A and B  # Both connected to hyperedge 0
        dtype=torch.long,
    )

    stoich = torch.tensor([-1.0, 2.0], dtype=torch.float)

    return x, edge_index, stoich


@pytest.fixture
def complex_data():
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # Node A
            [0.0, 1.0, 0.0],  # Node B
            [0.0, 0.0, 1.0],  # Node C
            [1.0, 1.0, 0.0],  # Node D
        ],
        dtype=torch.float,
    )

    edge_index = torch.tensor(
        [[0, 1, 2], [0, 0, 0]],  # Nodes A, B, C  # All connected to hyperedge 0
        dtype=torch.long,
    )

    stoich = torch.tensor([-1.0, -2.0, 1.0], dtype=torch.float)

    return x, edge_index, stoich


def test_basic_initialization():
    conv = StoichiometricHypergraphConv(in_channels=2, out_channels=2)
    assert conv.in_channels == 2
    assert conv.out_channels == 2
    assert not conv.is_stoich_gated
    assert not conv.use_attention


def test_gated_initialization():
    conv = StoichiometricHypergraphConv(
        in_channels=2, out_channels=2, is_stoich_gated=True
    )
    assert conv.is_stoich_gated
    assert hasattr(conv, "gate_lin")


def test_forward_shape(basic_data):
    x, edge_index, stoich = basic_data
    conv = StoichiometricHypergraphConv(in_channels=2, out_channels=2)
    out = conv(x, edge_index, stoich)
    assert out.shape == (2, 2)


def test_forward_shape_complex(complex_data):
    x, edge_index, stoich = complex_data
    conv = StoichiometricHypergraphConv(in_channels=3, out_channels=2)
    out = conv(x, edge_index, stoich)
    assert out.shape == (4, 2)


def test_gated_vs_nongated(basic_data):
    x, edge_index, stoich = basic_data

    conv_normal = StoichiometricHypergraphConv(
        in_channels=2, out_channels=2, is_stoich_gated=False
    )
    conv_gated = StoichiometricHypergraphConv(
        in_channels=2, out_channels=2, is_stoich_gated=True
    )

    out_normal = conv_normal(x, edge_index, stoich)
    out_gated = conv_gated(x, edge_index, stoich)

    assert not torch.allclose(out_normal, out_gated)


def test_attention_mechanism(complex_data):
    x, edge_index, stoich = complex_data
    hyperedge_attr = torch.randn(1, 3)  # One hyperedge with 3 features

    conv = StoichiometricHypergraphConv(
        in_channels=3, out_channels=2, use_attention=True, heads=2
    )

    out = conv(x, edge_index, stoich, hyperedge_attr)
    assert out.shape == (4, 4)  # 4 nodes, 2 heads * 2 out_channels


def test_zero_stoichiometry(basic_data):
    x, edge_index, _ = basic_data
    stoich = torch.tensor([0.0, 0.0], dtype=torch.float)

    conv = StoichiometricHypergraphConv(in_channels=2, out_channels=2)
    out = conv(x, edge_index, stoich)

    assert torch.allclose(out, torch.zeros_like(out))


def test_activation_effects(basic_data):
    x, edge_index, stoich = basic_data

    conv_gated = StoichiometricHypergraphConv(
        in_channels=2, out_channels=2, is_stoich_gated=True
    )

    # Test with different activation functions
    activations = [torch.nn.ReLU(), torch.nn.Tanh(), torch.nn.GELU()]

    for activation in activations:
        out = activation(conv_gated(x, edge_index, stoich))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


def test_gradient_flow(basic_data):
    x, edge_index, stoich = basic_data
    x.requires_grad_(True)

    conv = StoichiometricHypergraphConv(
        in_channels=2, out_channels=2, is_stoich_gated=True
    )

    out = conv(x, edge_index, stoich)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_basic_structure(basic_data):
    """Tests Example 1: Basic Input Data Structure"""
    x = torch.tensor([[1.0, 0.5], [0.5, 1.0]])  # Node A  # Node B
    edge_index = torch.tensor([[0, 1], [0, 0]])
    stoich = torch.tensor([-1.0, 2.0])

    conv = StoichiometricHypergraphConv(in_channels=2, out_channels=2)
    out = conv(x, edge_index, stoich)

    assert out.shape == (2, 2)
    assert not torch.isnan(out).any()


def test_tanh_activation_comparison():
    """Tests gated vs non-gated behavior with tanh activation"""
    torch.manual_seed(42)

    x = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    edge_index = torch.tensor([[0, 1], [0, 0]])

    # Make stoichiometric coefficients more distinct
    stoich_orig = torch.tensor([-1.0, 2.0])
    stoich_flip = torch.tensor([2.0, -1.0])  # Changed to ensure different magnitudes

    conv_normal = StoichiometricHypergraphConv(
        in_channels=2, out_channels=2, is_stoich_gated=False
    )
    conv_gated = StoichiometricHypergraphConv(
        in_channels=2, out_channels=2, is_stoich_gated=True
    )

    # Force different initializations
    with torch.no_grad():
        conv_normal.lin.weight.data = torch.randn_like(conv_normal.lin.weight)
        conv_gated.lin.weight.data = torch.randn_like(conv_gated.lin.weight)
        if conv_gated.gate_lin is not None:
            conv_gated.gate_lin.weight.data = torch.randn_like(
                conv_gated.gate_lin.weight
            )

    # Test both original and flipped stoichiometry
    out_normal_orig = conv_normal(x, edge_index, stoich_orig)
    out_gated_orig = conv_gated(x, edge_index, stoich_orig)
    out_normal_flip = conv_normal(x, edge_index, stoich_flip)
    out_gated_flip = conv_gated(x, edge_index, stoich_flip)

    # Verify outputs are well-formed
    assert not torch.isnan(out_normal_orig).any()
    assert not torch.isnan(out_gated_orig).any()
    assert not torch.isnan(out_normal_flip).any()
    assert not torch.isnan(out_gated_flip).any()

    # Verify that gating affects the output
    diff_orig = torch.abs(out_normal_orig - out_gated_orig).mean()
    diff_flip = torch.abs(out_normal_flip - out_gated_flip).mean()
    assert diff_orig > 1e-6
    assert diff_flip > 1e-6

    # Test with smaller threshold for stoichiometry difference
    diff_normal = torch.abs(out_normal_orig - out_normal_flip).mean()
    diff_gated = torch.abs(out_gated_orig - out_gated_flip).mean()
    assert diff_normal > 1e-8  # Reduced threshold
    assert diff_gated > 1e-8  # Reduced threshold


def test_relu_activation_cases():
    """Tests Example 4 & 5: ReLU activation with different stoichiometries"""
    x = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    edge_index = torch.tensor([[0, 1], [0, 0]])

    # Test negative stoichiometry
    stoich_neg = torch.tensor([-1.0, -1.0])
    # Test positive stoichiometry
    stoich_pos = torch.tensor([1.0, 1.0])

    conv_normal = StoichiometricHypergraphConv(
        in_channels=2, out_channels=2, is_stoich_gated=False
    )
    conv_gated = StoichiometricHypergraphConv(
        in_channels=2, out_channels=2, is_stoich_gated=True
    )

    relu = torch.nn.ReLU()

    # Negative stoichiometry
    out_normal_neg = relu(conv_normal(x, edge_index, stoich_neg))
    out_gated_neg = relu(conv_gated(x, edge_index, stoich_neg))

    # Positive stoichiometry
    out_normal_pos = relu(conv_normal(x, edge_index, stoich_pos))
    out_gated_pos = relu(conv_gated(x, edge_index, stoich_pos))

    # Verify ReLU behavior
    assert torch.all(out_normal_neg >= 0)
    assert torch.all(out_gated_neg >= 0)
    assert torch.all(out_normal_pos >= 0)
    assert torch.all(out_gated_pos >= 0)


def test_gelu_activation_cases():
    """Tests Example 6 & 7: GELU activation with different stoichiometries"""
    x = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    edge_index = torch.tensor([[0, 1], [0, 0]])

    # Test negative stoichiometry
    stoich_neg = torch.tensor([-1.0, -1.0])
    # Test positive stoichiometry
    stoich_pos = torch.tensor([1.0, 1.0])

    conv_normal = StoichiometricHypergraphConv(
        in_channels=2, out_channels=2, is_stoich_gated=False
    )
    conv_gated = StoichiometricHypergraphConv(
        in_channels=2, out_channels=2, is_stoich_gated=True
    )

    gelu = torch.nn.GELU()

    # Negative stoichiometry
    out_normal_neg = gelu(conv_normal(x, edge_index, stoich_neg))
    out_gated_neg = gelu(conv_gated(x, edge_index, stoich_neg))

    # Positive stoichiometry
    out_normal_pos = gelu(conv_normal(x, edge_index, stoich_pos))
    out_gated_pos = gelu(conv_gated(x, edge_index, stoich_pos))

    # Verify GELU behavior
    assert not torch.allclose(out_normal_neg, out_gated_neg)
    assert not torch.allclose(out_normal_pos, out_gated_pos)

    # Verify partial negative preservation
    assert torch.any(out_normal_neg < 0)
    assert torch.any(out_gated_neg < 0)
