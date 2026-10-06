# tests/torchcell/nn/test_nsa_encoder.py
"""Tests for the NSAEncoder graph encoder module."""

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import StochasticBlockModelDataset
from torch_geometric.utils import to_dense_adj

from torchcell.nn.nsa_encoder import NSAEncoder


@pytest.fixture
def simple_graph():
    """Fixture for a simple graph with community structure."""
    # Create a simple graph with 4 communities
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

    # Add node features
    num_features = 16
    data.x = torch.randn(data.num_nodes, num_features)

    # Add edge features
    num_edges = data.edge_index.size(1)
    data.edge_attr = torch.randn(num_edges, num_features)

    # Explicitly set num_nodes to avoid PyG warnings
    data.num_nodes = data.x.size(0)

    return data


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


def test_nsa_encoder_with_graph(simple_graph):
    """Test NSAEncoder on a simple graph without using FlexAttention."""
    # To avoid FlexAttention issues in testing
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    # Create encoder
    input_dim = simple_graph.x.size(1)
    hidden_dim = 16

    # Use a simpler pattern with only SAB blocks to avoid FlexAttention
    encoder = NSAEncoder(input_dim=input_dim, hidden_dim=hidden_dim, pattern=["S"])

    # Create adjacency matrix
    adj = to_dense_adj(simple_graph.edge_index)[0].bool()

    # Add a test property to the graph to pass through instead of direct edge_index
    simple_graph.adj = adj

    # Forward pass
    with torch.no_grad():
        node_embeddings = encoder(simple_graph.x, simple_graph)

    # Check output shape
    assert node_embeddings.shape == (simple_graph.num_nodes, hidden_dim)
    assert not torch.isnan(node_embeddings).any()


def test_nsa_encoder_with_metabolic_graph(metabolic_graph):
    """Test NSAEncoder on a metabolic network using manual calculations."""
    # To avoid FlexAttention issues in testing
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    # Setup encoder with simplified pattern
    input_dim = metabolic_graph.x.size(1)
    hidden_dim = 16
    encoder = NSAEncoder(input_dim=input_dim, hidden_dim=hidden_dim, pattern=["S"])

    # Create and attach adjacency matrix
    adj = to_dense_adj(metabolic_graph.edge_index)[0].bool()
    metabolic_graph.adj = adj

    # Forward pass with torch.no_grad to avoid FlexAttention compilation
    with torch.no_grad():
        node_embeddings = encoder(metabolic_graph.x, metabolic_graph)

    # Verify output shape
    assert node_embeddings.shape == (metabolic_graph.num_nodes, hidden_dim)
    assert not torch.isnan(node_embeddings).any()

    # Verify it's actually doing something
    assert torch.norm(node_embeddings) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nsa_encoder_with_cuda(simple_graph):
    """Test NSAEncoder on a CUDA device."""
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    # Move data to CUDA
    cuda_graph = simple_graph.to("cuda")

    # Create and attach adjacency matrix
    adj = to_dense_adj(cuda_graph.edge_index)[0].bool()
    cuda_graph.adj = adj

    # Setup encoder with simplified pattern
    input_dim = cuda_graph.x.size(1)
    hidden_dim = 32
    encoder = NSAEncoder(
        input_dim=input_dim, hidden_dim=hidden_dim, pattern=["S"]
    ).cuda()

    # Forward pass
    with torch.no_grad():
        node_embeddings = encoder(cuda_graph.x, cuda_graph)

    # Check output properties
    assert node_embeddings.device.type == "cuda"
    assert node_embeddings.shape == (cuda_graph.num_nodes, hidden_dim)
    assert not torch.isnan(node_embeddings).any()


def test_nsa_encoder_with_batched_graphs(simple_graph):
    """Test NSAEncoder with a manually created batch instead of a fixture."""
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    # Create a simpler batch directly to avoid fixture issues
    device = simple_graph.x.device

    # Create a small graph
    graph1 = Data(
        x=torch.randn(10, simple_graph.x.size(1), device=device),
        edge_index=torch.randint(0, 10, (2, 20), device=device),
        num_nodes=10,
    )

    # Create another small graph
    graph2 = Data(
        x=torch.randn(15, simple_graph.x.size(1), device=device),
        edge_index=torch.randint(0, 15, (2, 30), device=device),
        num_nodes=15,
    )

    # Batch them
    batch = Batch.from_data_list([graph1, graph2])

    # Setup encoder with simplified pattern
    input_dim = batch.x.size(1)
    hidden_dim = 32
    encoder = NSAEncoder(input_dim=input_dim, hidden_dim=hidden_dim, pattern=["S"]).to(
        batch.x.device
    )

    # Forward pass with batch information
    with torch.no_grad():
        node_embeddings = encoder(
            batch.x,
            batch.edge_index,
            None,
            batch.batch,  # No edge attributes
        )

    # Check output shape
    assert node_embeddings.shape == (batch.num_nodes, hidden_dim)
    assert not torch.isnan(node_embeddings).any()


# ---------------------------------------------------------------------------
# 2026.10.06 - Phase 21: adjacency sources, padding and edge attributes (CPU path)
#
# Fixture: a 3-node directed cycle 0->1->2->0 (and a 4-node 2-cycle pair for
# batching), input_dim 2, hidden_dim 4, two heads, dropout 0, eval mode, seeded
# weights. Random weights carry no float contract, so the tests pin structural
# identities: equal outputs across equivalent inputs, the exact layer list, and
# dependence (or not) on a batch companion. CUDA is hidden.
# ---------------------------------------------------------------------------

import re  # noqa: E402
import types  # noqa: E402
from collections.abc import Iterator  # noqa: E402

from torchcell.nn.masked_attention_block import NodeSelfAttention  # noqa: E402
from torchcell.nn.self_attention_block import SelfAttentionBlock  # noqa: E402

CYCLE = torch.tensor([[0, 1, 2], [1, 2, 0]])


@pytest.fixture(autouse=True)
def _fork_rng() -> Iterator[None]:
    """Every test runs inside its own RNG fork so seeding never leaks."""
    with torch.random.fork_rng():
        yield


def _enc(pattern: list[str] | None) -> NSAEncoder:
    torch.manual_seed(0)
    return NSAEncoder(2, 4, pattern=pattern, num_heads=2, dropout=0.0).eval()  # type: ignore[arg-type, unused-ignore]


def _x3() -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randn(3, 2)


def test_default_pattern_is_m_s_m_s_and_invalid_types_are_refused() -> None:
    """pattern=None builds [NSA, SAB, NSA, SAB]; 'Q' raises the exact message."""
    enc = _enc(None)
    assert [type(m) for m in enc.layers] == [
        NodeSelfAttention,
        SelfAttentionBlock,
        NodeSelfAttention,
        SelfAttentionBlock,
    ]
    with pytest.raises(ValueError, match=re.escape("Invalid block type 'Q'.")):
        NSAEncoder(2, 4, pattern=["M", "Q"], num_heads=2)  # type: ignore[list-item, unused-ignore]


def test_every_adjacency_source_gives_the_same_output() -> None:
    """edge_index tensor, Data(edge_index), Data.adj_mask [1,3,3] and Data.adj [1,3,3]
    all reach the M block as the same dense mask, so the outputs are bit-identical.
    """
    enc = _enc(["M", "S"])
    x = _x3()
    dense = to_dense_adj(CYCLE)
    with torch.no_grad():
        ref = enc(x, CYCLE)
        with_mask = Data(x=x, num_nodes=3)
        with_mask.adj_mask = dense.bool()
        with_adj = Data(x=x, num_nodes=3)
        with_adj.adj = dense
        for data in (Data(x=x, edge_index=CYCLE, num_nodes=3), with_mask, with_adj):
            assert torch.equal(enc(x, data), ref)
    assert ref.shape == (3, 4)


def test_data_without_adjacency_is_refused() -> None:
    """An object with no adj_mask / adj / edge_index raises the exact ValueError."""
    with pytest.raises(
        ValueError,
        match=re.escape("Cannot extract adjacency information from provided data."),
    ):
        _enc(["M"])(_x3(), types.SimpleNamespace())  # type: ignore[arg-type, unused-ignore]


def test_a_two_dimensional_adj_crashes_masked_blocks() -> None:
    """Finding: a 2-D ``data.adj`` [N, N] only works for S-only patterns.

    The existing tests above pass a 2-D adj with pattern ['S'], which never reads the
    mask. With an 'M' block, NodeSelfAttention receives h [1, 3, 4] and the 2-D mask,
    and expanding it over heads fails (masked_attention_block.py:496). Pinned until the
    encoder adds the batch axis to a 2-D adj.
    """
    data = Data(x=_x3(), num_nodes=3)
    data.adj = to_dense_adj(CYCLE)[0]
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "The expanded size of the tensor (2) must match the existing size (3) at "
            "non-singleton dimension 1.  Target sizes: [-1, 2, -1, -1].  "
            "Tensor sizes: [3, 1, 3]"
        ),
    ):
        _enc(["M"])(data.x, data)
    with torch.no_grad():
        assert _enc(["S"])(data.x, data).shape == (3, 4)


def test_edge_attributes_never_change_the_output() -> None:
    """Finding: NSAEncoder computes an edge-attribute dict and then loses it.

    It passes ``layer(h, adj_mask, edge_attr_dict)`` without edge_index
    (nsa_encoder.py:140), and NodeSelfAttention applies a dict only when edge_index is
    also given (masked_attention_block.py:502). So 1-D and 2-D attributes give outputs
    bit-identical to no attributes. Pinned until the encoder passes edge_index (and
    per-graph local indices) to the masked blocks.
    """
    enc = _enc(["M", "S"])
    x = _x3()
    with torch.no_grad():
        plain = enc(x, CYCLE)
        assert torch.equal(enc(x, CYCLE, torch.tensor([3.0, -4.0, 5.0])), plain)
        assert torch.equal(enc(x, CYCLE, torch.tensor([[1.0, 9.0]] * 3)), plain)
        # the Data path builds the dict from data.edge_index, an adj-only Data
        # builds an empty one; both are dropped the same way
        data = Data(x=x, edge_index=CYCLE, num_nodes=3)
        assert torch.equal(enc(x, data, torch.tensor([3.0, -4.0, 5.0])), plain)
        adj_only = Data(x=x, num_nodes=3)
        adj_only.adj = to_dense_adj(CYCLE)
        assert torch.equal(enc(x, adj_only, torch.tensor([3.0, -4.0, 5.0])), plain)


def test_self_attention_blocks_attend_to_padding_rows() -> None:
    """Finding: in a batch, an S block lets a short graph attend to padding.

    Graph A (2 nodes, edges 0<->1) alone vs batched with graph B (4 nodes): A is
    zero-padded to 4 rows (nsa_encoder.py:131-136) and SelfAttentionBlock has no mask,
    so A's outputs change with its batch companion under pattern ['S'] while pattern
    ['M'] (padding masked out, no isolated node) gives A the same output to 1e-6.
    Pinned until S blocks mask padded keys.
    """
    torch.manual_seed(12)
    xa, xb = torch.randn(2, 2), torch.randn(4, 2)
    a_edges = torch.tensor([[0, 1], [1, 0]])
    both_edges = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]])
    batch = torch.tensor([0, 0, 1, 1, 1, 1])
    for pattern, same in ((["S"], False), (["M"], True)):
        enc = _enc(pattern)
        with torch.no_grad():
            alone = enc(xa, a_edges)
            batched = enc(torch.cat([xa, xb]), both_edges, None, batch)
        assert batched.shape == (6, 4)
        assert torch.allclose(alone, batched[:2], atol=1e-6) is same, pattern


def test_batched_output_unpads_graphs_in_batch_order() -> None:
    """Each graph's rows come back in order: the batched rows of graph B equal B run
    alone when B is the largest graph (no padding for B) under pattern ['M'].

    B alone is also checked against the layer applied by hand, so the node order is
    pinned against something other than the encoder itself: with no edge attributes
    the encoder calls ``layer(h, adj_mask, None)`` (nsa_encoder.py:140, edge_attr_dict
    stays None), so enc(xb, CYCLE) = layers[0](input_proj(xb)[None],
    to_dense_adj(CYCLE).bool(), None)[0]. A padding loop that wrote x in reversed node
    order would break this identity.
    """
    torch.manual_seed(13)
    xa, xb = torch.randn(2, 2), torch.randn(3, 2)
    enc = _enc(["M"])
    with torch.no_grad():
        batched = enc(
            torch.cat([xa, xb]),
            torch.tensor([[0, 1, 2, 3, 4], [1, 0, 3, 4, 2]]),
            None,
            torch.tensor([0, 0, 1, 1, 1]),
        )
        b_alone = enc(xb, CYCLE)
        layer = enc.layers[0]
        assert isinstance(layer, NodeSelfAttention)
        by_hand = layer(enc.input_proj(xb)[None], to_dense_adj(CYCLE).bool(), None)[0]
    torch.testing.assert_close(batched[2:], b_alone)
    torch.testing.assert_close(b_alone, by_hand)
