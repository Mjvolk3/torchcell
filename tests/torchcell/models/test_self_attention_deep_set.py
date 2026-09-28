# tests/torchcell/models/test_self_attention_deep_set.py
# [[tests.torchcell.models.test_self_attention_deep_set]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_self_attention_deep_set.py
"""``SelfAttention`` and ``SelfAttentionDeepSet`` on six nodes in two sets of three.

Parameter arithmetic for ``SelfAttentionDeepSet(4, 8, 3, num_node_layers=3,
num_set_layers=2, num_heads=2)``: a block is ``in * out + out + 2 * out``. Attention
widens every node layer after the first to ``hidden * heads = 16``, so the node blocks
are 4->8 (56), 16->8 (152), 16->3 (57) = 265; the set blocks 3->8 (48) and 8->3 (33) =
81; ``SelfAttention(8, 8, 2)`` is three ``Linear(8, 16)`` = 3 * 144 = 432; total 778.

``SelfAttention`` closed form with identity projections on ``x = I_2``: scores are
``I / sqrt(2)``, so each row's softmax puts ``p = e^(1/sqrt 2) / (e^(1/sqrt 2) + 1)``
(0.66976...) on the diagonal and ``1 - p`` off it, and the output rows are ``[p, 1 - p]``
and ``[1 - p, p]``.
"""

import math
from typing import cast

import pytest
import torch

from torchcell.models.self_attention_deep_set import SelfAttention, SelfAttentionDeepSet

BATCH = torch.tensor([0, 0, 0, 1, 1, 1])
P_DIAG = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)


def _x(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(6, 4)


def _count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _identity_attention() -> SelfAttention:
    attn = SelfAttention(dim_in=2, dim_out=2, num_heads=1)
    with torch.no_grad():
        for linear in (attn.query, attn.key, attn.value):
            linear.weight.copy_(torch.eye(2))
            linear.bias.zero_()
    return attn


@pytest.mark.parametrize("norm", ["batch", "instance", "layer"])
def test_parameter_count_is_778_for_every_norm(norm: str) -> None:
    """265 node + 81 set + 432 attention parameters."""
    model = SelfAttentionDeepSet(4, 8, 3, 3, 2, num_heads=2, norm=norm)
    assert _count(model) == 778
    assert _count(model.node_layers) == 265
    assert _count(model.set_layers) == 81
    assert _count(model.self_attn) == 432
    widened = [
        cast(torch.nn.Linear, cast(torch.nn.Sequential, block)[0]).in_features
        for block in list(model.node_layers)[1:]
    ]
    assert widened == [16, 16]


def test_constructor_rejects_unknown_norm_and_activation() -> None:
    """Both guards are assertions with fixed messages."""
    with pytest.raises(AssertionError, match="Invalid norm type"):
        SelfAttentionDeepSet(4, 8, 3, 2, 2, norm="rms")
    with pytest.raises(AssertionError, match="Invalid activation type"):
        SelfAttentionDeepSet(4, 8, 3, 2, 2, activation="swish")


def test_identity_attention_on_the_identity_input_is_the_closed_form() -> None:
    """Rows ``[p, 1 - p]`` / ``[1 - p, p]`` with ``p = e^(1/sqrt2) / (e^(1/sqrt2) + 1)``."""
    attn = _identity_attention()
    out, weights = attn(torch.eye(2), torch.zeros(2, dtype=torch.long))
    expected = torch.tensor([[P_DIAG, 1 - P_DIAG], [1 - P_DIAG, P_DIAG]])
    torch.testing.assert_close(out, expected)
    assert len(weights) == 1
    assert weights[0].shape == (1, 2, 2)
    torch.testing.assert_close(weights[0][0], expected)
    torch.testing.assert_close(weights[0][0].sum(-1), torch.ones(2))


def test_attention_softmax_spans_the_whole_batch_not_each_graph() -> None:
    """Finding: the softmax runs over every node in the batch (``self_attention_deep_set.py:39-41``).

    Splitting the two identity nodes into two graphs leaves the scores unchanged, so each
    graph's 1x1 attention slice is ``[[p]]`` with ``p`` < 1: the per-graph weights do not
    sum to one and a node attends to nodes of other graphs.
    """
    attn = _identity_attention()
    out, weights = attn(torch.eye(2), torch.tensor([0, 1]))
    torch.testing.assert_close(
        out, torch.tensor([[P_DIAG, 1 - P_DIAG], [1 - P_DIAG, P_DIAG]])
    )
    assert [w.shape for w in weights] == [(1, 1, 1), (1, 1, 1)]
    torch.testing.assert_close(weights[0], torch.full((1, 1, 1), P_DIAG))
    torch.testing.assert_close(weights[1], torch.full((1, 1, 1), P_DIAG))
    assert weights[0].item() < 1.0


def test_set_output_of_one_graph_depends_on_the_other_graph_in_the_batch() -> None:
    """The cross-graph attention makes graph 0's pooled output change when graph 1 changes,
    while the node stack before attention (layer 0) is unaffected.
    """
    torch.manual_seed(1)
    model = SelfAttentionDeepSet(4, 8, 3, 3, 2, num_heads=2, norm="layer").eval()
    x = _x()
    shifted = x.clone()
    shifted[3:] += 1.0
    _, x_set, _ = model(x, BATCH)
    _, x_set_shifted, _ = model(shifted, BATCH)
    assert not torch.allclose(x_set[0], x_set_shifted[0])
    torch.testing.assert_close(
        model.node_layers[0](x)[:3], model.node_layers[0](shifted)[:3]
    )


def test_set_output_is_permutation_invariant_and_node_output_equivariant() -> None:
    """Permuting all nodes with their ``batch`` labels permutes ``x_node`` and fixes ``x_set``."""
    torch.manual_seed(1)
    model = SelfAttentionDeepSet(4, 8, 3, 3, 2, num_heads=2, norm="layer").eval()
    x = _x()
    perm = torch.tensor([5, 2, 0, 4, 1, 3])
    x_node, x_set, weights = model(x, BATCH)
    x_node_p, x_set_p, weights_p = model(x[perm], BATCH[perm])
    torch.testing.assert_close(x_set_p, x_set)
    torch.testing.assert_close(x_node_p, x_node[perm])
    assert x_set.shape == (2, 3)
    assert x_node.shape == (6, 3)
    # one attention call per node layer after the first, one weight block per graph
    assert len(weights) == 2
    assert [w.shape for w in weights[0]] == [(2, 3, 3), (2, 3, 3)]
    assert [w.shape for w in weights_p[1]] == [(2, 3, 3), (2, 3, 3)]


def test_seeded_construction_is_deterministic_and_every_parameter_trains() -> None:
    """Same seed, same state dict and outputs; ``x_set.sum()`` reaches all 778 parameters."""
    torch.manual_seed(3)
    first = SelfAttentionDeepSet(4, 8, 3, 3, 2, num_heads=2)
    torch.manual_seed(3)
    second = SelfAttentionDeepSet(4, 8, 3, 3, 2, num_heads=2)
    for (name, a), (_, b) in zip(
        first.state_dict().items(), second.state_dict().items()
    ):
        assert torch.equal(a, b), name
    x = _x()
    first.eval()
    second.eval()
    torch.testing.assert_close(first(x, BATCH)[1], second(x, BATCH)[1])
    first.train()
    _, x_set, _ = first(x, BATCH)
    assert torch.isfinite(x_set).all()
    x_set.sum().backward()
    assert [n for n, p in first.named_parameters() if p.grad is None] == []
    assert sum(p.numel() for p in first.parameters() if p.grad is not None) == 778


def test_skip_connections_add_the_attended_input_and_the_middle_set_input() -> None:
    """With one head the attention output is 8 wide, so the middle node block 8->8 takes
    ``block(a) + a``; with three set blocks (3->8, 8->8, 8->3) only the middle one skips.
    Parameters: node 56 + 88 + 33 = 177, set 48 + 88 + 33 = 169, attention 3 * 72 = 216; 562.
    """
    torch.manual_seed(4)
    model = SelfAttentionDeepSet(
        4, 8, 3, 3, 3, num_heads=1, norm="layer", skip_node=True, skip_set=True
    ).eval()
    assert _count(model) == 562
    assert _count(model.set_layers) == 169
    x = _x()
    b0, b1, b2 = model.node_layers[0], model.node_layers[1], model.node_layers[2]
    a1, w1 = model.self_attn(b0(x), BATCH)
    h1 = b1(a1) + a1
    a2, w2 = model.self_attn(h1, BATCH)
    x_node, weights = model.node_layers_forward(x, BATCH)
    torch.testing.assert_close(x_node, b2(a2))
    assert len(weights) == 2
    torch.testing.assert_close(weights[0][1], w1[1])
    torch.testing.assert_close(weights[1][0], w2[0])
    s0, s1, s2 = model.set_layers[0], model.set_layers[1], model.set_layers[2]
    pooled = torch.randn(2, 3)
    g0 = s0(pooled)
    g1 = s1(g0) + g0
    torch.testing.assert_close(model.set_layers_forward(pooled), s2(g1))


def test_single_node_layer_skips_attention_and_misfits_the_set_stack() -> None:
    """Finding: ``num_node_layers=1`` maps 4->8 with no attention call, and the set stack
    built for ``out_channels=3`` rejects the 8-wide input (``self_attention_deep_set.py:94-112``).
    """
    model = SelfAttentionDeepSet(4, 8, 3, num_node_layers=1, num_set_layers=2)
    x = _x()
    x_node, weights = model.node_layers_forward(x, BATCH)
    assert x_node.shape == (6, 8)
    assert weights == []
    with pytest.raises(
        RuntimeError, match=r"mat1 and mat2 shapes cannot be multiplied \(2x8 and 3x8\)"
    ):
        model(x, BATCH)
