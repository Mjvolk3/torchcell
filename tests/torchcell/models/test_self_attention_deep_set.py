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

2026.09.30 (Phase 12). Added:

- Two heads: query and key are two stacked identities and value is ``[I; 2I]``, so both
  heads score as above and the concatenated output rows are ``[p, 1 - p, 2p, 2(1 - p)]``
  and ``[1 - p, p, 2(1 - p), 2p]``; each graph's weight block is ``(2, 2, 2)``.
- Relabeling graphs (batch ``[1, 1, 1, 0, 0, 0]``) swaps the two rows of ``x_set`` and
  leaves ``x_node`` unchanged (sum pooling is indexed by the label). Float summation order
  changes under a node permutation, so the permutation identities use ``assert_close``
  at the default float32 tolerance rather than bitwise equality.
- Sum pooling: the set stack receives exactly ``x_node[:3].sum(0)`` and
  ``x_node[3:].sum(0)``; a gap in the labels (``[0, 0, 2, 2]``) pools a zero row for the
  absent graph 1, which still passes through the set stack.
- Dropout: the last set block is followed by ``Dropout(p)``; at ``p = 1`` in train mode
  ``x_set`` is exactly zero, in eval mode it equals the dropout-free stack.
- A first node layer whose width is preserved (``in = hidden = 8``) is skipped too:
  ``block(x) + x``.
- Findings: ``norm="instance"`` builds but cannot run a forward on the ``[N, C]`` node
  matrix; one set layer returns ``hidden_channels`` wide with no dropout; ``main()`` prints
  10064 parameters (node 416 + 2144 + 536, set 352 + 280, attention 3 * 2112) and leaves
  autograd anomaly detection on for the process.
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


def test_two_heads_concatenate_per_head_outputs() -> None:
    """Value ``[I; 2I]``: rows ``[p, 1 - p, 2p, 2(1 - p)]`` and ``[1 - p, p, 2(1 - p), 2p]``."""
    attn = SelfAttention(dim_in=2, dim_out=2, num_heads=2)
    eye = torch.eye(2)
    with torch.no_grad():
        attn.query.weight.copy_(torch.cat([eye, eye]))
        attn.key.weight.copy_(torch.cat([eye, eye]))
        attn.value.weight.copy_(torch.cat([eye, 2 * eye]))
        for linear in (attn.query, attn.key, attn.value):
            linear.bias.zero_()
    out, weights = attn(eye, torch.zeros(2, dtype=torch.long))
    p, q = P_DIAG, 1 - P_DIAG
    torch.testing.assert_close(
        out, torch.tensor([[p, q, 2 * p, 2 * q], [q, p, 2 * q, 2 * p]])
    )
    assert len(weights) == 1 and weights[0].shape == (2, 2, 2)
    torch.testing.assert_close(weights[0][0], weights[0][1])


def test_relabeling_graphs_swaps_the_set_rows() -> None:
    """Batch ``[1, 1, 1, 0, 0, 0]``: ``x_set`` rows swap, ``x_node`` is unchanged."""
    torch.manual_seed(1)
    model = SelfAttentionDeepSet(4, 8, 3, 3, 2, num_heads=2, norm="layer").eval()
    x = _x()
    x_node, x_set, _ = model(x, BATCH)
    x_node_r, x_set_r, _ = model(x, 1 - BATCH)
    torch.testing.assert_close(x_set_r, x_set[[1, 0]])
    torch.testing.assert_close(x_node_r, x_node)


def test_set_stack_receives_the_per_graph_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sum pooling: the set stack's input rows are ``x_node[:3].sum(0)``, ``x_node[3:].sum(0)``."""
    torch.manual_seed(2)
    model = SelfAttentionDeepSet(4, 8, 3, 3, 2, num_heads=2, norm="layer").eval()
    seen: list[torch.Tensor] = []
    original = model.set_layers_forward

    def spy(x_summed: torch.Tensor) -> torch.Tensor:
        seen.append(x_summed)
        return original(x_summed)

    monkeypatch.setattr(model, "set_layers_forward", spy)
    x_node, x_set, _ = model(_x(), BATCH)
    expected = torch.stack([x_node[:3].sum(0), x_node[3:].sum(0)])
    torch.testing.assert_close(seen[0], expected)
    torch.testing.assert_close(x_set, original(expected))


def test_gap_in_graph_labels_pools_a_zero_row() -> None:
    """Labels ``[0, 0, 2, 2]``: three pooled rows, row 1 is the set stack on zeros; two
    attention blocks (``torch.unique`` sees only 0 and 2).
    """
    torch.manual_seed(2)
    model = SelfAttentionDeepSet(4, 8, 3, 3, 2, num_heads=2, norm="layer").eval()
    x = _x()[:4]
    _, x_set, weights = model(x, torch.tensor([0, 0, 2, 2]))
    assert x_set.shape == (3, 3)
    torch.testing.assert_close(x_set[1], model.set_layers_forward(torch.zeros(1, 3))[0])
    assert [w.shape for w in weights[0]] == [(2, 2, 2), (2, 2, 2)]


def test_final_dropout_zeroes_the_set_output_in_training() -> None:
    """``dropout_prob=1``: train mode gives exact zeros; eval mode is the stack without it."""
    torch.manual_seed(5)
    model = SelfAttentionDeepSet(
        4, 8, 3, 3, 2, num_heads=2, norm="layer", dropout_prob=1.0
    )
    dropout = model.set_layers[-1]
    assert isinstance(dropout, torch.nn.Dropout) and dropout.p == 1.0
    assert len(model.set_layers) == 3
    x = _x()
    _, x_set, _ = model.train()(x, BATCH)
    assert torch.equal(x_set, torch.zeros(2, 3))
    model.eval()
    x_node, x_set_eval, _ = model(x, BATCH)
    pooled = torch.stack([x_node[:3].sum(0), x_node[3:].sum(0)])
    expected = model.set_layers[1](model.set_layers[0](pooled))
    torch.testing.assert_close(x_set_eval, expected)


def test_width_preserving_first_node_layer_is_skipped() -> None:
    """``in = hidden = 8``, one head: layer 0 gives ``block(x) + x`` before attention."""
    torch.manual_seed(6)
    model = SelfAttentionDeepSet(
        8, 8, 3, 2, 2, num_heads=1, norm="layer", skip_node=True
    ).eval()
    torch.manual_seed(0)
    x = torch.randn(6, 8)
    b0, b1 = model.node_layers[0], model.node_layers[1]
    h0 = b0(x) + x
    attended, _ = model.self_attn(h0, BATCH)
    x_node, _ = model.node_layers_forward(x, BATCH)
    # the last block maps 8 -> 3, so it is never skipped
    torch.testing.assert_close(x_node, b1(attended))


def test_instance_norm_cannot_run_on_the_node_matrix() -> None:
    """Finding: ``InstanceNorm1d(8)`` (``self_attention_deep_set.py:88``) reads the
    ``[N, C]`` node matrix as one unbatched ``[C, L]`` sample, so six nodes fail its
    ``num_features`` check (and with N == 8 it would normalize across nodes). The norm
    builds (the parameter count test) but every forward raises. Pinned until the block
    transposes or drops the option.
    """
    model = SelfAttentionDeepSet(4, 8, 3, 3, 2, num_heads=2, norm="instance")
    with pytest.raises(
        ValueError,
        match=r"^expected input's size at dim=0 to match num_features \(8\), but got: 6\.$",
    ):
        model(_x(), BATCH)


def test_one_set_layer_returns_hidden_width_without_dropout() -> None:
    """Finding: with ``num_set_layers=1`` only the ``i == 0`` branch runs
    (``self_attention_deep_set.py:116-118``): the set output is ``hidden_channels`` (8)
    wide instead of ``out_channels`` (3) and there is no dropout. Pinned until the single
    layer maps to ``out_channels``.
    """
    torch.manual_seed(7)
    model = SelfAttentionDeepSet(4, 8, 3, 3, 1, num_heads=2, norm="layer").eval()
    assert len(model.set_layers) == 1
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.set_layers.modules())
    x_node, x_set, _ = model(_x(), BATCH)
    assert x_set.shape == (2, 8)
    pooled = torch.stack([x_node[:3].sum(0), x_node[3:].sum(0)])
    torch.testing.assert_close(x_set, model.set_layers[0](pooled))


def test_main_prints_the_parameter_count_and_leaves_anomaly_detection_on(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``main()``: 10064 parameters (node 416 + 2144 + 536, set 352 + 280, attention
    3 * (32 * 64 + 64)); five graphs of 20 nodes, two attention layers of (2, 20, 20)
    blocks. Finding: it calls ``set_detect_anomaly(True)`` (``self_attention_deep_set.py:177``)
    and never resets it, so every later backward in the process runs anomaly checks. Pinned
    until it uses the context manager. The loss line is unseeded and
    is not asserted.
    """
    from torchcell.models.self_attention_deep_set import main

    assert not torch.is_anomaly_enabled()
    try:
        main()
        assert torch.is_anomaly_enabled()
    finally:
        torch.autograd.set_detect_anomaly(False)
    lines = capsys.readouterr().out.splitlines()
    block = [
        f"Attention weights shape for graph {g} at layer {layer}: torch.Size([2, 20, 20])"
        for layer in (1, 2)
        for g in range(1, 6)
    ]
    expected = [
        "Number of parameters: 10064",
        "x shape: torch.Size([100, 10])",
        "x_set shape: torch.Size([5, 8])",
        "x_nodes shape: torch.Size([100, 8])",
        "Number of attention weights: 2",
        "Number of graphs at layer 1: 5",
        *block[:5],
        "Number of graphs at layer 2: 5",
        *block[5:],
        "torch.Size([5, 8]) torch.Size([5, 8])",
    ]
    assert lines[: len(expected)] == expected
    assert lines[len(expected)].startswith("Loss: ")
    assert lines[len(expected) + 1 :] == ["Gradients computed successfully!"]
