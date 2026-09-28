# tests/torchcell/models/test_deep_set.py
# [[tests.torchcell.models.test_deep_set]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_deep_set.py
"""``DeepSet`` on six nodes in two sets of three (``batch = [0, 0, 0, 1, 1, 1]``).

Parameter arithmetic for ``DeepSet(4, 8, 3, num_node_layers=3, num_set_layers=2)``: a
block is ``Linear(in, out)`` plus a norm with ``2 * out`` affine parameters, so
``in * out + out + 2 * out``. Node blocks 4->8, 8->8, 8->3 are 56 + 88 + 33 = 177; set
blocks 3->8 and 8->3 are 48 + 33 = 81; total 258, the same under batch, instance and
layer norm since each carries ``2 * out`` parameters. With no node layers the first set
block reads ``in_channels`` (4->8 = 56) so the total is 56 + 33 = 89.

Random weights carry no float contract, so every forward assertion is an identity:
permutation invariance of the set output, equivariance of the node output, closed-form
sum/mean pooling when both stacks are empty, the skip connections written as
``block(x) + x``, seeded determinism and a gradient on every parameter.
"""

from typing import cast

import pytest
import torch
from torch_scatter import scatter_add, scatter_mean

from torchcell.models.deep_set import DeepSet

BATCH = torch.tensor([0, 0, 0, 1, 1, 1])


def _x(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(6, 4)


def _count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("norm", ["batch", "instance", "layer"])
def test_parameter_count_is_258_for_every_norm(norm: str) -> None:
    """177 node + 81 set parameters; the three norms all carry 2 * out affine weights."""
    model = DeepSet(4, 8, 3, num_node_layers=3, num_set_layers=2, norm=norm)
    assert _count(model) == 258
    assert _count(model.node_layers) == 177
    assert _count(model.set_layers) == 81
    assert len(model.node_layers) == 3
    assert len(model.set_layers) == 3  # two blocks and the trailing Dropout
    assert isinstance(model.set_layers[2], torch.nn.Dropout)
    assert model.set_layers[2].p == 0.2


def test_without_node_layers_the_first_set_block_reads_the_input_width() -> None:
    """56 (4->8) + 33 (8->3) = 89, and the node output is the input itself."""
    model = DeepSet(4, 8, 3, num_node_layers=0, num_set_layers=2)
    assert _count(model) == 89
    x = _x()
    x_node, x_set = model(x, BATCH)
    assert x_node is x
    assert x_set.shape == (2, 3)


def test_constructor_rejects_unknown_norm_activation_and_aggregation() -> None:
    """Each guard is an assertion with a fixed message."""
    with pytest.raises(AssertionError, match="Invalid norm type"):
        DeepSet(4, 8, 3, 2, 2, norm="rms")
    with pytest.raises(AssertionError, match="Invalid activation type"):
        DeepSet(4, 8, 3, 2, 2, activation="swish")
    with pytest.raises(AssertionError, match="Invalid aggregation method"):
        DeepSet(4, 8, 3, 2, 2, aggregation="max")


def test_empty_stacks_expose_exact_sum_and_mean_pooling() -> None:
    """With no layers the set output is ``scatter_add`` or ``scatter_mean`` of the input."""
    x = _x()
    _, summed = DeepSet(4, 8, 3, 0, 0, aggregation="sum")(x, BATCH)
    _, averaged = DeepSet(4, 8, 3, 0, 0, aggregation="mean")(x, BATCH)
    torch.testing.assert_close(summed, torch.stack([x[:3].sum(0), x[3:].sum(0)]))
    torch.testing.assert_close(averaged, torch.stack([x[:3].mean(0), x[3:].mean(0)]))
    torch.testing.assert_close(summed, scatter_add(x, BATCH, dim=0))
    torch.testing.assert_close(averaged, scatter_mean(x, BATCH, dim=0))


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_set_output_is_permutation_invariant_and_node_output_equivariant(
    aggregation: str,
) -> None:
    """Permuting nodes (and ``batch`` with them) leaves ``x_set`` unchanged and permutes ``x_node``."""
    torch.manual_seed(1)
    model = DeepSet(4, 8, 3, 3, 2, norm="layer", aggregation=aggregation).eval()
    x = _x()
    perm = torch.tensor([5, 2, 0, 4, 1, 3])
    x_node, x_set = model(x, BATCH)
    x_node_p, x_set_p = model(x[perm], BATCH[perm])
    torch.testing.assert_close(x_set_p, x_set)
    torch.testing.assert_close(x_node_p, x_node[perm])
    assert x_set.shape == (2, 3)
    assert x_node.shape == (6, 3)


def test_skip_connections_add_the_block_input_where_widths_match() -> None:
    """``skip_node``: ``x_node = block(x) + x`` for a 2->2 block. ``skip_set`` adds the
    input of every set block but never around the trailing Dropout.
    """
    torch.manual_seed(2)
    node_model = DeepSet(2, 2, 2, num_node_layers=1, num_set_layers=0, norm="layer")
    x = torch.randn(6, 2)
    plain = node_model.node_layers_forward(x)
    node_model.skip_node = True
    torch.testing.assert_close(node_model.node_layers_forward(x), plain + x)
    torch.testing.assert_close(plain, node_model.node_layers[0](x))

    set_model = DeepSet(
        2, 2, 2, num_node_layers=0, num_set_layers=2, norm="layer"
    ).eval()
    pooled = scatter_add(x, BATCH, dim=0)
    b0, b1 = set_model.set_layers[0], set_model.set_layers[1]
    torch.testing.assert_close(set_model.set_layers_forward(pooled), b1(b0(pooled)))
    set_model.skip_set = True
    h1 = b0(pooled) + pooled
    torch.testing.assert_close(set_model.set_layers_forward(pooled), b1(h1) + h1)


def test_three_set_layers_insert_a_hidden_to_hidden_block() -> None:
    """Blocks 3->8, 8->8, 8->3 plus Dropout: 48 + 88 + 33 = 169 set parameters, 346 total."""
    model = DeepSet(4, 8, 3, num_node_layers=3, num_set_layers=3)
    assert _count(model) == 346
    assert _count(model.set_layers) == 169
    assert len(model.set_layers) == 4
    middle = cast(torch.nn.Sequential, model.set_layers[1])
    linear = cast(torch.nn.Linear, middle[0])
    assert (linear.in_features, linear.out_features) == (8, 8)
    assert model(_x(), BATCH)[1].shape == (2, 3)


def test_seeded_construction_is_deterministic_and_every_parameter_trains() -> None:
    """Same seed, same weights and outputs; ``x_set.sum()`` reaches all 258 parameters."""
    torch.manual_seed(3)
    first = DeepSet(4, 8, 3, 3, 2)
    torch.manual_seed(3)
    second = DeepSet(4, 8, 3, 3, 2)
    for (name, a), (_, b) in zip(
        first.state_dict().items(), second.state_dict().items()
    ):
        assert torch.equal(a, b), name
    x = _x()
    first.eval()
    second.eval()
    torch.testing.assert_close(first(x, BATCH)[1], second(x, BATCH)[1])
    first.train()
    _, x_set = first(x, BATCH)
    assert torch.isfinite(x_set).all()
    x_set.sum().backward()
    without_grad = [n for n, p in first.named_parameters() if p.grad is None]
    assert without_grad == []
    assert sum(p.numel() for p in first.parameters() if p.grad is not None) == 258


def test_single_layer_stacks_do_not_reach_out_channels() -> None:
    """Finding: with one layer the ``i == 0`` branch wins, so the block maps to ``hidden``.

    ``num_set_layers=1`` yields a single 3->8 block with no Dropout and a set output of
    width 8, not ``out_channels`` (``deep_set.py:82-91``); ``num_node_layers=1`` maps
    4->8 and the first set block (built 3->8) then fails on the 8-wide input.
    """
    x = _x()
    one_set = DeepSet(4, 8, 3, num_node_layers=2, num_set_layers=1)
    assert len(one_set.set_layers) == 1
    assert _count(one_set.set_layers) == 48
    assert one_set(x, BATCH)[1].shape == (2, 8)
    one_node = DeepSet(4, 8, 3, num_node_layers=1, num_set_layers=2)
    with pytest.raises(
        RuntimeError, match=r"mat1 and mat2 shapes cannot be multiplied \(2x8 and 3x8\)"
    ):
        one_node(x, BATCH)


def test_instance_norm_cannot_run_on_a_node_by_feature_matrix() -> None:
    """Finding: ``norm="instance"`` builds but ``InstanceNorm1d`` reads a 2-D input as
    ``(C, L)`` and rejects 6 rows against 8 features (``deep_set.py:59``).
    """
    model = DeepSet(4, 8, 3, 3, 2, norm="instance")
    with pytest.raises(
        ValueError,
        match=r"expected input's size at dim=0 to match num_features \(8\), but got: 6",
    ):
        model(_x(), BATCH)
