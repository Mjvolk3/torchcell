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

2026.09.30 - Phase 13 additions on the same six-node fixture. Relabeling the sets
(swapping ids 0 and 1) swaps the rows of ``x_set``; a set id with no nodes aggregates
to the zero vector, so its output row is ``set_layers(0)``; with node layers present,
``mean`` pooling times the set size (3) equals ``sum`` pooling for the same weights.
``norm="batch"`` in train mode couples the sets (moving set 1's nodes by +5 changes set
0's output) while ``norm="layer"`` does not; the trailing Dropout at ``p = 1`` zeroes
every set output in train mode and is the identity in eval. On
``DeepSet(4, 8, 3, 3, 2, skip_node=True)`` only the 8->8 middle node block gets a skip
(4->8 and 8->3 differ in width), so ``x_node = b2(b1(b0(x)) + b0(x))``. ``main()`` is run
with its printed shape lines pinned and anomaly detection off again after it returns. An
unknown ``aggregation`` raises ``ValueError`` naming it, at construction and at forward
(2026.09.30, issue #525: both Findings retired).
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
    with pytest.raises(
        ValueError,
        match=r"^Unknown aggregation 'max'; expected one of \('sum', 'mean'\)$",
    ):
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


# --- 2026.09.30 Phase 13 ------------------------------------------------------------ #
def _layer_model(seed: int = 4, skip_node: bool = False) -> DeepSet:
    torch.manual_seed(seed)
    return DeepSet(4, 8, 3, 3, 2, norm="layer", skip_node=skip_node).eval()


def test_relabeling_the_sets_permutes_the_set_rows() -> None:
    """``batch`` [1, 1, 1, 0, 0, 0] is the fixture with the two set ids swapped, so the
    set output rows swap while the node output is unchanged.
    """
    model = _layer_model()
    x = _x()
    x_node, x_set = model(x, BATCH)
    x_node_r, x_set_r = model(x, 1 - BATCH)
    torch.testing.assert_close(x_set_r, x_set[[1, 0]])
    torch.testing.assert_close(x_node_r, x_node)


def test_an_empty_set_id_aggregates_to_zero() -> None:
    """``batch`` [0, 0, 0, 2, 2, 2] leaves set 1 empty: ``scatter`` gives it the zero
    vector, and its output row is the set stack applied to zeros.
    """
    model = _layer_model()
    x = _x()
    gapped = torch.tensor([0, 0, 0, 2, 2, 2])
    x_node, x_set = model(x, gapped)
    assert x_set.shape == (3, 3)
    zero_row = model.set_layers_forward(torch.zeros(1, 3))
    torch.testing.assert_close(x_set[1:2], zero_row)
    torch.testing.assert_close(x_set[[0, 2]], model(x, BATCH)[1])
    torch.testing.assert_close(scatter_add(x_node, gapped, dim=0)[1], torch.zeros(3))


def test_mean_pooling_is_sum_pooling_over_the_set_size() -> None:
    """Same seed, node layers present, no set layers: every set has 3 nodes, so
    ``3 * mean == sum`` and the node outputs agree exactly.
    """
    torch.manual_seed(5)
    summed = DeepSet(4, 8, 3, 3, 0, norm="layer", aggregation="sum").eval()
    torch.manual_seed(5)
    averaged = DeepSet(4, 8, 3, 3, 0, norm="layer", aggregation="mean").eval()
    x = _x()
    node_s, set_s = summed(x, BATCH)
    node_m, set_m = averaged(x, BATCH)
    torch.testing.assert_close(node_s, node_m)
    torch.testing.assert_close(set_m * 3, set_s)
    torch.testing.assert_close(
        set_s, torch.stack([node_s[:3].sum(0), node_s[3:].sum(0)])
    )


@pytest.mark.parametrize(("norm", "coupled"), [("batch", True), ("layer", False)])
def test_batch_norm_in_train_mode_couples_the_sets(norm: str, coupled: bool) -> None:
    """Set 0's nodes are fixed and set 1's are shifted by +5. Under ``BatchNorm1d`` in
    train mode the batch statistics include set 1, so set 0's output moves; under
    ``LayerNorm`` each row is normalized alone and set 0's output is unchanged.
    Permutation invariance still holds under train-mode batch norm, because the batch
    statistics are symmetric in the rows.
    """
    torch.manual_seed(6)
    model = DeepSet(4, 8, 3, 3, 2, norm=norm, dropout_prob=0.0).train()
    x = _x()
    shifted = x.clone()
    shifted[3:] += 5.0
    set0 = model(x, BATCH)[1][0]
    set0_shifted = model(shifted, BATCH)[1][0]
    assert (not torch.allclose(set0, set0_shifted)) is coupled
    perm = torch.tensor([5, 2, 0, 4, 1, 3])
    torch.testing.assert_close(model(x[perm], BATCH[perm])[1], model(x, BATCH)[1])


def test_trailing_dropout_at_one_zeroes_train_output_and_is_identity_in_eval() -> None:
    """Dropout is the last set module, so at ``p = 1`` every train-mode set output is 0;
    in eval it is skipped and the output equals the ``p = 0`` model at the same seed.
    """
    torch.manual_seed(7)
    full = DeepSet(4, 8, 3, 3, 2, norm="layer", dropout_prob=1.0)
    torch.manual_seed(7)
    none = DeepSet(4, 8, 3, 3, 2, norm="layer", dropout_prob=0.0)
    x = _x()
    assert torch.equal(full.train()(x, BATCH)[1], torch.zeros(2, 3))
    torch.testing.assert_close(full.eval()(x, BATCH)[1], none.eval()(x, BATCH)[1])


def test_skip_node_adds_only_where_the_block_keeps_its_width() -> None:
    """Blocks 4->8, 8->8, 8->3: only the middle one adds its input."""
    model = _layer_model(skip_node=True)
    x = _x()
    b0, b1, b2 = model.node_layers
    h0 = b0(x)
    torch.testing.assert_close(model.node_layers_forward(x), b2(b1(h0) + h0))


def test_activation_name_selects_the_block_activation_and_is_validated() -> None:
    """``activation="tanh"`` makes tanh the last module of every block (line 62), so
    every node output lies strictly inside (-1, 1); an unregistered name fails the
    constructor's assertion (line 45) with its exact message.
    """
    torch.manual_seed(0)
    model = DeepSet(4, 8, 3, 3, 2, norm="layer", activation="tanh").eval()
    x_node, x_set = model(_x(), BATCH)
    assert x_node.shape == (6, 3)
    assert float(x_node.abs().max()) < 1.0
    assert float(x_set.abs().max()) < 1.0
    with pytest.raises(AssertionError, match="Invalid activation type"):
        DeepSet(4, 8, 3, 3, 2, activation="nope")


def test_aggregation_changed_after_construction_raises_value_error_at_forward() -> None:
    """``forward`` re-checks the attribute: set to ``"max"`` after construction, it
    raises the same named ``ValueError`` as the constructor, before any layer runs.
    """
    model = _layer_model()
    model.aggregation = "max"
    with pytest.raises(
        ValueError,
        match=r"^Unknown aggregation 'max'; expected one of \('sum', 'mean'\)$",
    ):
        model(_x(), BATCH)


def test_main_prints_the_demo_shapes_and_leaves_anomaly_detection_off(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main()`` enables anomaly detection only inside its forward/backward: the
    loss is computed with it on (recorded by a ``MSELoss.forward`` spy) and it is off
    again when ``main`` returns. The shape lines follow from 100 nodes in 5 sets of 20
    with ``in_channels`` 10 and ``out_channels`` 8.
    """
    assert not torch.is_anomaly_enabled()
    seen: list[bool] = []
    real_forward = torch.nn.MSELoss.forward

    def spy(self: torch.nn.MSELoss, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        seen.append(torch.is_anomaly_enabled())
        return real_forward(self, a, b)

    monkeypatch.setattr(torch.nn.MSELoss, "forward", spy)
    try:
        torch.manual_seed(0)
        from torchcell.models.deep_set import main

        main()
        after = torch.is_anomaly_enabled()
    finally:
        torch.autograd.set_detect_anomaly(False)
    assert (seen, after) == ([True], False)
    lines = capsys.readouterr().out.splitlines()
    assert lines[:6] == [
        "x shape: torch.Size([100, 10])",
        "x_set shape: torch.Size([5, 8])",
        "batch shape: torch.Size([100])",
        "batch unique: tensor([0, 1, 2, 3, 4])",
        "x_nodes shape: torch.Size([100, 8])",
        "torch.Size([5, 8]) torch.Size([5, 8])",
    ]
    assert lines[6].startswith("Loss: ")
    assert float(lines[6].removeprefix("Loss: ")) >= 0.0
    assert lines[7:] == ["Gradients computed successfully!"]
