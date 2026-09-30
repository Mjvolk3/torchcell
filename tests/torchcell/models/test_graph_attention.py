# tests/torchcell/models/test_graph_attention.py
# [[tests.torchcell.models.test_graph_attention]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_graph_attention.py
"""``GraphAttention``: legacy DeepSet encoder, stacked GATv2 layers, sum readout, set MLP.

2026.09.30, Phase 18. The shipped class cannot be built, for two independent reasons
(both Findings, pinned by the first two tests): it calls ``DeepSet(in_channels=...,
node_layers=..., set_layers=[], ...)``, a signature the current ``DeepSet`` no longer
has, and it asks for ``GATConv(in_dim, out_dim, v2=True)``, a keyword PyG's
``GATConv`` does not take (GATv2 is the separate class ``GATv2Conv``). Everything after
that is the wrapper's own code, exercised with two module-level names replaced:

* ``DeepSet`` by ``_LegacyDeepSet``, which accepts the legacy keywords, records them,
  keeps ``skip_set`` (the wrapper reads ``self.deepset.skip_set``) and encodes nodes
  with plain ``nn.Linear`` layers along ``[input_dim, *node_layers]`` (identity when
  ``node_layers`` is empty);
* ``GATConv`` by a factory that records ``(in_dim, out_dim, keywords)`` and returns the
  real ``GATv2Conv(in_dim, out_dim)``, which is what ``v2=True`` asks for.

Parameter arithmetic: ``GATv2Conv(i, o)`` (one head, separate source and target
weights) holds ``lin_l`` and ``lin_r`` (each ``o * i`` weights plus ``o`` bias), ``att``
``[1, 1, o]`` and ``bias`` ``[o]``, so ``2 * o * (i + 2)`` parameters;
``nn.Linear(i, o)`` has ``o * (i + 1)``.

Closed form, one layer at width 2 on the directed star 0 -> 1, 2 -> 1 with ``att``
zeroed: every attention logit is 0, so each node averages ``lin_l`` of itself (PyG adds
the self loop) and its in-neighbors, and ``lin_r`` drops out. With ``lin_l.weight = W =
[[1, 1], [0, 1]]``, ``lin_l.bias = c = (0.25, 0)``, ``bias = b = (0.5, -0.5)`` and ``x =
[(1, 0), (0, 1), (2, 2)]``, ``W x = [(1, 0), (1, 1), (4, 2)]`` and

* node 0: ``(1, 0) + c + b = (1.75, -0.5)``
* node 1: ``((1, 0) + (1, 1) + (4, 2)) / 3 + c + b = (2, 1) + c + b = (2.75, 0.5)``
* node 2: ``(4, 2) + c + b = (4.75, 1.5)``

(checked against ``GATv2Conv`` directly before being written in). With ``batch = [0, 1,
1]`` the sum readout gives set 0 = node 0 and set 1 = node 1 + node 2.
"""

import math
from typing import Any

import pytest
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

import torchcell.models.graph_attention as ga_module
from torchcell.models.graph_attention import GraphAttention


class _LegacyDeepSet(nn.Module):
    """Accepts the keywords ``GraphAttention`` passes; Linear node layers only."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.received = dict(kwargs)
        self.skip_set = bool(kwargs["skip_set"])
        dims = [int(kwargs["in_channels"]), *kwargs["node_layers"]]
        self.node_layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])
        )

    def node_layers_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Linear layers in order."""
        for layer in self.node_layers:
            x = layer(x)
        return x


GatCall = tuple[int, int, dict[str, object]]


@pytest.fixture
def gat_calls(monkeypatch: pytest.MonkeyPatch) -> list[GatCall]:
    """Swap in the legacy DeepSet and a recording GATv2 factory; return its log."""
    calls: list[GatCall] = []

    def factory(in_dim: int, out_dim: int, **kwargs: object) -> GATv2Conv:
        calls.append((in_dim, out_dim, kwargs))
        return GATv2Conv(in_dim, out_dim)

    monkeypatch.setattr(ga_module, "DeepSet", _LegacyDeepSet)
    monkeypatch.setattr(ga_module, "GATConv", factory)
    return calls


def _build(**overrides: Any) -> GraphAttention:
    """Input 3, node layers [4], set layers [2], hidden 5, two GAT layers."""
    kwargs: dict[str, Any] = {
        "input_dim": 3,
        "node_layers": [4],
        "set_layers": [2],
        "hidden_channels": 5,
        "num_layers": 2,
    }
    kwargs.update(overrides)
    return GraphAttention(**kwargs)


def _convs(model: GraphAttention) -> list[GATv2Conv]:
    """The model's message-passing layers, typed."""
    layers = list(model.gat_layers)
    convs = [layer for layer in layers if isinstance(layer, GATv2Conv)]
    assert len(convs) == len(layers)
    return convs


def _linears(model: GraphAttention) -> list[nn.Linear]:
    """The model's post-readout set layers, typed."""
    layers = list(model.set_layers)
    linears = [layer for layer in layers if isinstance(layer, nn.Linear)]
    assert len(linears) == len(layers)
    return linears


def _widths(model: GraphAttention) -> list[tuple[int, int]]:
    """``(in_features, out_features)`` of each set layer."""
    return [(lin.in_features, lin.out_features) for lin in _linears(model)]


STAR = torch.tensor([[0, 2], [1, 1]])
X3 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
W = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
C = torch.tensor([0.25, 0.0])
B = torch.tensor([0.5, -0.5])


def _toy_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Five nodes in two sets, six directed edges, seeded features of width 3."""
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(5, 3, generator=gen)
    batch = torch.tensor([0, 0, 1, 1, 1])
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 2], [1, 0, 3, 4, 2, 4]])
    return x, batch, edge_index


def test_constructor_refuses_against_the_current_deep_set() -> None:
    """Finding: the shipped class cannot be constructed (first failure: the encoder).

    graph_attention.py line 33 passes ``node_layers=`` to ``DeepSet``, whose current
    keywords are ``in_channels``, ``hidden_channels``, ``out_channels``,
    ``num_node_layers``, ``num_set_layers``. Pinned until the wrapper is ported or
    retired (its only caller is under
    ``experiments/DEPRECATED_costanzo_smf_dmf_supervised``).
    """
    with pytest.raises(TypeError) as excinfo:
        _build()
    assert str(excinfo.value) == (
        "DeepSet.__init__() got an unexpected keyword argument 'node_layers'."
        " Did you mean 'num_node_layers'?"
    )


def test_gat_conv_refuses_the_v2_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding: past the encoder, ``GATConv(..., v2=True)`` is refused by PyG.

    graph_attention.py line 58 relies on a ``v2`` switch that ``GATConv`` does not
    have; the keyword falls through to ``MessagePassing.__init__``. Pinned until the
    wrapper uses ``GATv2Conv``.
    """
    monkeypatch.setattr(ga_module, "DeepSet", _LegacyDeepSet)
    with pytest.raises(TypeError) as excinfo:
        _build()
    assert str(excinfo.value) == (
        "MessagePassing.__init__() got an unexpected keyword argument 'v2'"
    )


def test_forwards_the_legacy_keywords_and_requests_v2_per_layer(
    gat_calls: list[GatCall],
) -> None:
    """The encoder gets exactly these keywords; each GAT layer asks for ``v2=True``.

    ``set_layers`` is always ``[]`` (the wrapper builds its own set MLP), the unknown
    ``unused=1`` is swallowed, and with ``out_channels=6`` the layer requests are
    (4, 5) then (5, 6).
    """
    model = _build(
        out_channels=6,
        dropout_prob=0.3,
        norm="layer",
        activation="gelu",
        skip_node=True,
        skip_set=True,
        unused=1,
    )
    assert isinstance(model.deepset, _LegacyDeepSet)
    assert model.deepset.received == {
        "in_channels": 3,
        "node_layers": [4],
        "set_layers": [],
        "dropout_prob": 0.3,
        "norm": "layer",
        "activation": "gelu",
        "skip_node": True,
        "skip_set": True,
    }
    assert gat_calls == [(4, 5, {"v2": True}), (5, 6, {"v2": True})]


@pytest.mark.usefixtures("gat_calls")
def test_layer_widths_and_parameter_count_follow_the_constructor_arithmetic() -> None:
    """``out_channels=6``: GAT 4 -> 5 -> 6, set Linear 6 -> 2; 60 + 84 + 14 = 158.

    Counts per the module docstring: GATv2(4, 5) 2 * 5 * 6, GATv2(5, 6) 2 * 6 * 7,
    Linear(6, 2) 2 * 7. The stand-in encoder's Linear(3, 4) is excluded.
    """
    model = _build(out_channels=6)
    shapes = {
        name: tuple(p.shape)
        for name, p in model.named_parameters()
        if not name.startswith("deepset.")
    }
    assert shapes == {
        "gat_layers.0.att": (1, 1, 5),
        "gat_layers.0.bias": (5,),
        "gat_layers.0.lin_l.weight": (5, 4),
        "gat_layers.0.lin_l.bias": (5,),
        "gat_layers.0.lin_r.weight": (5, 4),
        "gat_layers.0.lin_r.bias": (5,),
        "gat_layers.1.att": (1, 1, 6),
        "gat_layers.1.bias": (6,),
        "gat_layers.1.lin_l.weight": (6, 5),
        "gat_layers.1.lin_l.bias": (6,),
        "gat_layers.1.lin_r.weight": (6, 5),
        "gat_layers.1.lin_r.bias": (6,),
        "set_layers.0.weight": (2, 6),
        "set_layers.0.bias": (2,),
    }
    assert sum(math.prod(s) for s in shapes.values()) == 158


def test_out_channels_none_keeps_hidden_width_and_empty_node_layers_use_input(
    gat_calls: list[GatCall],
) -> None:
    """Without ``out_channels`` every GAT is hidden-wide; no node layers feed input_dim.

    ``node_layers=[]``, ``num_layers=3``: GAT 3 -> 5, 5 -> 5, 5 -> 5, set Linear 5 -> 2.
    """
    model = _build(node_layers=[], num_layers=3)
    assert [(i, o) for i, o, _ in gat_calls] == [(3, 5), (5, 5), (5, 5)]
    assert _widths(model) == [(5, 2)]


@pytest.mark.usefixtures("gat_calls")
def test_set_layers_argument_is_mutated_in_place() -> None:
    """Finding: the caller's ``set_layers`` list gains the GAT width at index 0.

    graph_attention.py line 62 runs ``set_layers.insert(0, in_dim)`` on the argument,
    so reusing one list for a second model adds a Linear(5, 5) in front: the first
    model gets [Linear(5, 2)], the second [Linear(5, 5), Linear(5, 2)]. Pinned until
    the wrapper copies the list.
    """
    shared = [2]
    first = _build(set_layers=shared)
    assert shared == [5, 2]
    second = _build(set_layers=shared)
    assert shared == [5, 5, 2]
    assert _widths(first) == [(5, 2)]
    assert _widths(second) == [(5, 5), (5, 2)]


@pytest.mark.usefixtures("gat_calls")
def test_one_gat_layer_with_zero_attention_is_a_neighborhood_mean() -> None:
    """Zeroed ``att`` gives the uniform-attention values in the module docstring.

    ``lin_r`` is left at its random initialization: with zero logits it cannot
    matter, so a regression that fed ``lin_r`` into the message would fail here.
    """
    torch.manual_seed(0)
    model = _build(
        input_dim=2, node_layers=[], set_layers=[], hidden_channels=2, num_layers=1
    )
    (conv,) = _convs(model)
    with torch.no_grad():
        conv.lin_l.weight.copy_(W)
        conv.lin_l.bias.copy_(C)
        conv.att.zero_()
        conv.bias.copy_(B)
    x_node, x_set = model(X3, torch.tensor([0, 1, 1]), STAR)
    expected = torch.tensor([[1.75, -0.5], [2.75, 0.5], [4.75, 1.5]])
    torch.testing.assert_close(x_node, expected)
    torch.testing.assert_close(x_set, torch.tensor([[1.75, -0.5], [7.5, 2.0]]))


@pytest.mark.usefixtures("gat_calls")
def test_skip_mp_adds_the_layer_input_only_where_widths_match() -> None:
    """With ``skip_mp`` the 2 -> 2 layer adds its input; the 2 -> 3 layer cannot.

    Both models share one state dict. Expected, from the model's own convs:
    skip ``conv1(conv0(x) + x)``, no skip ``conv1(conv0(x))``.
    """
    torch.manual_seed(0)
    plain = _build(
        input_dim=2,
        node_layers=[],
        set_layers=[],
        hidden_channels=2,
        num_layers=2,
        out_channels=3,
    )
    skip = _build(
        input_dim=2,
        node_layers=[],
        set_layers=[],
        hidden_channels=2,
        num_layers=2,
        out_channels=3,
        skip_mp=True,
    )
    skip.load_state_dict(plain.state_dict())
    batch = torch.tensor([0, 0, 0])
    conv0, conv1 = _convs(plain)
    with torch.no_grad():
        torch.testing.assert_close(
            skip(X3, batch, STAR)[0], conv1(conv0(X3, STAR) + X3, STAR)
        )
        torch.testing.assert_close(
            plain(X3, batch, STAR)[0], conv1(conv0(X3, STAR), STAR)
        )


@pytest.mark.usefixtures("gat_calls")
@pytest.mark.parametrize(("skip_set", "scale"), [(True, 1.0), (False, 0.0)])
def test_skip_set_is_read_from_the_encoder_and_adds_on_equal_widths(
    skip_set: bool, scale: float
) -> None:
    """A zero-initialized 2 -> 2 set layer is the identity with skip, zero without.

    ``set_layers=[2, 3]`` builds Linear(2, 2) then Linear(2, 3). With the first zeroed
    and the second ``[[1, 0], [0, 1], [1, 1]]`` (no bias), the output is ``scale *
    [s0, s1, s0 + s1]`` for the summed set vector s; the 2 -> 3 layer has no skip.
    """
    model = _build(
        input_dim=2,
        node_layers=[],
        set_layers=[2, 3],
        hidden_channels=2,
        num_layers=1,
        skip_set=skip_set,
    )
    first, second = _linears(model)
    with torch.no_grad():
        first.weight.zero_()
        first.bias.zero_()
        second.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        second.bias.zero_()
        x_node, x_set = model(X3, torch.tensor([0, 0, 1]), STAR)
    s = torch.stack([x_node[0] + x_node[1], x_node[2]])
    expected = scale * torch.stack([s[:, 0], s[:, 1], s[:, 0] + s[:, 1]], dim=1)
    torch.testing.assert_close(x_set, expected)


@pytest.mark.usefixtures("gat_calls")
def test_node_permutation_equivariance_and_set_invariance() -> None:
    """Relabeling nodes permutes node outputs and leaves the set outputs unchanged.

    ``perm`` maps new position k to old node ``perm[k]``; edges are relabeled with the
    inverse permutation and the batch vector is permuted with the features. The
    attention softmax runs per target node, so it commutes with the relabeling.
    """
    torch.manual_seed(1)
    model = _build(out_channels=6, set_layers=[3]).eval()
    x, batch, edge_index = _toy_graph()
    perm = torch.tensor([3, 0, 4, 2, 1])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(5)
    with torch.no_grad():
        node, sets = model(x, batch, edge_index)
        node_p, sets_p = model(x[perm], batch[perm], inv[edge_index])
    torch.testing.assert_close(node_p, node[perm])
    torch.testing.assert_close(sets_p, sets)


@pytest.mark.usefixtures("gat_calls")
def test_gradient_reaches_every_parameter() -> None:
    """A loss on both outputs gives every parameter a finite, nonzero gradient.

    Includes ``att`` and ``lin_r`` of both GAT layers, which reach the output only
    through the attention weights.
    """
    torch.manual_seed(2)
    model = _build(out_channels=6, set_layers=[3])
    x, batch, edge_index = _toy_graph()
    node, sets = model(x, batch, edge_index)
    gen = torch.Generator().manual_seed(3)
    loss = (node * torch.randn(node.shape, generator=gen)).sum() + (
        sets * torch.randn(sets.shape, generator=gen)
    ).sum()
    loss.backward()
    names = [name for name, _ in model.named_parameters()]
    reached = [
        name
        for name, p in model.named_parameters()
        if p.grad is not None
        and bool(torch.isfinite(p.grad).all())
        and float(p.grad.abs().sum()) > 0.0
    ]
    assert reached == names
    assert len(names) == 16


@pytest.mark.usefixtures("gat_calls")
def test_seeded_construction_is_deterministic() -> None:
    """The same seed gives identical parameters and identical outputs."""
    x, batch, edge_index = _toy_graph()
    torch.manual_seed(5)
    first = _build(out_channels=6)
    torch.manual_seed(5)
    second = _build(out_channels=6)
    for (name_a, a), (name_b, b) in zip(
        first.state_dict().items(), second.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(a, b)
    with torch.no_grad():
        out_a = first(x, batch, edge_index)
        out_b = second(x, batch, edge_index)
    assert torch.equal(out_a[0], out_b[0])
    assert torch.equal(out_a[1], out_b[1])
