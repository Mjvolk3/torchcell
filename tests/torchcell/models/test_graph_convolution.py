# tests/torchcell/models/test_graph_convolution.py
# [[tests.torchcell.models.test_graph_convolution]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_graph_convolution.py
"""``GraphConvolution``: legacy DeepSet encoder, stacked GCN layers, sum readout, set MLP.

2026.09.30, Phase 18. The constructor calls ``DeepSet(input_dim=..., node_layers=...,
set_layers=[], ...)``, a signature the current ``DeepSet`` (``in_channels``,
``hidden_channels``, ``out_channels``, ``num_node_layers``, ``num_set_layers``) no
longer has, so the class cannot be built as shipped (a Finding, pinned by the first
test). Everything after that is the wrapper's own code, exercised with the module-level
``DeepSet`` name replaced by ``_LegacyDeepSet``: a stand-in that accepts the legacy
keywords, records them, keeps ``skip_set`` (the wrapper reads ``self.deepset.skip_set``)
and encodes nodes with plain ``nn.Linear`` layers along ``[input_dim, *node_layers]``
(identity when ``node_layers`` is empty). The GCN layers are the real PyG ``GCNConv``.

Parameter arithmetic: ``GCNConv(i, o)`` holds ``lin.weight`` ``[o, i]`` and ``bias``
``[o]``, so ``o * (i + 1)`` parameters; ``nn.Linear(i, o)`` also ``o * (i + 1)``.

Closed form, one layer at width 2 on the directed star 0 -> 1, 2 -> 1: GCNConv adds a
self loop per node and weights edge j -> i by ``1 / sqrt(d_j * d_i)`` with d the
in-degree counting the self loop, here d = (1, 3, 1). With ``lin.weight = W =
[[1, 1], [0, 1]]``, ``bias = b = (0.5, -0.5)`` and ``x = [(1, 0), (0, 1), (2, 2)]``,
``W x = [(1, 0), (1, 1), (4, 2)]`` and

* node 0: ``(1, 0) + b = (1.5, -0.5)``
* node 1: ``(1, 1) / 3 + ((1, 0) + (4, 2)) / sqrt(3) + b``
  ``= (1/3 + 5/sqrt(3) + 0.5, 1/3 + 2/sqrt(3) - 0.5) = (3.7201, 0.9880)``
* node 2: ``(4, 2) + b = (4.5, 1.5)``

(checked against ``GCNConv`` directly before being written in). With ``batch = [0, 1,
1]`` the sum readout gives set 0 = node 0 and set 1 = node 1 + node 2.
"""

import math
from typing import Any

import pytest
import torch
from torch import nn
from torch_geometric.nn import GCNConv

import torchcell.models.graph_convolution as gc_module
from torchcell.models.graph_convolution import GraphConvolution


class _LegacyDeepSet(nn.Module):
    """Accepts the keywords ``GraphConvolution`` passes; Linear node layers only."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.received = dict(kwargs)
        self.skip_set = bool(kwargs["skip_set"])
        dims = [int(kwargs["input_dim"]), *kwargs["node_layers"]]
        self.node_layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])
        )

    def node_layers_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Linear layers in order."""
        for layer in self.node_layers:
            x = layer(x)
        return x


@pytest.fixture
def legacy_deep_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the module's ``DeepSet`` for the legacy-signature stand-in."""
    monkeypatch.setattr(gc_module, "DeepSet", _LegacyDeepSet)


def _build(**overrides: Any) -> GraphConvolution:
    """Input 3, node layers [4], set layers [2], hidden 5, two GCN layers."""
    kwargs: dict[str, Any] = {
        "input_dim": 3,
        "node_layers": [4],
        "set_layers": [2],
        "hidden_channels": 5,
        "num_layers": 2,
    }
    kwargs.update(overrides)
    return GraphConvolution(**kwargs)


def _convs(model: GraphConvolution) -> list[GCNConv]:
    """The model's message-passing layers, typed."""
    layers = list(model.gcn_layers)
    convs = [layer for layer in layers if isinstance(layer, GCNConv)]
    assert len(convs) == len(layers)
    return convs


def _linears(model: GraphConvolution) -> list[nn.Linear]:
    """The model's post-readout set layers, typed."""
    layers = list(model.set_layers)
    linears = [layer for layer in layers if isinstance(layer, nn.Linear)]
    assert len(linears) == len(layers)
    return linears


def _widths(model: GraphConvolution) -> list[tuple[int, int]]:
    """``(in_features, out_features)`` of each set layer."""
    return [(lin.in_features, lin.out_features) for lin in _linears(model)]


STAR = torch.tensor([[0, 2], [1, 1]])
X3 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
W = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
B = torch.tensor([0.5, -0.5])


def _toy_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Five nodes in two sets, six directed edges, seeded features of width 3."""
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(5, 3, generator=gen)
    batch = torch.tensor([0, 0, 1, 1, 1])
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 2], [1, 0, 3, 4, 2, 4]])
    return x, batch, edge_index


def test_constructor_refuses_against_the_current_deep_set() -> None:
    """Finding: the shipped class cannot be constructed.

    graph_convolution.py line 33 passes ``input_dim=`` (and ``node_layers=``,
    ``set_layers=``) to ``DeepSet``, whose current keywords are ``in_channels``,
    ``hidden_channels``, ``out_channels``, ``num_node_layers``, ``num_set_layers``.
    Pinned until the wrapper is ported or retired (its only caller is under
    ``experiments/DEPRECATED_costanzo_smf_dmf_supervised``).
    """
    with pytest.raises(TypeError) as excinfo:
        _build()
    assert str(excinfo.value) == (
        "DeepSet.__init__() got an unexpected keyword argument 'input_dim'"
    )


@pytest.mark.usefixtures("legacy_deep_set")
def test_forwards_the_legacy_keywords_to_deep_set_and_drops_extras() -> None:
    """The encoder receives exactly these keywords; ``**kwargs`` go nowhere.

    ``set_layers`` is always ``[]`` (the wrapper builds its own set MLP) and the
    unknown ``unused=1`` is swallowed.
    """
    model = _build(
        dropout_prob=0.3,
        norm="layer",
        activation="gelu",
        skip_node=True,
        skip_set=True,
        unused=1,
    )
    assert isinstance(model.deepset, _LegacyDeepSet)
    assert model.deepset.received == {
        "input_dim": 3,
        "node_layers": [4],
        "set_layers": [],
        "dropout_prob": 0.3,
        "norm": "layer",
        "activation": "gelu",
        "skip_node": True,
        "skip_set": True,
    }


@pytest.mark.usefixtures("legacy_deep_set")
def test_layer_widths_and_parameter_count_follow_the_constructor_arithmetic() -> None:
    """``out_channels=6``: GCN 4 -> 5 -> 6, set Linear 6 -> 2; 25 + 36 + 14 = 75.

    Counts per the module docstring: GCN(4, 5) 5 * 5, GCN(5, 6) 6 * 6, Linear(6, 2)
    2 * 7. The stand-in encoder's Linear(3, 4) is excluded.
    """
    model = _build(out_channels=6)
    shapes = {
        name: tuple(p.shape)
        for name, p in model.named_parameters()
        if not name.startswith("deepset.")
    }
    assert shapes == {
        "gcn_layers.0.bias": (5,),
        "gcn_layers.0.lin.weight": (5, 4),
        "gcn_layers.1.bias": (6,),
        "gcn_layers.1.lin.weight": (6, 5),
        "set_layers.0.weight": (2, 6),
        "set_layers.0.bias": (2,),
    }
    assert sum(math.prod(s) for s in shapes.values()) == 75


@pytest.mark.usefixtures("legacy_deep_set")
def test_out_channels_none_keeps_hidden_width_and_empty_node_layers_use_input() -> None:
    """Without ``out_channels`` every GCN is hidden-wide; no node layers feed input_dim.

    ``node_layers=[]``, ``num_layers=3``: GCN 3 -> 5, 5 -> 5, 5 -> 5, set Linear 5 -> 2.
    """
    model = _build(node_layers=[], num_layers=3)
    widths = [(layer.in_channels, layer.out_channels) for layer in _convs(model)]
    assert widths == [(3, 5), (5, 5), (5, 5)]
    assert _widths(model) == [(5, 2)]


@pytest.mark.usefixtures("legacy_deep_set")
def test_set_layers_argument_is_mutated_in_place() -> None:
    """Finding: the caller's ``set_layers`` list gains the GCN width at index 0.

    graph_convolution.py line 59 runs ``set_layers.insert(0, in_dim)`` on the argument,
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


@pytest.mark.usefixtures("legacy_deep_set")
def test_one_gcn_layer_matches_the_symmetric_normalized_closed_form() -> None:
    """Hand-set weights on the directed star give the values in the module docstring.

    The in-neighbors of node 1 are weighted 1/sqrt(3), its self loop 1/3; a plain sum
    or mean aggregation would give different numbers.
    """
    model = _build(
        input_dim=2, node_layers=[], set_layers=[], hidden_channels=2, num_layers=1
    )
    (conv,) = _convs(model)
    with torch.no_grad():
        conv.lin.weight.copy_(W)
        conv.bias.copy_(B)
    x_node, x_set = model(X3, torch.tensor([0, 1, 1]), STAR)
    r3 = math.sqrt(3.0)
    node1 = torch.tensor([1 / 3 + 5 / r3 + 0.5, 1 / 3 + 2 / r3 - 0.5])
    expected = torch.stack([torch.tensor([1.5, -0.5]), node1, torch.tensor([4.5, 1.5])])
    torch.testing.assert_close(x_node, expected)
    torch.testing.assert_close(
        x_set, torch.stack([expected[0], expected[1] + expected[2]])
    )


@pytest.mark.usefixtures("legacy_deep_set")
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


@pytest.mark.usefixtures("legacy_deep_set")
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


@pytest.mark.usefixtures("legacy_deep_set")
def test_node_permutation_equivariance_and_set_invariance() -> None:
    """Relabeling nodes permutes node outputs and leaves the set outputs unchanged.

    ``perm`` maps new position k to old node ``perm[k]``; edges are relabeled with the
    inverse permutation and the batch vector is permuted with the features.
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


@pytest.mark.usefixtures("legacy_deep_set")
def test_gradient_reaches_every_parameter() -> None:
    """A loss on both outputs gives every parameter a finite, nonzero gradient."""
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
    assert len(names) == 8


@pytest.mark.usefixtures("legacy_deep_set")
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
