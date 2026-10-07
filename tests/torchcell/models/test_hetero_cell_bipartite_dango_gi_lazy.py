# tests/torchcell/models/test_hetero_cell_bipartite_dango_gi_lazy.py
# [[tests.torchcell.models.test_hetero_cell_bipartite_dango_gi_lazy]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_hetero_cell_bipartite_dango_gi_lazy.py
"""The 006 lazy ``GeneInteractionDango`` on a hand-built five-gene, two-graph batch.

Fixture. Five genes 0..4 and two gene graphs (``_multigraph`` from the eager test; the
model reads only its sorted keys ``["physical", "regulatory"]``). The wildtype
``cell_graph`` has ``gene.num_nodes = 5`` and edges physical 0->1, 1->2, 2->3, 3->4
(positions 0..3) and regulatory 4->0, 0->2, 1->3 (positions 0..2). A lazy sample
follows the ``LazySubgraphRepresentation`` output contract (graph_processor.py:1245-1259):
the FULL graph (``num_nodes = 5``, full ``edge_index``), ``pert_mask`` / ``mask`` over the
five genes, ``perturbation_indices`` (wildtype ids), and per edge type a boolean
``mask`` that is True iff neither endpoint is perturbed. Samples are collated with the
real ``lazy_collate_hetero`` (what ``LazyCollater`` runs) or, for the pre-#549/#571
history, PyG's ``Batch.from_data_list``. Genotypes used most: {1}, {0, 3}, {0, 2, 4}.

Kept edge positions, derived by hand: {1}: physical {2, 3}, regulatory {0, 1};
{0, 3}: physical {1}, regulatory {}; {0, 2, 4}: physical {}, regulatory {2}.

Two follow_batch lists matter (issue #596): ``FOLLOW_INERT = ["x", "x_pert"]`` (the
datamodule default; the preprocessed script, slurm 074/075/077, and the lazy script
before 53c257c22, slurm 062-065/069 as committed) and ``FOLLOW_LIVE`` which adds
``"perturbation_indices"`` (lazy script from 53c257c22, slurm 073, 076, 078-084).

Tiny configuration (``_lazy``): d = 8 hidden, 1 conv layer, GIN encoder with the default
2-layer MLP, "sum" aggregation, norm "layer", local predictor 2 heads x 1 attention
layer, gating. Parameter count by component:

* gene_embedding 5 * 8 = 40
* preprocessor Linear(8, 8) twice (72 + 72) + ONE shared PyG LayerNorm (8 + 8) = 160
* convs, per graph: MaskedGINConv eps 1 + Linear(8, 8) twice 144 + LayerNorm 16 = 161;
  two graphs 322 (proj is Identity because the GIN MLP ends at 8)
* gene_interaction_predictor: static Linear 72 + q/k/v/out 4 * 72 + beta 1 = 361, plus
  prediction Linear(8, 1) 9 = 370
* global_aggregator: gate Linear(8, 4) 36 + Linear(4, 1) 5 + transform Linear(8, 8) 72
  = 113
* global_interaction_predictor Linear(8, 8) 72 + Linear(8, 1) 9 = 81
* gate_mlp Linear(2, 8) 24 + Linear(8, 2) 18 = 42

Total 1128. The eager comparison uses the eager test's own four-gene fixture
(``_tiny``, ``_batch``, ``_cell_graph``) with the eager weights loaded into the lazy model.
"""

import math
import re
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import BatchNorm as PygBatchNorm
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.nn import LayerNorm as PygLayerNorm

from tests.torchcell.models.test_hetero_cell_bipartite_dango_gi import (
    HIDDEN,
    PHYS,
    REG,
    _edge_index,
    _multigraph,
)
from tests.torchcell.models.test_hetero_cell_bipartite_dango_gi import (
    N_GENES as EAGER_N_GENES,
)
from tests.torchcell.models.test_hetero_cell_bipartite_dango_gi import (
    WILDTYPE_EDGES as EAGER_EDGES,
)
from tests.torchcell.models.test_hetero_cell_bipartite_dango_gi import (
    _batch as _eager_batch,
)
from tests.torchcell.models.test_hetero_cell_bipartite_dango_gi import (
    _cell_graph as _eager_cell_graph,
)
from tests.torchcell.models.test_hetero_cell_bipartite_dango_gi import (
    _tiny as _eager_tiny,
)
from torchcell.data.graph_processor import LazySubgraphRepresentation
from torchcell.datamodules.lazy_collate import LazyCollater, lazy_collate_hetero
from torchcell.models.act import act_register
from torchcell.models.hetero_cell_bipartite_dango_gi_lazy import (
    AttentionalGraphAggregation,
    AttentionConvWrapper,
    DangoLikeHyperSAGNN,
    GeneInteractionDango,
    GeneInteractionPredictor,
    HeteroConvAggregator,
    PairwiseGraphAggregation,
    PreProcessor,
    SelfAttentionGraphAggregation,
    calculate_rolling_correlation,
    calculate_weight_l2_norm,
    create_conv_layer,
    get_norm_layer,
)
from torchcell.models.norm import norm_register
from torchcell.nn.masked_gin_conv import MaskedGINConv

N = 5
EDGES = {
    "physical": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "regulatory": [(4, 0), (0, 2), (1, 3)],
}
GENOTYPES = [[1], [0, 3], [0, 2, 4]]
FOLLOW_INERT = ["x", "x_pert"]
FOLLOW_LIVE = ["x", "x_pert", "perturbation_indices"]
ACT_NAMES = "['relu', 'gelu', 'sigmoid', 'leaky_relu', 'tanh']"
CONCAT = {"num_heads": 2, "num_attention_layers": 1, "combination_method": "concat"}


@pytest.fixture(autouse=True)
def _isolated_rng() -> Iterator[None]:
    """Run every test inside a forked CPU RNG so its manual seeds do not leak."""
    with torch.random.fork_rng(devices=[]):
        yield


def _cell_graph(
    n: int = N, edges: dict[str, list[tuple[int, int]]] = EDGES
) -> HeteroData:
    data = HeteroData()
    data["gene"].num_nodes = n
    data["gene"].x = torch.zeros(n, 1)
    for name, pairs in edges.items():
        data["gene", name, "gene"].edge_index = _edge_index(pairs)
    return data


def _lazy_sample(
    pert: list[int], n: int = N, edges: dict[str, list[tuple[int, int]]] = EDGES
) -> HeteroData:
    """Full graph plus masks; ``perturbation_indices`` keeps the order given."""
    data = HeteroData()
    data["gene"].num_nodes = n
    data["gene"].x = torch.zeros(n, 1)
    pert_mask = torch.zeros(n, dtype=torch.bool)
    pert_mask[pert] = True
    data["gene"].pert_mask = pert_mask
    data["gene"].mask = ~pert_mask
    data["gene"].perturbation_indices = torch.tensor(pert, dtype=torch.long)
    for name, pairs in edges.items():
        store = data["gene", name, "gene"]
        store.edge_index = _edge_index(pairs)
        store.num_edges = len(pairs)
        store.mask = torch.tensor(
            [s not in pert and t not in pert for s, t in pairs], dtype=torch.bool
        )
    return data


def _collate(
    perts: list[list[int]],
    follow: list[str],
    n: int = N,
    edges: dict[str, list[tuple[int, int]]] = EDGES,
    collater: str = "lazy",
) -> HeteroData:
    samples = [_lazy_sample(p, n, edges) for p in perts]
    if collater == "lazy":
        return lazy_collate_hetero(samples, follow)
    if collater == "LazyCollater":
        batch: HeteroData = LazyCollater(samples, follow_batch=follow)(samples)
        return batch
    out: HeteroData = Batch.from_data_list(samples, follow_batch=follow)
    return out


def _lazy(seed: int = 0, **overrides: Any) -> GeneInteractionDango:
    torch.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "gene_num": N,
        "hidden_channels": HIDDEN,
        "num_layers": 1,
        "gene_multigraph": _multigraph(),
        "dropout": 0.0,
        "gene_encoder_config": {
            "encoder_type": "gin",
            "graph_aggregation_method": "sum",
        },
        "local_predictor_config": {"num_heads": 2, "num_attention_layers": 1},
    }
    kwargs.update(overrides)
    return GeneInteractionDango(**kwargs)


def _predictor(model: GeneInteractionDango) -> GeneInteractionPredictor:
    predictor = model.gene_interaction_predictor
    assert predictor is not None
    return predictor


def _graph_layer_norm(z: torch.Tensor, norm: PygLayerNorm) -> torch.Tensor:
    """PyG LayerNorm, mode "graph", no batch: one mean and one std over ALL entries."""
    centered = z - z.mean()
    weight: torch.Tensor = norm.weight
    bias: torch.Tensor = norm.bias
    return centered / (centered.std(unbiased=False) + 1e-5) * weight + bias


# ---------------------------------------------------------------- small helpers


def test_norm_factory_returns_pyg_graph_norms_and_refuses_by_name() -> None:
    """``get_norm_layer`` maps through ``norm_register``: "layer" is PyG's LayerNorm in
    its default mode "graph" (eps 1e-5), "batch" PyG's BatchNorm. The eager model's
    factory returns ``nn.LayerNorm`` for "layer" instead; see the PreProcessor test.
    """
    layer = get_norm_layer(6, "layer")
    assert isinstance(layer, PygLayerNorm) and type(layer) is PygLayerNorm
    assert (layer.in_channels, layer.mode, layer.eps) == (6, "graph", 1e-5)
    assert type(get_norm_layer(6, "batch")) is PygBatchNorm
    with pytest.raises(
        ValueError,
        match=re.escape(
            "norm 'group' not found in norm_register. "
            f"Available: {list(norm_register.keys())}"
        ),
    ):
        get_norm_layer(6, "group")


def test_weight_l2_norm_and_rolling_correlation_closed_forms() -> None:
    """Linear(2, 1), W = [3, 4], b = 12: sqrt(9 + 16 + 144) = 13; freezing b leaves 5.
    Rolling r over x = [1, 2, 3, 4], y = [2, 4, 6, 5], window 3: [1, 0.5] (second
    window centered x [-1, 0, 1], y [-1, 1, 0], r = 1 / sqrt(2 * 2)); a constant
    window gives 0.0; a series shorter than the window gives [].
    """
    linear = nn.Linear(2, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[3.0, 4.0]]))
        linear.bias.copy_(torch.tensor([12.0]))
    assert calculate_weight_l2_norm(linear) == pytest.approx(13.0)
    linear.bias.requires_grad_(False)
    assert calculate_weight_l2_norm(linear) == pytest.approx(5.0)
    rolling = calculate_rolling_correlation([1, 2, 3, 4], [2, 4, 6, 5], window=3)
    assert rolling == pytest.approx([1.0, 0.5])
    assert calculate_rolling_correlation([1, 1, 1], [1, 2, 3], window=3) == [0.0]
    assert calculate_rolling_correlation([1, 2], [1, 2], window=3) == []


def test_preprocessor_normalizes_over_every_row_of_its_input() -> None:
    """Finding: the lazy ``PreProcessor`` builds its norm with ``get_norm_layer``
    (hetero_cell_bipartite_dango_gi_lazy.py:748), i.e. PyG ``LayerNorm`` in mode "graph"
    called without a batch vector, so ONE mean and ONE std are taken over all rows and
    channels. The eager model's PreProcessor uses per-node ``nn.LayerNorm``. Row 0 of
    pre(x) therefore changes when other rows are present. The single norm module is
    shared by both layers (Linear 48 + Linear 72 + one norm 16 = 136 parameters).

    Oracle: numpy, h = gelu(gLN(W x + b)) per layer with gLN(z) = (z - mean(z)) /
    (std_ddof0(z) + 1e-5), weight 1 and bias 0 at construction.
    Pinned until the PreProcessor normalizes per node as the eager model does.
    """
    torch.manual_seed(0)
    pre = PreProcessor(5, 8, num_layers=2, dropout=0.0, activation="gelu").eval()
    norms = [m for m in pre.mlp if isinstance(m, PygLayerNorm)]
    assert len(norms) == 2 and norms[0] is norms[1]
    assert sum(p.numel() for p in pre.parameters()) == 136
    assert pre.act is act_register["gelu"]
    x = torch.randn(3, 5)
    h = x.numpy().astype(np.float64)
    for linear in [pre.mlp[0], pre.mlp[4]]:
        assert isinstance(linear, nn.Linear)
        z = h @ linear.weight.detach().numpy().T + linear.bias.detach().numpy()
        z = (z - z.mean()) / (z.std() + 1e-5)
        h = 0.5 * z * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    torch.testing.assert_close(pre(x), torch.tensor(h, dtype=torch.float32))
    assert not torch.allclose(pre(x)[:1], pre(x[:1]), atol=1e-3)


def test_preprocessor_refuses_missing_and_unknown_activations_by_name() -> None:
    """``None`` and an unregistered name are refused with their full messages."""
    with pytest.raises(
        ValueError, match=re.escape("activation must be specified for PreProcessor")
    ):
        PreProcessor(5, 8, activation=None)  # type: ignore[arg-type, unused-ignore]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"activation 'silu' not found in act_register. Available: {ACT_NAMES}"
        ),
    ):
        PreProcessor(5, 8, activation="silu")


# ---------------------------------------------------------------- conv construction


def test_gin_layer_is_masked_with_trainable_eps_and_configured_depth() -> None:
    """``create_conv_layer("gin")`` builds a ``MaskedGINConv`` with eps a parameter
    (init 0). gin_num_layers 3, gin_hidden_dim 16: Linear(8, 16), Linear(16, 16),
    Linear(16, 8). gin_hidden_dim None (the 006 configs) falls back to out_channels.

    Finding (shared with the eager model): gin_num_layers 1 runs only the first branch
    (hetero_cell_bipartite_dango_gi_lazy.py:909-912), so the conv emits gin_hidden_dim
    16, not out_channels 8. Pinned until a one-layer MLP maps to out_channels.
    """
    deep = create_conv_layer(
        "gin", 8, 8, {"gin_hidden_dim": 16, "gin_num_layers": 3}, "relu"
    )
    assert isinstance(deep, MaskedGINConv)
    assert isinstance(deep.eps, nn.Parameter) and deep.eps.item() == 0.0
    assert isinstance(deep.nn, nn.Sequential)
    shapes = [
        (m.in_features, m.out_features) for m in deep.nn if isinstance(m, nn.Linear)
    ]
    assert shapes == [(8, 16), (16, 16), (16, 8)]
    default = create_conv_layer("gin", 8, 6, {"gin_hidden_dim": None}, "relu")
    assert isinstance(default, MaskedGINConv) and isinstance(default.nn, nn.Sequential)
    assert [
        (m.in_features, m.out_features) for m in default.nn if isinstance(m, nn.Linear)
    ] == [(8, 6), (6, 6)]
    shallow = create_conv_layer(
        "gin", 8, 8, {"gin_hidden_dim": 16, "gin_num_layers": 1}, "relu"
    )
    assert isinstance(shallow, MaskedGINConv)
    assert isinstance(shallow.nn, nn.Sequential)
    assert [type(m) for m in shallow.nn] == [nn.Linear, nn.ReLU, nn.Dropout]
    assert shallow(torch.randn(3, 8), _edge_index([(0, 1)])).shape == (3, 16)


def test_conv_factory_refusals_carry_their_full_messages() -> None:
    """gatv2 is not implemented for the lazy path; unknown encoders and activations are
    refused; the activation is checked before the encoder type.
    """
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "GATv2 not yet supported for lazy architecture - use GIN encoder. "
            "To implement: create MaskedGATv2Conv following MaskedGINConv pattern."
        ),
    ):
        create_conv_layer("gatv2", 8, 8, {}, "relu")
    with pytest.raises(ValueError, match=re.escape("Unknown encoder type: sage")):
        create_conv_layer("sage", 8, 8, {}, "relu")
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"activation 'silu' not found in act_register. Available: {ACT_NAMES}"
        ),
    ):
        create_conv_layer("gatv2", 8, 8, {}, "silu")


def test_wrapper_width_rules_norm_options_and_refusals() -> None:
    """A GIN MLP ending at 8 -> Identity proj; ending at 16 -> Linear(16, 8). GATv2Conv
    with 2 heads x 3 concatenated -> width 6 -> Linear(6, 8); GCNConv(8, 5) -> Linear(5,
    8). norm None -> no norm module; dropout 0 -> no dropout module. ``activation=None``
    is refused (unlike the eager wrapper, where None means Identity).
    """
    gin = MaskedGINConv(nn.Sequential(nn.Linear(8, 8)))
    assert isinstance(AttentionConvWrapper(gin, 8, activation="relu").proj, nn.Identity)
    wide = MaskedGINConv(nn.Sequential(nn.Linear(8, 16)))
    proj = AttentionConvWrapper(wide, 8, activation="relu").proj
    assert isinstance(proj, nn.Linear) and (proj.in_features, proj.out_features) == (
        16,
        8,
    )
    gat = AttentionConvWrapper(GATv2Conv(8, 3, heads=2), 8, activation="relu")
    assert isinstance(gat.proj, nn.Linear) and gat.proj.in_features == 6
    gcn = AttentionConvWrapper(
        GCNConv(8, 5), 8, norm=None, activation="tanh", dropout=0
    )
    assert isinstance(gcn.proj, nn.Linear) and gcn.proj.in_features == 5
    assert gcn.norm is None and gcn.dropout is None and type(gcn.act) is nn.Tanh
    with pytest.raises(
        ValueError,
        match=re.escape("activation must be specified for AttentionConvWrapper"),
    ):
        AttentionConvWrapper(gin, 8)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"activation 'silu' not found in act_register. Available: {ACT_NAMES}"
        ),
    ):
        AttentionConvWrapper(gin, 8, activation="silu")


def test_wrapper_around_a_gin_mlp_with_no_linear_raises_unbound_local() -> None:
    """Finding: when a GIN MLP is a Sequential with no ``nn.Linear`` the width loop
    (hetero_cell_bipartite_dango_gi_lazy.py:788-791) never assigns ``expected_dim`` and
    line 807 raises UnboundLocalError instead of a named refusal (the eager wrapper was
    given one under issue #540). Pinned until the lazy wrapper refuses by name.
    """
    conv = MaskedGINConv(nn.Sequential(nn.ReLU()))
    with pytest.raises(
        UnboundLocalError,
        match=re.escape(
            "cannot access local variable 'expected_dim' where it is not associated "
            "with a value"
        ),
    ):
        AttentionConvWrapper(conv, 8, activation="relu")


def test_wrapper_masks_messages_exactly_like_deleting_the_masked_edges() -> None:
    """eval: out = relu(gLN(conv(x, e, mask))), and a masked edge contributes nothing:
    the output equals the wrapper on ``e[:, mask]`` with no mask. With identity MLP
    and eps 0 the conv is x_i + sum_{kept j->i} x_j: edges 0->1 (kept), 1->2 (masked),
    3->2 (kept) give rows [x0, x0 + x1, x2 + x3, x3].
    """
    conv = MaskedGINConv(nn.Identity())
    wrapper = AttentionConvWrapper(conv, 3, norm="layer", activation="relu").eval()
    x = torch.arange(12.0).reshape(4, 3)
    edges = _edge_index([(0, 1), (1, 2), (3, 2)])
    mask = torch.tensor([True, False, True])
    expected_conv = torch.stack([x[0], x[0] + x[1], x[2] + x[3], x[3]])
    torch.testing.assert_close(conv(x, edges, edge_mask=mask), expected_conv)
    assert isinstance(wrapper.norm, PygLayerNorm)
    expected = torch.relu(_graph_layer_norm(expected_conv, wrapper.norm))
    torch.testing.assert_close(wrapper(x, edges, edge_mask=mask), expected)
    torch.testing.assert_close(
        wrapper(x, edges[:, mask]), wrapper(x, edges, edge_mask=mask)
    )


# ---------------------------------------------------------------- graph aggregation


def test_self_attention_aggregation_count_one_graph_closed_form_and_key_order() -> None:
    """4d^2 + 4d = 288 attention parameters at d = 8 plus G * d = 16 graph embeddings:
    304. With one graph each node attends to itself with weight 1, so the output is
    out_proj(v_proj(x + e_0)). Sorted graph names make insertion order irrelevant.
    Empty input gives (None, None).
    """
    torch.manual_seed(0)
    agg = SelfAttentionGraphAggregation(HIDDEN, num_graphs=2, num_heads=2).eval()
    assert sum(p.numel() for p in agg.parameters()) == 304
    x = torch.randn(3, HIDDEN)
    out, weights = agg({"physical": x})
    attn = agg.multihead_attn
    v = (x + agg.graph_embeddings[0]) @ attn.in_proj_weight[2 * HIDDEN :].t()
    v = v + attn.in_proj_bias[2 * HIDDEN :]
    torch.testing.assert_close(out, attn.out_proj(v))
    assert weights is not None and torch.equal(weights, torch.ones(3, 1, 1))
    a, b = torch.randn(4, HIDDEN), torch.randn(4, HIDDEN)
    first, w2 = agg({"physical": a, "regulatory": b})
    second, _ = agg({"regulatory": b, "physical": a})
    assert first is not None and second is not None and torch.equal(first, second)
    assert w2 is not None and w2.shape == (4, 2, 2)
    torch.testing.assert_close(w2.sum(-1), torch.ones(4, 2))
    assert agg({}) == (None, None)


def test_pairwise_aggregation_count_and_the_four_option_weighted_sum() -> None:
    """Bottleneck defaults to d // 2 = 4. Per pair: Linear(16, 4) 68 + Linear(4, 8) 40
    = 108, three pairs 324; scorer Linear(8, 4) 36 + Linear(4, 1) 5 = 41: 365. A third
    MLP layer adds Linear(4, 4) = 20 per pair (425). Options are [pp(a, a), pr(a, b),
    rr(b, b), identity (a + b) / 2] weighted by softmax(scorer(option)).
    """
    torch.manual_seed(0)
    agg = PairwiseGraphAggregation(HIDDEN, ["regulatory", "physical"]).eval()
    assert list(agg.interaction_mlps) == [
        "physical_physical",
        "physical_regulatory",
        "regulatory_regulatory",
    ]
    assert sum(p.numel() for p in agg.parameters()) == 365
    deeper = PairwiseGraphAggregation(HIDDEN, ["physical", "regulatory"], num_layers=3)
    assert sum(p.numel() for p in deeper.parameters()) == 425
    a, b = torch.randn(2, HIDDEN), torch.randn(2, HIDDEN)
    out, weights = agg({"physical": a, "regulatory": b})
    mlps = agg.interaction_mlps
    stacked = torch.stack(
        [
            mlps["physical_physical"](torch.cat([a, a], -1)),
            mlps["physical_regulatory"](torch.cat([a, b], -1)),
            mlps["regulatory_regulatory"](torch.cat([b, b], -1)),
            (a + b) / 2,
        ],
        dim=1,
    )
    expected_w = torch.softmax(agg.pair_scorer(stacked).squeeze(-1), dim=-1)
    assert weights is not None and weights.shape == (2, 4)
    torch.testing.assert_close(weights, expected_w)
    torch.testing.assert_close(out, (stacked * expected_w.unsqueeze(-1)).sum(1))


def test_pairwise_aggregation_norm_single_graph_and_fallbacks() -> None:
    """Norm "layer" applies graph-mode LayerNorm to the weighted sum (the 006 configs
    set ``aggregation_norm: "layer"``). One graph: options [pp(x, x), x]. No known graph
    name: (mean of inputs, None). Empty: (None, None). Unknown activation refused.
    """
    torch.manual_seed(0)
    agg = PairwiseGraphAggregation(HIDDEN, ["physical"], norm="layer").eval()
    assert isinstance(agg.norm, PygLayerNorm) and agg.norm.mode == "graph"
    x = torch.randn(3, HIDDEN)
    out, weights = agg({"physical": x})
    options = torch.stack(
        [agg.interaction_mlps["physical_physical"](torch.cat([x, x], -1)), x], dim=1
    )
    w = torch.softmax(agg.pair_scorer(options).squeeze(-1), dim=-1)
    raw = (options * w.unsqueeze(-1)).sum(1)
    assert weights is not None
    torch.testing.assert_close(weights, w)
    torch.testing.assert_close(out, _graph_layer_norm(raw, agg.norm))
    mean, none = agg({"x": torch.ones(2, HIDDEN), "y": 3 * torch.ones(2, HIDDEN)})
    assert mean is not None and torch.equal(mean, 2 * torch.ones(2, HIDDEN))
    assert none is None and agg({}) == (None, None)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"activation 'silu' not found in act_register. Available: {ACT_NAMES}"
        ),
    ):
        PairwiseGraphAggregation(HIDDEN, ["physical"], activation="silu")


class _MaskedScale(nn.Module):
    """Parameter-free conv returning c * x and recording the edge mask it receives."""

    def __init__(self, c: float) -> None:
        super().__init__()
        self.c = c
        self.seen: list[torch.Tensor | None] = []

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.seen.append(edge_mask)
        return self.c * x


def test_hetero_conv_sum_mean_skip_and_mask_routing() -> None:
    """Convs 2x (physical) and 3x (regulatory): sum 5x, mean 2.5x, no weights. Each
    conv receives the mask of ITS edge type, or None when the dict lacks it. A missing
    edge type is skipped (2x only); no edges at all gives ({}, None).
    """
    phys, reg = _MaskedScale(2.0), _MaskedScale(3.0)
    convs: dict[Any, nn.Module] = {PHYS: phys, REG: reg}
    x = {"gene": torch.arange(6.0).reshape(3, 2)}
    edges = {PHYS: _edge_index([(0, 1)]), REG: _edge_index([(1, 2), (2, 0)])}
    m_phys = torch.tensor([False])
    summed, attn = HeteroConvAggregator(convs, 2, "sum")(x, edges, {PHYS: m_phys})
    assert torch.equal(summed["gene"], 5 * x["gene"]) and attn is None
    assert phys.seen[-1] is m_phys and reg.seen[-1] is None
    mean, _ = HeteroConvAggregator(convs, 2, "mean")(x, edges)
    assert torch.equal(mean["gene"], 2.5 * x["gene"])
    only, _ = HeteroConvAggregator(convs, 2, "sum")(x, {PHYS: edges[PHYS]})
    assert torch.equal(only["gene"], 2 * x["gene"])
    assert HeteroConvAggregator(convs, 2, "sum")(x, {}) == ({}, None)
    with pytest.raises(ValueError, match=re.escape("Unknown aggregation method: max")):
        HeteroConvAggregator(convs, 2, "max")


def test_hetero_conv_learned_aggregators_return_their_weights() -> None:
    """cross_attention -> {"gene": [3, 2, 2]}; pairwise -> {"gene": [3, 4]} (three
    pairs plus identity); each output equals the aggregator applied to {rel: c * x}.
    """
    torch.manual_seed(0)
    x = {"gene": torch.randn(3, HIDDEN)}
    edges = {PHYS: _edge_index([(0, 1)]), REG: _edge_index([(1, 2)])}
    per_graph = {"physical": 2 * x["gene"], "regulatory": 3 * x["gene"]}
    for method, shape in [
        ("cross_attention", (3, 2, 2)),
        ("pairwise_interaction", (3, 4)),
    ]:
        convs: dict[Any, nn.Module] = {PHYS: _MaskedScale(2.0), REG: _MaskedScale(3.0)}
        layer = HeteroConvAggregator(convs, HIDDEN, method, {"num_heads": 2}).eval()
        out, attn = layer(x, edges)
        assert attn is not None and attn["gene"] is not None
        assert tuple(attn["gene"].shape) == shape
        assert layer.aggregator is not None
        expected, _ = layer.aggregator(per_graph)
        assert expected is not None
        torch.testing.assert_close(out["gene"], expected)


def test_attentional_pooling_is_a_per_group_softmax_weighted_sum() -> None:
    """pool[g] = sum_{i in g} softmax_g(gate(x_i)) transform(x_i); node order within a
    group does not matter; an unknown activation is refused.
    """
    torch.manual_seed(0)
    pool = AttentionalGraphAggregation(HIDDEN, 4, dropout=0.0).eval()
    x = torch.randn(5, HIDDEN)
    index = torch.tensor([0, 0, 1, 1, 1])
    out = pool(x, index, dim_size=2)
    gate = pool.gate_nn(x).squeeze(-1)
    values = pool.transform_nn(x)
    expected = torch.stack(
        [
            (torch.softmax(gate[index == g], 0).unsqueeze(-1) * values[index == g]).sum(
                0
            )
            for g in (0, 1)
        ]
    )
    torch.testing.assert_close(out, expected)
    perm = torch.tensor([1, 0, 4, 2, 3])
    torch.testing.assert_close(pool(x[perm], index[perm], dim_size=2), out)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"activation 'silu' not found in act_register. Available: {ACT_NAMES}"
        ),
    ):
        AttentionalGraphAggregation(HIDDEN, 4, activation="silu")


# ---------------------------------------------------------------- Dango local head


def test_hypersagnn_rezero_two_gene_closed_form_and_refusals() -> None:
    """Beta starts at 0.01; with 2 genes the masked diagonal leaves weight 1 on the
    other gene, so dynamic_i = x_i + beta * out_proj(v_proj(x_j)); static =
    relu(W x + b). One gene passes through unchanged. hidden 8 with 3 heads trips the
    divisibility assertion; an unknown activation is refused.
    """
    torch.manual_seed(0)
    sagnn = DangoLikeHyperSAGNN(HIDDEN, num_heads=2, num_layers=1, dropout=0.0)
    assert sagnn.beta_params[0].item() == pytest.approx(0.01)
    layer = sagnn.attention_layers[0]
    assert isinstance(layer, nn.ModuleDict)
    x = torch.randn(2, HIDDEN)
    static, dynamic = sagnn(x)
    other = layer["out_proj"](layer["v_proj"](x.flip(0)))
    torch.testing.assert_close(dynamic, x + sagnn.beta_params[0] * other)
    torch.testing.assert_close(static, torch.relu(sagnn.static_embedding[0](x)))
    one = torch.randn(1, HIDDEN)
    assert torch.equal(sagnn(one)[1], one)
    with pytest.raises(
        AssertionError, match=re.escape("hidden_dim 8 must be divisible by num_heads 3")
    ):
        DangoLikeHyperSAGNN(HIDDEN, num_heads=3)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"activation 'silu' not found in act_register. Available: {ACT_NAMES}"
        ),
    ):
        DangoLikeHyperSAGNN(HIDDEN, num_heads=2, activation="silu")


def test_hypersagnn_batched_groups_are_independent_and_singletons_pass_through() -> (
    None
):
    """Batch [0, 1, 1, 2, 2, 2] (the genotype sizes 1, 2, 3 of the fixture): group 0
    has one gene and keeps its input row; groups 1 and 2 equal the unbatched call on
    their rows (two layers); an all-singleton batch returns the input itself.
    """
    torch.manual_seed(0)
    sagnn = DangoLikeHyperSAGNN(HIDDEN, num_heads=2, num_layers=2, dropout=0.0)
    x = torch.randn(6, HIDDEN)
    _, dynamic = sagnn(x, torch.tensor([0, 1, 1, 2, 2, 2]))
    assert torch.equal(dynamic[0], x[0])
    torch.testing.assert_close(dynamic[1:3], sagnn(x[1:3])[1])
    torch.testing.assert_close(dynamic[3:], sagnn(x[3:])[1])
    assert torch.equal(sagnn(x[:3], torch.tensor([0, 1, 2]))[1], x[:3])


def test_interaction_predictor_scores_are_group_means_of_squared_differences() -> None:
    """score_b = mean_{i in b} (w . (dyn_i - static_i)^2 + c); an empty group id (1 in
    [0, 0, 2, 2]) scores 0; with no batch vector the one score has shape [1, 1].
    """
    torch.manual_seed(0)
    predictor = GeneInteractionPredictor(HIDDEN, num_heads=2, num_layers=1, dropout=0.0)
    x = torch.randn(4, HIDDEN)
    batch = torch.tensor([0, 0, 2, 2])
    scores = predictor(x, batch)
    static, dynamic = predictor.hyper_sagnn(x, batch)
    gene = predictor.prediction_layer((dynamic - static) ** 2).squeeze(-1)
    expected = torch.stack([gene[:2].mean(), torch.tensor(0.0), gene[2:].mean()])
    torch.testing.assert_close(scores, expected.unsqueeze(-1))
    single = predictor(x[:2])
    assert single.shape == (1, 1)
    torch.testing.assert_close(single[0, 0], scores[0, 0])


# ---------------------------------------------------------------- model build


def test_tiny_model_parameter_count_matches_the_hand_derivation() -> None:
    """Component counts from the module docstring, total 1128; concat drops the gate
    (1086); ``use_local_predictor: false`` (006 configs 078-080) drops the predictor and
    the gate (1128 - 370 - 42 = 716).
    """
    model = _lazy()
    assert model.graph_names == ["physical", "regulatory"]
    assert model.num_parameters == {
        "gene_embedding": 40,
        "preprocessor": 160,
        "convs": 322,
        "gene_interaction_predictor": 370,
        "global_aggregator": 113,
        "global_interaction_predictor": 81,
        "gate_mlp": 42,
        "total": 1128,
    }
    assert model.num_parameters["total"] == sum(p.numel() for p in model.parameters())
    concat = _lazy(local_predictor_config=CONCAT)
    assert concat.gate_mlp is None and concat.num_parameters["total"] == 1086
    off = _lazy(local_predictor_config={"use_local_predictor": False})
    assert off.gene_interaction_predictor is None and off.gate_mlp is None
    assert off.num_parameters["total"] == 716
    assert "gene_interaction_predictor" not in off.num_parameters


def test_shipped_pairwise_encoder_flags_reach_the_aggregator() -> None:
    """The 006 encoder block (pairwise_interaction, pairwise_hidden_dim, aggregation_norm
    "layer", model activation gelu): each conv layer gets pair MLPs Linear(16, 4) GELU
    Linear(4, 8) (108 each, 324), scorer 41, LayerNorm 16 = 381 on top of 322: convs
    703, total 1509.

    The model's dropout and activation REPLACE the aggregation config's (lines 1026-1029):
    a config dropout of 0.3 with model dropout 0.0 builds attention dropout 0.0, and a
    config activation "relu" builds GELU. The eager model lets the config's dropout win.
    The 006 configs set both dropouts to 0.0, so they are unaffected.
    """
    model = _lazy(
        activation="gelu",
        gene_encoder_config={
            "encoder_type": "gin",
            "graph_aggregation_method": "pairwise_interaction",
            "graph_aggregation_config": {
                "aggregation_norm": "layer",
                "pairwise_hidden_dim": 4,
                "pairwise_num_layers": 2,
                "activation": "relu",
            },
        },
    )
    assert model.num_parameters["convs"] == 703
    assert model.num_parameters["total"] == 1509
    layer = model.convs[0]
    assert isinstance(layer, HeteroConvAggregator)
    agg = layer.aggregator
    assert isinstance(agg, PairwiseGraphAggregation)
    assert isinstance(agg.norm, PygLayerNorm)
    pair_mlp = agg.interaction_mlps["physical_regulatory"]
    assert isinstance(pair_mlp, nn.Sequential)
    assert [type(m) for m in pair_mlp] == [nn.Linear, nn.GELU, nn.Dropout, nn.Linear]
    cross = _lazy(
        gene_encoder_config={
            "encoder_type": "gin",
            "graph_aggregation_method": "cross_attention",
            "graph_aggregation_config": {"dropout": 0.3, "num_heads": 2},
        }
    )
    cross_layer = cross.convs[0]
    assert isinstance(cross_layer, HeteroConvAggregator)
    assert isinstance(cross_layer.aggregator, SelfAttentionGraphAggregation)
    assert cross_layer.aggregator.multihead_attn.dropout == 0.0
    assert cross_layer.aggregator.multihead_attn.num_heads == 2


def test_default_encoder_is_gatv2_and_is_refused_at_construction() -> None:
    """``encoder_type`` defaults to "gatv2" (line 999), which the lazy factory refuses,
    so a config without an explicit encoder cannot build the lazy model.
    """
    with pytest.raises(NotImplementedError, match=re.escape("GATv2 not yet supported")):
        _lazy(gene_encoder_config={})


def test_init_zeroes_every_linear_bias_and_sets_layernorm_to_identity() -> None:
    """``_init_weights``: every nn.Linear bias is 0 and every weight is Kaiming normal
    with mode fan_out, std = sqrt(2 / out_features). PyG LayerNorm is not an
    nn.LayerNorm, so it keeps its own (1, 0) reset.

    Weight check on a seeded d = 256 model: the non-square gate Linear(256, 128) of the
    global aggregator has 32768 entries, so fan_out gives std sqrt(2 / 128) = 0.125,
    fan_in would give sqrt(2 / 256) = 0.0884 and PyTorch's default init 1 / sqrt(3 *
    256) = 0.0361; the sample std of 32768 normals has relative SE 1 / sqrt(2 n) =
    0.39%, so a 3% band separates them. The square preprocessor Linear(256, 256) gives
    sqrt(2 / 256) = 0.0884 (65536 entries).
    """
    wide = _lazy(hidden_channels=256)
    gate = wide.global_aggregator.gate_nn[0]
    square = wide.preprocessor.mlp[0]
    assert isinstance(gate, nn.Linear) and isinstance(square, nn.Linear)
    assert (gate.in_features, gate.out_features) == (256, 128)
    for linear, expected_std in [
        (gate, math.sqrt(2 / 128)),
        (square, math.sqrt(2 / 256)),
    ]:
        w = linear.weight.detach().double()
        assert abs(w.std().item() / expected_std - 1.0) < 0.03
        assert abs(w.mean().item()) < 0.01
    model = _lazy()
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 19
    assert all(m.bias is not None and not m.bias.any() for m in linears)
    norms = [m for m in model.modules() if isinstance(m, PygLayerNorm)]
    assert len(norms) == 3
    for norm in norms:
        assert torch.equal(norm.weight, torch.ones(HIDDEN))
        assert torch.equal(norm.bias, torch.zeros(HIDDEN))


# ---------------------------------------------------------------- lazy masks


def test_fixture_masks_match_the_lazy_producer_and_the_hand_index_sets() -> None:
    """``LazySubgraphRepresentation._process_gene_interactions`` on the five-gene graph
    produces the same per-genotype masks as the fixture, and the kept positions are the
    hand-derived sets from the module docstring.
    """
    cell_graph = _cell_graph()
    expected_kept = {
        (1,): {"physical": [2, 3], "regulatory": [0, 1]},
        (0, 3): {"physical": [1], "regulatory": []},
        (0, 2, 4): {"physical": [], "regulatory": [2]},
    }
    processor = LazySubgraphRepresentation()
    for pert in GENOTYPES:
        produced = HeteroData()
        processor._process_gene_interactions(
            produced, cell_graph, {"remove_subset": torch.tensor(pert)}
        )
        sample = _lazy_sample(pert)
        for name in ["physical", "regulatory"]:
            et = ("gene", name, "gene")
            assert torch.equal(produced[et].mask, sample[et].mask)
            kept = sample[et].mask.nonzero().flatten().tolist()
            assert kept == expected_kept[tuple(pert)][name]


def test_collated_batch_offsets_edges_and_concatenates_masks() -> None:
    """lazy_collate offsets each sample's edges by 5 genes and concatenates masks;
    ``FOLLOW_LIVE`` adds ``perturbation_indices_ptr`` [0, 1, 3, 6] while
    ``FOLLOW_INERT`` adds no ``perturbation_indices_*`` key (``x_pert`` is absent from
    lazy samples, so only ``x_batch`` / ``x_ptr`` appear). perturbation_indices are
    concatenated WITHOUT offset (wildtype ids): [1, 0, 3, 0, 2, 4].
    """
    live = _collate(GENOTYPES, FOLLOW_LIVE)
    inert = _collate(GENOTYPES, FOLLOW_INERT)
    assert live[PHYS].edge_index.tolist() == [
        [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13],
        [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14],
    ]
    assert live[PHYS].mask.tolist() == [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    assert live[REG].mask.tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 1]
    assert live["gene"].perturbation_indices.tolist() == [1, 0, 3, 0, 2, 4]
    assert live["gene"].perturbation_indices_ptr.tolist() == [0, 1, 3, 6]
    inert_keys = set(inert["gene"].keys())
    assert {"x_batch", "x_ptr"} <= inert_keys
    assert (
        not {
            "perturbation_indices_ptr",
            "perturbation_indices_batch",
            "x_pert_batch",
            "x_pert_ptr",
        }
        & inert_keys
    )


def test_forward_hands_the_convs_all_true_wildtype_masks_and_the_batch_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spy on ``HeteroConvAggregator.forward`` records the mask dict per call: the
    wildtype cell graph has no ``mask`` so it gets all-True masks of lengths 4 and 3
    (lines 1197-1202); the batch passes its concatenated masks unchanged.
    """
    seen: list[dict[Any, torch.Tensor]] = []
    original = HeteroConvAggregator.forward

    def spy(
        self: HeteroConvAggregator,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[Any, torch.Tensor],
        edge_mask_dict: dict[Any, torch.Tensor] | None = None,
    ) -> Any:
        assert edge_mask_dict is not None
        seen.append({k: v.clone() for k, v in edge_mask_dict.items()})
        return original(self, x_dict, edge_index_dict, edge_mask_dict)

    monkeypatch.setattr(HeteroConvAggregator, "forward", spy)
    model = _lazy().eval()
    with torch.no_grad():
        model(_cell_graph(), _collate(GENOTYPES, FOLLOW_LIVE))
    assert len(seen) == 2
    assert seen[0][PHYS].tolist() == [True] * 4
    assert seen[0][REG].tolist() == [True] * 3
    assert seen[1][PHYS].tolist() == [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    assert seen[1][REG].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 1]


def test_masking_equals_deleting_the_masked_edges_but_keeps_the_deleted_genes() -> None:
    """The batched node embeddings with masks equal those of the same batch whose masked
    edges were physically removed (all-True masks), under the shipped norm "layer".
    Deleted genes stay as rows: 15 rows for 3 genotypes x 5 genes.
    """
    model = _lazy().eval()
    masked = _collate(GENOTYPES, FOLLOW_LIVE)
    filtered = _collate(GENOTYPES, FOLLOW_LIVE)
    for et in [PHYS, REG]:
        keep = filtered[et].mask
        filtered[et].edge_index = filtered[et].edge_index[:, keep]
        filtered[et].mask = torch.ones(int(keep.sum()), dtype=torch.bool)
    with torch.no_grad():
        z_masked = model.forward_single(masked)
        z_filtered = model.forward_single(filtered)
    assert z_masked.shape == (15, HIDDEN)
    torch.testing.assert_close(z_masked, z_filtered)


def test_train_batch_norm_statistics_include_the_deleted_rows() -> None:
    """Finding: under norm "batch" in train mode the conv-wrapper BatchNorm updates its
    running statistics from every row it sees, and the lazy batch keeps the deleted
    genes as rows: 15 rows for the three genotypes, of which 6 are deleted genes.

    A forward hook records the wrapper norm's two inputs (wildtype 5 rows, then the
    batch 15 rows). With momentum 0.1 from (mean 0, var 1):
    mean = 0.9 * (0.1 * m_wt) + 0.1 * m_batch and
    var = 0.9 * (0.9 + 0.1 * v_wt) + 0.1 * v_batch (unbiased variances), where m_batch
    and v_batch are over ALL 15 rows (numpy float64 oracle). Using only the 9 kept rows
    gives a different running mean. ``num_batches_tracked`` is 2.
    Pinned until normalization excludes deleted genes.
    """
    model = _lazy(norm="batch").train()
    layer = model.convs[0]
    assert isinstance(layer, HeteroConvAggregator)
    wrapper = layer.convs[str(PHYS)]
    assert isinstance(wrapper, AttentionConvWrapper)
    norm = wrapper.norm
    assert isinstance(norm, PygBatchNorm)
    seen: list[np.ndarray[Any, np.dtype[np.float64]]] = []

    def hook(
        module: nn.Module, args: tuple[torch.Tensor, ...], out: torch.Tensor
    ) -> None:
        seen.append(args[0].detach().double().numpy().copy())

    handle = norm.register_forward_hook(hook)
    batch = _collate(GENOTYPES, FOLLOW_LIVE)
    model(_cell_graph(), batch)
    handle.remove()
    assert [a.shape for a in seen] == [(5, HIDDEN), (15, HIDDEN)]
    wt, rows = seen
    kept = (~batch["gene"].pert_mask).numpy()
    assert int(kept.sum()) == 9
    mean = 0.9 * (0.1 * wt.mean(0)) + 0.1 * rows.mean(0)
    var = 0.9 * (0.9 + 0.1 * wt.var(0, ddof=1)) + 0.1 * rows.var(0, ddof=1)
    kept_mean = 0.9 * (0.1 * wt.mean(0)) + 0.1 * rows[kept].mean(0)
    inner = norm.module
    torch.testing.assert_close(inner.running_mean.double(), torch.from_numpy(mean))
    torch.testing.assert_close(inner.running_var.double(), torch.from_numpy(var))
    assert np.abs(mean - kept_mean).max() > 1e-3
    assert inner.num_batches_tracked.item() == 2


def test_tiling_misaligns_a_sample_whose_node_count_is_not_gene_num() -> None:
    """Finding: the batched path tiles the whole embedding table once per sample
    (``expand(batch_size)``, lines 1154-1155) and never checks that each sample has
    ``gene_num`` nodes. Samples of 4 and 6 nodes (gene_num 5, 10 rows in total) run
    without error, but batch row r receives gene r mod 5: the preprocessor input is
    embedding rows [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], while the sample-local node ids are
    [0, 1, 2, 3, 0, 1, 2, 3, 4, 5]; the second sample's node 0 gets gene 4's
    embedding. The 006 producers always emit the full gene count, so the runs do not
    reach it. Pinned until the tiling is checked against ``gene.ptr``.
    """
    small = {"physical": [(0, 1), (2, 3)], "regulatory": [(1, 2)]}
    large = {"physical": [(0, 1), (4, 5)], "regulatory": [(1, 2)]}
    batch = lazy_collate_hetero(
        [_lazy_sample([1], 4, small), _lazy_sample([0], 6, large)], FOLLOW_LIVE
    )
    assert batch["gene"].ptr.tolist() == [0, 4, 10]
    model = _lazy().eval()
    inputs: list[torch.Tensor] = []

    def hook(
        module: nn.Module, args: tuple[torch.Tensor, ...], out: torch.Tensor
    ) -> None:
        inputs.append(args[0].detach().clone())

    handle = model.preprocessor.register_forward_hook(hook)
    with torch.no_grad():
        pred, _ = model(_cell_graph(), batch)
    handle.remove()
    assert pred.shape == (2, 1) and torch.isfinite(pred).all()
    table = model.gene_embedding.weight.detach()
    assert torch.equal(inputs[1], table[[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]])
    local_ids = (torch.arange(10) - batch["gene"].ptr[batch["gene"].batch]).tolist()
    assert local_ids == [0, 1, 2, 3, 0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------- lazy vs eager


def _eager_to_lazy_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Eager nn.BatchNorm1d in the preprocessor -> PyG BatchNorm (wraps ``.module``)."""
    out = {}
    for key, value in state.items():
        parts = key.split(".")
        if key.startswith("preprocessor.mlp.") and parts[2] in ("1", "5"):
            key = ".".join([*parts[:3], "module", *parts[3:]])
        out[key] = value
    return out


def _fill_seeded(module: nn.Module, seed: int) -> None:
    """Every parameter normal(0, 0.5); BatchNorm running mean normal(0, 0.3) and
    running variance uniform(0.5, 2), from one numpy generator (as in the decoder test).
    """
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for _, p in module.named_parameters():
            p.copy_(torch.from_numpy(rng.normal(0.0, 0.5, size=tuple(p.shape))))
        for name, buf in module.named_buffers():
            if name.endswith("running_mean"):
                buf.copy_(torch.from_numpy(rng.normal(0.0, 0.3, size=tuple(buf.shape))))
            elif name.endswith("running_var"):
                buf.copy_(
                    torch.from_numpy(rng.uniform(0.5, 2.0, size=tuple(buf.shape)))
                )


def test_lazy_equals_eager_exactly_under_eval_batch_norm() -> None:
    """With norm "batch" in eval (running statistics, a per-node affine map) and the
    eager weights loaded (strict), the lazy model on lazy samples reproduces the eager
    model on relabeled subgraphs: z_w, z_i, pert_gene_embs, local, global and the
    prediction agree to 1e-6, for {0, 1}, {2, 3} and for {0}, {1, 2}, {3}. The masked
    full-graph path is therefore the subgraph path whenever normalization is per node.

    Every eager parameter (Linear biases, BN weight and bias, ReZero beta, GIN eps
    included) and every BN running mean and variance is first filled with distinct
    seeded values, so each BN is a distinct affine map; the transfer is checked key by
    key (every lazy state-dict entry equals its mapped eager entry, and the key sets
    match). Without the fill every BN is x / sqrt(1 + 1e-5) and skipping one, or
    swapping the two edge types' norms, moved the outputs by at most 1e-6.
    """
    eager = _eager_tiny(norm="batch").eval()
    _fill_seeded(eager, seed=7)
    lazy = _lazy(gene_num=EAGER_N_GENES, norm="batch").eval()
    mapped = _eager_to_lazy_state(eager.state_dict())
    lazy.load_state_dict(mapped, strict=True)
    lazy_state = lazy.state_dict()
    assert set(lazy_state) == set(mapped)
    for key, value in lazy_state.items():
        assert torch.equal(value, mapped[key]), key
    eager_state = eager.state_dict()
    phys_var = eager_state[f"convs.0.convs.{PHYS}.norm.module.running_var"]
    reg_var = eager_state[f"convs.0.convs.{REG}.norm.module.running_var"]
    assert not torch.allclose(phys_var, reg_var)
    cell_graph = _eager_cell_graph()
    for perts in ([[0, 1], [2, 3]], [[0], [1, 2], [3]]):
        lazy_batch = _collate(perts, FOLLOW_LIVE, EAGER_N_GENES, EAGER_EDGES)
        with torch.no_grad():
            pred_e, out_e = eager(cell_graph, _eager_batch(perts))
            pred_l, out_l = lazy(cell_graph, lazy_batch)
        torch.testing.assert_close(pred_l, pred_e, atol=1e-6, rtol=0.0)
        for key in [
            "z_w",
            "z_i",
            "z_p",
            "pert_gene_embs",
            "local_interaction",
            "global_interaction",
            "gate_weights",
        ]:
            torch.testing.assert_close(out_l[key], out_e[key], atol=1e-6, rtol=0.0)


def test_lazy_differs_from_eager_under_the_shipped_layer_norm() -> None:
    """Finding: under norm "layer" the same weights (strict load, identical keys) give
    different embeddings: the lazy PreProcessor normalizes over all genes at once
    (graph mode, line 748), while the eager PreProcessor is exactly per-node
    ``nn.LayerNorm`` (checked below against the closed form relu(LN_row(W x + b)) with
    LN_row(z) = (z - mean_row) / sqrt(var_row + 1e-5) * w + b), so even the wildtype z_w
    differs; the perturbed z_i also differs.

    Reach: the graph-mode PreProcessor norm entered the lazy model at 8b68e55d5, so it
    applies to slurm 069 and 073 onward (all ``norms: "layer"``); 062-065 ran the
    4760f653c model, whose ``get_norm_layer`` built per-node ``nn.LayerNorm``.
    Pinned until the lazy norms match the eager ones.
    """
    eager = _eager_tiny(norm="layer").eval()
    lazy = _lazy(gene_num=EAGER_N_GENES, norm="layer").eval()
    lazy.load_state_dict(eager.state_dict(), strict=True)
    perts = [[0, 1], [2, 3]]
    with torch.no_grad():
        _, out_e = eager(_eager_cell_graph(), _eager_batch(perts))
        _, out_l = lazy(
            _eager_cell_graph(),
            _collate(perts, FOLLOW_LIVE, EAGER_N_GENES, EAGER_EDGES),
        )
        x = lazy.gene_embedding.weight
        pre_e, pre_l = eager.preprocessor(x), lazy.preprocessor(x)
    assert not torch.allclose(out_l["z_w"], out_e["z_w"], atol=1e-3)
    assert not torch.allclose(out_l["z_i"], out_e["z_i"], atol=1e-3)
    assert not torch.allclose(pre_l, pre_e, atol=1e-3)
    e_lin1, e_norm, e_lin2 = (
        eager.preprocessor.mlp[0],
        eager.preprocessor.mlp[1],
        eager.preprocessor.mlp[4],
    )
    assert isinstance(e_norm, nn.LayerNorm) and e_norm is eager.preprocessor.mlp[5]
    assert isinstance(e_lin1, nn.Linear) and isinstance(e_lin2, nn.Linear)

    def row_norm(z: torch.Tensor) -> torch.Tensor:
        centered = z - z.mean(-1, keepdim=True)
        var = (centered**2).mean(-1, keepdim=True)
        return centered / torch.sqrt(var + 1e-5) * e_norm.weight + e_norm.bias

    with torch.no_grad():
        per_node = torch.relu(row_norm(e_lin2(torch.relu(row_norm(e_lin1(x))))))
        torch.testing.assert_close(pre_e, per_node)
    lin = lazy.preprocessor.mlp[0]
    norm = lazy.preprocessor.mlp[1]
    assert isinstance(lin, nn.Linear) and isinstance(norm, PygLayerNorm)
    with torch.no_grad():
        first = torch.relu(_graph_layer_norm(lin(x), norm))
        torch.testing.assert_close(lazy.preprocessor.mlp[:4](x), first)


def test_deleted_gene_embedding_reaches_kept_genes_through_layer_norm() -> None:
    """Finding: in the lazy path the deleted gene stays a row and every edge touching it
    is masked, yet under norm "layer" its learned embedding still changes the kept
    genes' pooled z_i, because graph-mode LayerNorm (PreProcessor line 748 and
    AttentionConvWrapper line 853) takes its mean and std over all rows including the
    deleted ones. Adding 1.0 to gene 0's embedding moves z_i of genotype {0, 1} (max
    abs change > 0.1); under eval batch norm the change is exactly 0.

    Reach: every lazy run with ``norms: "layer"``. The conv-wrapper norm was PyG
    graph-mode LayerNorm already at 4760f653c (slurm 062-065), so the leak exists
    there too, through the wrapper alone (0.074 on this fixture with the 4760f653c
    file, 0.575 with 8b68e55d5 for slurm 069).
    Pinned until normalization excludes deleted genes (or is per node).
    """
    batch = _collate([[0, 1]], FOLLOW_LIVE)
    changes = {}
    for norm in ["layer", "batch"]:
        model = _lazy(norm=norm).eval()
        with torch.no_grad():
            _, before = model(_cell_graph(), batch)
            model.gene_embedding.weight[0] += 1.0
            _, after = model(_cell_graph(), batch)
        changes[norm] = (after["z_i"] - before["z_i"]).abs().max().item()
    assert changes["layer"] > 0.1
    assert changes["batch"] == 0.0


def test_eval_prediction_of_a_genotype_depends_on_its_batch_mates_under_layer_norm() -> (
    None
):
    """Finding: with norm "layer" the conv-wrapper LayerNorm is called without a batch
    vector (line 853), so its statistics span all genotypes in the batch: each genotype's
    eval prediction batched with the other two differs from the genotype alone. Under
    eval batch norm every batched row equals the genotype run alone (1e-6).

    Reach: every lazy run with ``norms: "layer"``, including 062-065 (the 4760f653c
    wrapper norm is graph-mode too; max batched-vs-alone difference 0.086 on this
    fixture with that file, 0.106 with 8b68e55d5).
    Pinned until the wrapper norm is given the batch vector.
    """
    for norm in ["layer", "batch"]:
        model = _lazy(norm=norm, local_predictor_config=CONCAT).eval()
        with torch.no_grad():
            batched, _ = model(_cell_graph(), _collate(GENOTYPES, FOLLOW_LIVE))
            alone = torch.cat(
                [model(_cell_graph(), _collate([g], FOLLOW_LIVE))[0] for g in GENOTYPES]
            )
        if norm == "batch":
            torch.testing.assert_close(batched, alone, atol=1e-6, rtol=0.0)
        else:
            assert not torch.allclose(batched, alone, atol=1e-3)


# ---------------------------------------------------------------- issue #596


@pytest.mark.parametrize("collater", ["LazyCollater", "pyg"])
def test_local_predictor_is_exactly_zero_without_perturbation_indices_in_follow_batch(
    collater: str,
) -> None:
    """Finding (issue #596): with ``follow_batch = ["x", "x_pert"]`` the batch has no
    ``perturbation_indices_ptr`` / ``_batch``, so ``batch_assign`` is None (line
    1316-1321), the local predictor pools the six perturbed genes of all three
    genotypes into ONE [1, 1] score, and because 1 != batch size 3 the expansion
    (line 1351-1364) allocates zeros and skips the scatter at line 1356
    (``if batch_assign is not None``). The local term is exactly zeros(3, 1) and, with
    combination "concat", the prediction is exactly 0.5 * global. The global term is
    unchanged. With "perturbation_indices" followed, local = predictor(z_w[[1, 0, 3, 0,
    2, 4]], [0, 1, 1, 2, 2, 2]) and every batched prediction equals that genotype run
    alone (eval batch norm, so no cross-sample statistics). A batch of ONE genotype
    keeps its nonzero local score under either list.

    The PyG ``Batch.from_data_list`` collation (what ran before PRs #549/#571) and
    ``LazyCollater(dataset, follow_batch=...)`` (what both 006 lazy scripts build) give
    the same tensors.

    Reach (all seven configs use ``combination_method: "concat"``): exactly
    0.5 * global for every genotype in slurm 074, 075, 077 (preprocessed script,
    hetero_cell_bipartite_dango_gi_lazy_preprocessed.py:366, batch 28, 28, 24; model
    at 53c257c22, which has this expansion). Slurm 062-065 (model at 4760f653c) and
    069 (model at 8b68e55d5), as committed, ran an older expansion that wrote local row
    i to ``batch_assign[i] if not None else 0``: row 0 of each batch got 0.5 * global +
    0.5 * (the batch-pooled local score) and the other rows 0.5 * global (the audit ran
    both historical files: local [-2.545, 0, 0] and [-0.142, 0, 0] on this fixture).
    Issue #596 names slurm/config numbers, not W&B run ids.
    Pinned until the scripts follow perturbation_indices or the model refuses a
    multi-genotype batch without an assignment.
    """
    model = _lazy(norm="batch", local_predictor_config=CONCAT).eval()
    cell_graph = _cell_graph()
    live = _collate(GENOTYPES, FOLLOW_LIVE, collater=collater)
    inert = _collate(GENOTYPES, FOLLOW_INERT, collater=collater)
    assert not hasattr(inert["gene"], "perturbation_indices_ptr")
    assert not hasattr(inert["gene"], "perturbation_indices_batch")
    predictor = _predictor(model)
    with torch.no_grad():
        pred_live, out_live = model(cell_graph, live)
        pred_inert, out_inert = model(cell_graph, inert)
        z_w = model.forward_single(cell_graph)
        pert_embs = z_w[[1, 0, 3, 0, 2, 4]]
        per_genotype = predictor(pert_embs, torch.tensor([0, 1, 1, 2, 2, 2]))
        pooled = predictor(pert_embs, None)
        alone = [model(cell_graph, _collate([g], FOLLOW_LIVE))[1] for g in GENOTYPES]
        alone_inert = model(cell_graph, _collate([[0, 3]], FOLLOW_INERT))[1]
    # the inert list: exact zeros, prediction exactly half the global term
    assert torch.equal(out_inert["local_interaction"], torch.zeros(3, 1))
    assert torch.equal(pred_inert, 0.5 * out_inert["global_interaction"])
    assert torch.equal(out_inert["global_interaction"], out_live["global_interaction"])
    assert pooled.shape == (1, 1) and pooled.abs().item() > 0.0
    # the live list: per-genotype local scores, each equal to the genotype alone
    torch.testing.assert_close(out_live["pert_gene_embs"], pert_embs)
    torch.testing.assert_close(out_live["local_interaction"], per_genotype)
    assert (out_live["local_interaction"] != 0).all()
    torch.testing.assert_close(
        pred_live,
        0.5 * out_live["global_interaction"] + 0.5 * out_live["local_interaction"],
    )
    for row, single in enumerate(alone):
        torch.testing.assert_close(
            out_live["local_interaction"][row], single["local_interaction"][0]
        )
        torch.testing.assert_close(
            pred_live[row], single["gene_interaction"][0], atol=1e-6, rtol=0.0
        )
    # a batch of one is not affected by the follow list
    torch.testing.assert_close(
        alone_inert["local_interaction"], out_live["local_interaction"][1:2]
    )


def test_local_predictor_gets_no_gradient_without_perturbation_indices() -> None:
    """Finding (issue #596): because the inert-list local term is a fresh zeros tensor
    (line 1353-1364), backward from the prediction leaves every one of the local
    predictor's 13 parameter tensors (370 values: static 2, q/k/v/out 8,
    beta 1, prediction 2) with grad None, and every other
    parameter with a gradient; with "perturbation_indices" followed, no parameter is
    missed. (The trainer's ``_ensure_no_unused_params_loss`` then adds 0 * param, so
    these parameters see a zero gradient.) That holds for slurm 074, 075, 077 (model at
    53c257c22); under the older expansion of 062-065 and 069, batch row 0 carried the
    pooled local score and therefore a gradient.
    Pinned until the scripts follow perturbation_indices.
    """
    expected_missing = sorted(
        f"gene_interaction_predictor.{n}"
        for n, _ in _predictor(_lazy()).named_parameters()
    )
    assert len(expected_missing) == 13
    for follow, missing_expected in [
        (FOLLOW_INERT, expected_missing),
        (FOLLOW_LIVE, []),
    ]:
        model = _lazy(seed=1, local_predictor_config=CONCAT)
        pred, _ = model(_cell_graph(), _collate(GENOTYPES, follow))
        pred.sum().backward()
        missing = sorted(n for n, p in model.named_parameters() if p.grad is None)
        assert missing == missing_expected, follow
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
    assert sum(p.numel() for p in _predictor(_lazy()).parameters()) == 370


def test_stored_perturbation_indices_batch_equals_the_ptr_path() -> None:
    """Without ``_ptr`` the model falls back to ``perturbation_indices_batch`` (line
    1317-1319): storing [0, 1, 1, 2, 2, 2] by hand gives the same local scores.
    """
    model = _lazy().eval()
    by_batch = _collate(GENOTYPES, FOLLOW_INERT)
    by_batch["gene"].perturbation_indices_batch = torch.tensor([0, 1, 1, 2, 2, 2])
    with torch.no_grad():
        _, out_ptr = model(_cell_graph(), _collate(GENOTYPES, FOLLOW_LIVE))
        _, out_batch = model(_cell_graph(), by_batch)
    torch.testing.assert_close(
        out_batch["local_interaction"], out_ptr["local_interaction"]
    )


def test_an_empty_genotype_scores_zero_or_crashes_the_local_expansion() -> None:
    """A genotype with no perturbations in the MIDDLE ({1}, {}, {0, 3}): batch_assign
    [0, 2, 2], scatter_mean over 3 groups, so its local score is exactly 0 and the
    others equal their genotypes alone. At the END with one gene per genotype ({1},
    {2}, {}): 2 local rows for 3 genotypes, batch_assign [0, 1] places them correctly
    and the empty genotype scores 0.

    Finding: at the END with more genes than non-empty genotypes ({0, 3}, {1}, {}),
    the predictor returns max(batch_assign) + 1 = 2 rows and the re-expansion indexes
    them with the 3-gene ``valid_mask`` (line 1358-1362), raising IndexError. Kuzmin
    TMI genotypes always carry deletions, so the 006 runs do not reach it.
    Pinned until the expansion is indexed by genotype, not by gene.
    """
    model = _lazy().eval()
    with torch.no_grad():
        _, middle = model(_cell_graph(), _collate([[1], [], [0, 3]], FOLLOW_LIVE))
        _, end = model(_cell_graph(), _collate([[1], [2], []], FOLLOW_LIVE))
        alone = {
            tuple(g): model(_cell_graph(), _collate([g], FOLLOW_LIVE))[1][
                "local_interaction"
            ][0]
            for g in ([1], [2], [0, 3])
        }
    assert middle["local_interaction"][1].item() == 0.0
    torch.testing.assert_close(middle["local_interaction"][0], alone[(1,)])
    torch.testing.assert_close(middle["local_interaction"][2], alone[(0, 3)])
    assert end["local_interaction"][2].item() == 0.0
    torch.testing.assert_close(end["local_interaction"][0], alone[(1,)])
    torch.testing.assert_close(end["local_interaction"][1], alone[(2,)])
    with pytest.raises(
        IndexError,
        match=re.escape(
            "The shape of the mask [3] at index 0 does not match the shape of the "
            "indexed tensor [2, 1] at index 0"
        ),
    ):
        model(_cell_graph(), _collate([[0, 3], [1], []], FOLLOW_LIVE))


# ---------------------------------------------------------------- forward identities


def test_gating_combines_global_and_local_with_softmax_gates() -> None:
    """Gating (default): gates = softmax(gate_mlp(cat(global, local))), rows sum to 1,
    prediction = sum(cat * gates); z_p = z_w - z_i exactly; z_w is [1, 8], z_i [3, 8];
    ``layer_aggregation_weights`` is [] for "sum".
    """
    model = _lazy().eval()
    with torch.no_grad():
        pred, out = model(_cell_graph(), _collate(GENOTYPES, FOLLOW_LIVE))
        stack = torch.cat([out["global_interaction"], out["local_interaction"]], dim=1)
        assert model.gate_mlp is not None
        gate = torch.softmax(model.gate_mlp(stack), dim=1)
    assert out["z_w"].shape == (1, HIDDEN) and out["z_i"].shape == (3, HIDDEN)
    assert torch.equal(out["z_p"], out["z_w"].expand(3, -1) - out["z_i"])
    torch.testing.assert_close(
        out["global_interaction"], model.global_interaction_predictor(out["z_p"])
    )
    torch.testing.assert_close(out["gate_weights"], gate)
    torch.testing.assert_close(out["gate_weights"].sum(1), torch.ones(3))
    torch.testing.assert_close(pred, (stack * gate).sum(1, keepdim=True))
    assert out["layer_aggregation_weights"] == []


def test_z_i_pools_only_the_kept_genes_of_each_genotype() -> None:
    """z_i[b] = attentional pool over the rows of genotype b whose pert_mask is False:
    genotype {0, 2, 4} pools rows 11 and 13 of the 15 batched rows.
    """
    model = _lazy().eval()
    batch = _collate(GENOTYPES, FOLLOW_LIVE)
    with torch.no_grad():
        _, out = model(_cell_graph(), batch)
        z = model.forward_single(batch)
        pooled = model.global_aggregator(z[[11, 13]], torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(out["z_i"][2:3], pooled)


def test_global_only_mode_returns_the_global_term_with_unit_gates() -> None:
    """``use_local_predictor: false`` (006 configs 078-080): prediction = global, gates
    ones(3, 1), and no "local_interaction" key.
    """
    model = _lazy(local_predictor_config={"use_local_predictor": False}).eval()
    with torch.no_grad():
        pred, out = model(_cell_graph(), _collate(GENOTYPES, FOLLOW_INERT))
    assert torch.equal(pred, out["global_interaction"])
    assert torch.equal(out["gate_weights"], torch.ones(3, 1))
    assert sorted(out) == [
        "gate_weights",
        "gene_interaction",
        "global_interaction",
        "layer_aggregation_weights",
        "pert_gene_embs",
        "z_i",
        "z_p",
        "z_w",
    ]


def test_an_unknown_combination_builds_and_fails_only_at_forward() -> None:
    """``combination_method`` is not validated in ``__init__``; the ValueError comes
    after the encoders have run (lines 1424-1427).
    """
    model = _lazy(local_predictor_config={"combination_method": "product"})
    assert model.gate_mlp is None
    with pytest.raises(
        ValueError, match=re.escape("Unknown combination method: product")
    ):
        model(_cell_graph(), _collate(GENOTYPES, FOLLOW_LIVE))


def test_cross_attention_model_reports_the_batch_weights_per_layer() -> None:
    """With cross_attention and 2 conv layers, ``layer_aggregation_weights`` holds one
    {"gene": [15, 2, 2]} per layer from the LAST forward_single call (the batch; the
    wildtype's weights are overwritten).
    """
    model = _lazy(
        num_layers=2,
        gene_encoder_config={
            "encoder_type": "gin",
            "graph_aggregation_method": "cross_attention",
            "graph_aggregation_config": {"num_heads": 2},
        },
    ).eval()
    with torch.no_grad():
        _, out = model(_cell_graph(), _collate(GENOTYPES, FOLLOW_LIVE))
    weights = out["layer_aggregation_weights"]
    assert len(weights) == 2
    assert [tuple(w["gene"].shape) for w in weights] == [(15, 2, 2), (15, 2, 2)]


def test_genotype_score_ignores_gene_order_and_batch_order_permutes_rows() -> None:
    """perturbation_indices [4, 0, 2] instead of [0, 2, 4] gives the same prediction
    (the local head is permutation equivariant and pools by mean; the global path reads
    masks). Reversing the genotype order of the batch reverses the predictions (norm
    "layer": its statistics are sums over all rows, so order-free).
    """
    model = _lazy().eval()
    with torch.no_grad():
        sorted_pred, _ = model(_cell_graph(), _collate([[0, 2, 4]], FOLLOW_LIVE))
        shuffled_pred, _ = model(_cell_graph(), _collate([[4, 0, 2]], FOLLOW_LIVE))
        forward, _ = model(_cell_graph(), _collate(GENOTYPES, FOLLOW_LIVE))
        backward, _ = model(_cell_graph(), _collate(GENOTYPES[::-1], FOLLOW_LIVE))
    torch.testing.assert_close(shuffled_pred, sorted_pred)
    torch.testing.assert_close(backward, forward.flip(0))


def test_forward_is_seeded_and_every_parameter_receives_gradient() -> None:
    """Same seed -> identical train-mode predictions; backward from the summed
    prediction gives every parameter a finite gradient (gating, live follow list).
    """
    batch = _collate(GENOTYPES, FOLLOW_LIVE)
    model = _lazy(seed=3)
    pred, _ = model(_cell_graph(), batch)
    again, _ = _lazy(seed=3)(_cell_graph(), batch)
    assert torch.equal(pred, again)
    pred.sum().backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize(
    ("parameter", "message"),
    [
        ("gene_embedding.weight", "NaN detected in wildtype embeddings (z_w)"),
        (
            "global_aggregator.gate_nn.0.weight",
            "NaN detected in global wildtype embeddings (z_w_global)",
        ),
        (
            "gene_interaction_predictor.prediction_layer.weight",
            "NaN detected in local interaction predictions",
        ),
        (
            "global_interaction_predictor.3.weight",
            "NaN detected in global interaction predictions",
        ),
        ("gate_mlp.0.weight", "NaN detected in gate logits"),
    ],
)
def test_a_nan_parameter_is_named_by_the_first_stage_it_reaches(
    parameter: str, message: str
) -> None:
    """A NaN parameter raises RuntimeError naming the first guarded stage it reaches."""
    model = _lazy().eval()
    with torch.no_grad():
        dict(model.named_parameters())[parameter].fill_(float("nan"))
    with pytest.raises(RuntimeError, match=re.escape(message)):
        model(_cell_graph(), _collate(GENOTYPES, FOLLOW_LIVE))


def test_an_infinite_parameter_passes_every_guard() -> None:
    """Finding: every stage check is ``torch.isnan`` (e.g. line 1340), so an infinite
    final bias of the global predictor returns predictions [inf, inf, inf] with no
    error (the eager model's checks are finiteness checks since issue #540).
    Pinned until the lazy guards test ``isfinite``.
    """
    model = _lazy(local_predictor_config=CONCAT).eval()
    last = model.global_interaction_predictor[3]
    assert isinstance(last, nn.Linear)
    with torch.no_grad():
        last.bias.fill_(float("inf"))
        pred, _ = model(_cell_graph(), _collate(GENOTYPES, FOLLOW_LIVE))
    assert pred.flatten().tolist() == [float("inf")] * 3


# ---------------------------------------------------------------------------
# 2026.10.06 - Phase 21: remaining forward branches, the two guards an infinite
# parameter reaches, and the init branches the lazy factory cannot build.
# Expected values are structural identities: the model's own submodules applied by
# hand to the subset or index the branch promises.
# ---------------------------------------------------------------------------


def _wildtype_embeddings(model: GeneInteractionDango) -> torch.Tensor:
    out: torch.Tensor = model.preprocessor(model.gene_embedding(torch.arange(N)))
    return out


@pytest.mark.parametrize(
    ("parameter", "message"),
    [
        (
            "global_aggregator.transform_nn.0.bias",
            "NaN detected in perturbation difference (z_p_global)",
        ),
        (
            "gene_interaction_predictor.prediction_layer.bias",
            "NaN detected in gate weights after softmax",
        ),
    ],
)
def test_an_infinite_parameter_trips_the_late_guards(
    parameter: str, message: str
) -> None:
    """+inf (not NaN) reaches two guards no NaN parameter can reach first: an infinite
    transform bias makes z_w_global and z_i_global both +inf, so z_w - z_i is NaN
    (line 1304); an infinite local prediction makes both gate logits +inf, finite
    under isnan, and their softmax NaN (line 1399). Found by filling each parameter in
    turn with +inf and -inf (54 parameters x 2 signs x 2 combination methods).
    """
    model = _lazy().eval()
    with torch.no_grad():
        dict(model.named_parameters())[parameter].fill_(float("inf"))
    with pytest.raises(RuntimeError, match=re.escape(message)):
        model(_cell_graph(), _collate(GENOTYPES, FOLLOW_LIVE))


def test_a_wildtype_pert_mask_restricts_the_wildtype_pool() -> None:
    """A cell_graph carrying pert_mask (gene 4 removed) pools z_w over genes 0..3 only
    (line 1240): z_w equals the global aggregator on those four rows; without the mask
    it pools all five.
    """
    model = _lazy().eval()
    cell = _cell_graph()
    cell["gene"].pert_mask = torch.tensor([False, False, False, False, True])
    with torch.no_grad():
        _, out = model(cell, _collate(GENOTYPES, FOLLOW_LIVE))
        z_w = model.forward_single(_cell_graph())
        kept = model.global_aggregator(
            z_w[:4], index=torch.zeros(4, dtype=torch.long), dim_size=1
        )
        full = model.global_aggregator(
            z_w, index=torch.zeros(5, dtype=torch.long), dim_size=1
        )
        _, plain = model(_cell_graph(), _collate(GENOTYPES, FOLLOW_LIVE))
    torch.testing.assert_close(out["z_w"], kept)
    torch.testing.assert_close(plain["z_w"], full)
    assert not torch.allclose(kept, full)


def test_a_batch_without_pert_mask_pools_every_gene_by_its_batch_vector() -> None:
    """Without ``pert_mask`` the perturbed pool uses all rows and the full batch vector
    (lines 1275-1276): z_i equals the aggregator on forward_single(batch) indexed by
    gene.batch, so deleted genes are pooled too.
    """
    model = _lazy().eval()
    batch = _collate(GENOTYPES, FOLLOW_LIVE)
    del batch["gene"].pert_mask
    with torch.no_grad():
        _, out = model(_cell_graph(), batch)
        z_i = model.forward_single(batch)
        expected = model.global_aggregator(z_i, index=batch["gene"].batch)
    torch.testing.assert_close(out["z_i"], expected)
    assert out["z_i"].shape == (3, HIDDEN)


def test_forward_single_skips_absent_and_index_free_relations() -> None:
    """A graph without the regulatory relation, or with a regulatory store holding only
    a mask, runs the physical conv alone (lines 1177 and 1188): both equal the first
    conv layer applied to {physical: edges} with an all-True mask.
    """
    model = _lazy().eval()
    absent = _cell_graph(edges={"physical": EDGES["physical"]})
    index_free = _cell_graph(edges={"physical": EDGES["physical"]})
    index_free[REG].mask = torch.tensor([True, True, True])
    phys_edges = _edge_index(EDGES["physical"])
    with torch.no_grad():
        layer = model.convs[0]
        assert isinstance(layer, HeteroConvAggregator)
        expected, _ = layer(
            {"gene": _wildtype_embeddings(model)},
            {PHYS: phys_edges},
            {PHYS: torch.ones(4, dtype=torch.bool)},
        )
        torch.testing.assert_close(model.forward_single(absent), expected["gene"])
        torch.testing.assert_close(model.forward_single(index_free), expected["gene"])


def test_pairwise_aggregation_skips_a_missing_second_graph() -> None:
    """Names [physical, regulatory], only physical present: the inner loop skips
    regulatory (line 206), leaving options [pp(a, a), a] (identity = mean of one).
    """
    with torch.random.fork_rng():
        torch.manual_seed(0)
        agg = PairwiseGraphAggregation(HIDDEN, ["physical", "regulatory"]).eval()
        a = torch.randn(3, HIDDEN)
        with torch.no_grad():
            out, weights = agg({"physical": a})
            options = torch.stack(
                [agg.interaction_mlps["physical_physical"](torch.cat([a, a], -1)), a],
                dim=1,
            )
            w = torch.softmax(agg.pair_scorer(options).squeeze(-1), dim=-1)
        assert weights is not None and weights.shape == (3, 2)
        torch.testing.assert_close(weights, w)
        torch.testing.assert_close(out, (options * w.unsqueeze(-1)).sum(1))


def test_wrapper_passes_kwargs_to_a_non_gin_conv() -> None:
    """A GCNConv inside the wrapper is called as conv(x, edge_index, **kwargs)
    (line 849): edge_weight reaches it, and the output is act(proj(conv(...))).
    """
    with torch.random.fork_rng():
        torch.manual_seed(0)
        conv = GCNConv(HIDDEN, 5)
        wrapper = AttentionConvWrapper(
            conv, HIDDEN, norm=None, activation="tanh", dropout=0
        ).eval()
        x = torch.randn(4, HIDDEN)
        edges = _edge_index([(0, 1), (1, 2), (3, 2)])
        weight = torch.tensor([1.0, 0.5, 2.0])
        with torch.no_grad():
            out = wrapper(
                x,
                edges,
                edge_mask=torch.tensor([False, False, False]),
                edge_weight=weight,
            )
            expected = torch.tanh(wrapper.proj(conv(x, edges, edge_weight=weight)))
            unweighted = wrapper(x, edges)
        torch.testing.assert_close(out, expected)
        assert not torch.allclose(out, unweighted)


def test_init_resets_batch_norm_and_leaves_gatv2_untouched() -> None:
    """``_init_weights`` on any module tree: nn.LayerNorm and BatchNorm1d go to (1, 0)
    (lines 1098-1103; the model itself uses PyG norms, which are neither); a GATv2Conv keeps every parameter, because the branch looks for
    ``lin_src`` / ``lin_dst`` / ``att_src`` / ``att_dst`` and PyG 2.8 names them
    ``lin_l`` / ``lin_r`` / ``att`` (the eager model's pinned finding,
    test_hetero_cell_bipartite_dango_gi.py:620; the lazy factory refuses GATv2, so
    here the branch is dead code). A GATv2Conv given those names is re-initialized
    exactly as the branch says: zero biases, Kaiming fan_out weights, Xavier
    attention vectors (checked by replaying the draws under the same seed).
    """
    holder = nn.Module()
    bn = nn.BatchNorm1d(3)
    ln = nn.LayerNorm(3)
    gat = GATv2Conv(4, 2, heads=2)
    holder.bn = bn
    holder.ln = ln
    holder.gat = gat
    with torch.no_grad():
        for norm in (bn, ln):
            norm.weight.fill_(5.0)
            norm.bias.fill_(-2.0)
    before = {n: p.detach().clone() for n, p in gat.named_parameters()}
    GeneInteractionDango._init_weights(holder)  # type: ignore[arg-type, unused-ignore]
    state = holder.state_dict()
    for norm_name in ("bn", "ln"):
        assert torch.equal(state[f"{norm_name}.weight"], torch.ones(3)), norm_name
        assert torch.equal(state[f"{norm_name}.bias"], torch.zeros(3)), norm_name
    for name, value in gat.named_parameters():
        assert torch.equal(value, before[name]), name

    stand_in = GATv2Conv(4, 2, heads=2)  # given the names the branch expects
    stand_in.lin_src = stand_in.lin_l
    stand_in.lin_dst = stand_in.lin_r
    stand_in.att_src = nn.Parameter(torch.zeros(1, 2, 2))
    stand_in.att_dst = nn.Parameter(torch.zeros(1, 2, 2))
    with torch.no_grad():
        stand_in.lin_l.bias.fill_(3.0)
        stand_in.lin_r.bias.fill_(3.0)
    holder2 = nn.Module()
    holder2.gat = stand_in
    with torch.random.fork_rng():
        torch.manual_seed(0)
        GeneInteractionDango._init_weights(holder2)  # type: ignore[arg-type, unused-ignore]
    assert not stand_in.lin_l.bias.any() and not stand_in.lin_r.bias.any()
    # replay: apply() visits the GATv2Conv after its children (PyG Linear, untouched),
    # then draws kaiming(lin_src), kaiming(lin_dst), xavier(att_src), xavier(att_dst)
    replay = [
        torch.empty(4, 4),
        torch.empty(4, 4),
        torch.empty(1, 2, 2),
        torch.empty(1, 2, 2),
    ]
    with torch.random.fork_rng():
        torch.manual_seed(0)
        nn.init.kaiming_normal_(replay[0], mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(replay[1], mode="fan_out", nonlinearity="relu")
        nn.init.xavier_normal_(replay[2])
        nn.init.xavier_normal_(replay[3])
    state = holder2.state_dict()
    for key, expected in zip(
        ["gat.lin_l.weight", "gat.lin_r.weight", "gat.att_src", "gat.att_dst"], replay
    ):
        assert torch.equal(state[key], expected), key


# ---------------------------------------------------------------------------
# 2026.10.07 - Phase 24: the fallback sum, edge stores given as plain dicts, the
# perturbed-stage guards, the 1-D unsqueezes, and the three late guards an infinite
# (not NaN) prediction reaches. Each guard is reached by replacing the ONE submodule
# that produces the tensor it checks; the lazy guards test ``isnan``, so +inf passes
# them and only inf - inf or inf * 0 trips one.
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from tests.torchcell.models.test_hetero_cell_bipartite_dango_gi import (  # noqa: E402
    _Fill,
    _Squeeze,
)

INF = float("inf")


def test_hetero_conv_fallback_sum_when_the_method_has_no_aggregator() -> None:
    """Built as "sum" then rewritten to "cross_attention" (no aggregator was built), the
    final ``else`` (lines 370-373) sums the convs 2x + 3x = 5x, stores a None weight,
    and so returns no attention weights at all.
    """
    convs: dict[Any, nn.Module] = {PHYS: _MaskedScale(2.0), REG: _MaskedScale(3.0)}
    layer = HeteroConvAggregator(convs, 2, "sum")
    layer.aggregation_method = "cross_attention"
    assert layer.aggregator is None
    x = {"gene": torch.arange(6.0).reshape(3, 2)}
    edges = {PHYS: _edge_index([(0, 1)]), REG: _edge_index([(1, 2)])}
    out, attn = layer(x, edges)
    assert torch.equal(out["gene"], 5 * x["gene"])
    assert attn is None


class _DictStores:
    """A graph whose edge stores are plain dicts (``"edge_index" in store`` works,
    ``hasattr(store, "edge_index")`` does not) and whose gene store is a namespace.
    """

    def __init__(self, n: int, stores: dict[tuple[str, str, str], dict[str, Any]]):
        self.gene = SimpleNamespace(num_nodes=n)
        self.stores = stores

    @property
    def edge_types(self) -> list[tuple[str, str, str]]:
        return list(self.stores)

    def __getitem__(self, key: Any) -> Any:
        return self.gene if key == "gene" else self.stores[key]


def test_forward_single_reads_dict_edge_stores_by_key() -> None:
    """Edge index and mask under dict KEYS (lines 1176-1177 and 1187-1188): physical with
    mask [T, F, T, T], regulatory with no mask (all-True fallback). The embeddings equal
    those of the same graph as a HeteroData with the mask as an attribute, and differ
    from the unmasked graph's, so the dict mask is applied.
    """
    model = _lazy().eval()
    phys_mask = torch.tensor([True, False, True, True])
    fake = _DictStores(
        N,
        {
            PHYS: {"edge_index": _edge_index(EDGES["physical"]), "mask": phys_mask},
            REG: {"edge_index": _edge_index(EDGES["regulatory"])},
        },
    )
    hetero = _cell_graph()
    hetero[PHYS].mask = phys_mask
    with torch.no_grad():
        from_dict = model.forward_single(fake)  # type: ignore[arg-type, unused-ignore]
        from_attrs = model.forward_single(hetero)
        unmasked = model.forward_single(_cell_graph())
    assert torch.equal(from_dict, from_attrs)
    assert not torch.equal(from_dict, unmasked)


class _NthNaN(nn.Module):
    """Wraps a stage; its ``k``-th call (1-based) returns NaN everywhere."""

    def __init__(self, inner: Any, k: int) -> None:
        super().__init__()
        self.inner = inner
        self.k = k
        self.calls = 0

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.calls += 1
        out: torch.Tensor = self.inner(*args, **kwargs)
        return torch.full_like(out, float("nan")) if self.calls == self.k else out


def test_perturbed_stage_guards_name_their_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``forward_single``'s 2nd call (the batch) returning NaN: "perturbed embeddings
    (z_i)" (line 1256); the global aggregator's 2nd call (the batch pool) returning
    NaN: "global perturbed embeddings (z_i_global)" (line 1275). The first calls (the
    wildtype) pass, so each message names the batch stage.
    """
    model = _lazy().eval()
    wrapped = _NthNaN(model.forward_single, 2)
    object.__setattr__(model, "forward_single", wrapped)
    with pytest.raises(
        RuntimeError, match=r"^NaN detected in perturbed embeddings \(z_i\)$"
    ):
        model(_cell_graph(), _collate(GENOTYPES, FOLLOW_LIVE))
    assert wrapped.calls == 2

    model = _lazy().eval()
    monkeypatch.setattr(model, "global_aggregator", _NthNaN(model.global_aggregator, 2))
    with pytest.raises(
        RuntimeError,
        match=r"^NaN detected in global perturbed embeddings \(z_i_global\)$",
    ):
        model(_cell_graph(), _collate(GENOTYPES, FOLLOW_LIVE))


@pytest.mark.parametrize("which", ["global", "local"])
def test_a_one_dimensional_interaction_is_unsqueezed_to_a_column(
    monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    """The global (line 1366) or local (line 1368) predictor wrapped to return [3]
    instead of [3, 1]: the forward unsqueezes it back, so the predictions are
    bit-identical to the unwrapped model's, [3, 1].
    """
    batch = _collate(GENOTYPES, FOLLOW_LIVE)
    with torch.no_grad():
        expected, _ = _lazy().eval()(_cell_graph(), batch)
        model = _lazy().eval()
        name = (
            "global_interaction_predictor"
            if which == "global"
            else "gene_interaction_predictor"
        )
        monkeypatch.setattr(model, name, _Squeeze(getattr(model, name)))
        pred, _ = model(_cell_graph(), batch)
    assert pred.shape == (3, 1)
    assert torch.equal(pred, expected)


@pytest.mark.parametrize(
    ("config", "local", "gate", "message"),
    [
        # a zero gate on the infinite global prediction: inf * 0 = NaN
        (None, 1.0, [-200.0, 0.0], "NaN detected in weighted predictions"),
        # concat: 0.5 * inf + 0.5 * (-inf) = NaN
        (CONCAT, -INF, None, "NaN detected in concatenated gene interaction"),
        # gating with equal gates: inf * 0.5 + (-inf) * 0.5 = NaN in the sum only
        (None, -INF, [0.0, 0.0], "NaN detected in final gene interaction output"),
    ],
)
def test_an_infinite_prediction_trips_the_guard_where_it_first_becomes_nan(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any] | None,
    local: float,
    gate: list[float] | None,
    message: str,
) -> None:
    """The global predictor returns +inf (``isnan`` lets it through every earlier
    guard), the local predictor ``local``, the gate MLP fixed logits:

    * logits [-200, 0]: softmax weights [exp(-200) = 0 in float32, 1], so the weighted
      global entry is inf * 0 = NaN (line 1398);
    * concat mode, local -inf: 0.5 inf + 0.5 (-inf) = NaN (line 1414);
    * logits [0, 0]: weights [0.5, 0.5], weighted entries +inf and -inf (no NaN), whose
      sum inf - inf is NaN, caught by the final check (line 1423).
    """
    overrides: dict[str, Any] = (
        {} if config is None else {"local_predictor_config": config}
    )
    model = _lazy(**overrides).eval()
    monkeypatch.setattr(model, "global_interaction_predictor", _Fill(INF))
    monkeypatch.setattr(model, "gene_interaction_predictor", _Fill(local, local=True))
    if gate is not None:
        monkeypatch.setattr(model, "gate_mlp", _Fill(gate))
    with pytest.raises(RuntimeError, match=f"^{re.escape(message)}$"):
        model(_cell_graph(), _collate(GENOTYPES, FOLLOW_LIVE))
