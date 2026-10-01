# tests/torchcell/models/test_hetero_cell_bipartite_dango_gi.py
# [[tests.torchcell.models.test_hetero_cell_bipartite_dango_gi]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_hetero_cell_bipartite_dango_gi.py
"""``GeneInteractionDango`` and its parts on a hand-built four-gene, two-graph batch.

Fixture. Four genes 0..3 and two gene graphs, ``physical`` and ``regulatory`` (a real
``GeneMultiGraph``; the model reads only its sorted keys). The wildtype ``cell_graph``
has ``gene.num_nodes = 4`` and edges physical 0->1, 1->2, 2->3 and regulatory 3->0,
0->2. A perturbed sample keeps the unperturbed genes in index order, relabels the
wildtype edges among them, and stores ``pert_mask`` (length 4) and
``perturbation_indices`` (wildtype ids, which PyG does not increment on collation, since
"indices" does not contain the substring "index"); samples are collated with
``follow_batch=["perturbation_indices"]`` so ``perturbation_indices_ptr`` exists.

Tiny configuration (``_tiny``): d = 8 hidden, 1 conv layer, GIN encoder with the default
2-layer MLP, "sum" graph aggregation, LayerNorm, local predictor with 2 heads and 1
attention layer, gating. Parameter count by component:

* gene_embedding 4 * 8 = 32
* preprocessor Linear(8, 8) twice (72 + 72) + ONE shared LayerNorm (16) = 160
* convs, per graph: GIN eps (1) + Linear(8, 8) twice (144) + PyG LayerNorm (16) = 161,
  two graphs = 322 (proj is Identity because the GIN MLP already ends at 8)
* gene_interaction_predictor: static Linear 72 + q/k/v/out 4 * 72 + beta 1 = 361, plus
  prediction Linear(8, 1) 9 = 370
* global_aggregator: gate Linear(8, 4) 36 + Linear(4, 1) 5 + transform Linear(8, 8) 72
  = 113 (``AttentionalAggregation`` re-registers the same modules; counted once)
* global_interaction_predictor Linear(8, 8) 72 + Linear(8, 1) 9 = 81
* gate_mlp Linear(2, 8) 24 + Linear(8, 2) 18 = 42

Total 1120. Component closed forms are derived in each test docstring.

2026.09.30, Phase 17. Config flags each get a pinned consequence on the same fixture:
the exact total for every encoder x aggregation pair (cross_attention +304, pairwise
+645, GATv2 with 2 heads +30), ``num_layers`` 2 (+322) and ``num_attention_layers`` 2
(+289), a gradient reaching every parameter in all eight builds, and two masked-softmax
identities (a two-gene HyperSAGNN sample and GATv2 nodes of in-degree 1 give their
logit parameters a gradient of exactly 0; three genes or a second in-edge make it
nonzero). Permuting the genotypes of a batch permutes the predictions. The
training script ``main`` runs on the fixture with the genome, the graph builder, the
loader, ``load_dotenv``, ``timestamp`` and ``plt.savefig`` faked: the plot schedule is
``epoch % n == 0`` or the last epoch, the warmup scheduler is stepped once per epoch,
and an unknown loss is refused before the plot directory exists.

2026.09.30, issue #540. Every stage check is a finiteness check, so an infinite
parameter is named at the first stage it reaches; ``activation`` maps through
``act_register`` (gelu builds GELU); a GIN MLP with no Linear is refused by name; the
aggregation config's own ``dropout`` wins over the model dropout; ``main`` seeds from
the config's ``seed``, so at lr 0 its final metrics are those of ``_tiny(seed)``
whatever the caller's RNG state; the shipped Wasserstein loss trains and prints its
own ``weighted_wasserstein`` component; every composite loss saves the final
components figure.

2026.10.01, issue #540 (last item). ``aggregation_norm`` is an explicit, validated
argument: null (or absent) builds exactly the module main built (same parameter count,
same state_dict keys), and any other value raises
``AggregationNormNotImplementedError`` with an exact message, both from the model and
from ``main``'s config reader.
"""

import os
import os.path as osp
import re
import sys
import types
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sortedcontainers import SortedDict
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv
from torch_geometric.nn import LayerNorm as PygLayerNorm

import torchcell.models.hetero_cell_bipartite_dango_gi as dango_module
from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.losses.logcosh import LogCoshLoss
from torchcell.losses.mle_wasserstein import MleWassSupCR
from torchcell.models.act import act_register
from torchcell.models.hetero_cell_bipartite_dango_gi import (
    AggregationNormNotImplementedError,
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
from torchcell.scheduler.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torchcell.sequence import GeneSet

N_GENES = 4
HIDDEN = 8
WILDTYPE_EDGES = {"physical": [(0, 1), (1, 2), (2, 3)], "regulatory": [(3, 0), (0, 2)]}
PHYS = ("gene", "physical", "gene")
REG = ("gene", "regulatory", "gene")


def _multigraph() -> GeneMultiGraph:
    genes = GeneSet(["YAL001C", "YAL002W", "YAL003W", "YAL004W"])
    graphs = {}
    for name in ["physical", "regulatory"]:
        graph = nx.Graph()
        graph.add_nodes_from(genes)
        graphs[name] = GeneGraph(name=name, graph=graph, max_gene_set=genes)
    return GeneMultiGraph(graphs=SortedDict(graphs))


def _edge_index(pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _cell_graph(edges: dict[str, list[tuple[int, int]]] = WILDTYPE_EDGES) -> HeteroData:
    data = HeteroData()
    data["gene"].num_nodes = N_GENES
    for name, pairs in edges.items():
        data["gene", name, "gene"].edge_index = _edge_index(pairs)
    return data


def _sample(
    pert: list[int], edges: dict[str, list[tuple[int, int]]] = WILDTYPE_EDGES
) -> HeteroData:
    """Keep unperturbed genes in index order and relabel the wildtype edges among them."""
    keep = [g for g in range(N_GENES) if g not in pert]
    new_id = {g: i for i, g in enumerate(keep)}
    data = HeteroData()
    data["gene"].num_nodes = len(keep)
    mask = torch.zeros(N_GENES, dtype=torch.bool)
    mask[pert] = True
    data["gene"].pert_mask = mask
    data["gene"].perturbation_indices = torch.tensor(pert, dtype=torch.long)
    for name, pairs in edges.items():
        kept = [(new_id[s], new_id[t]) for s, t in pairs if s in new_id and t in new_id]
        data["gene", name, "gene"].edge_index = _edge_index(kept)
    return data


def _batch(
    perts: list[list[int]], edges: dict[str, list[tuple[int, int]]] = WILDTYPE_EDGES
) -> Batch:
    return Batch.from_data_list(
        [_sample(p, edges) for p in perts], follow_batch=["perturbation_indices"]
    )


def _tiny(seed: int = 0, **overrides: Any) -> GeneInteractionDango:
    torch.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "gene_num": N_GENES,
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


# ---------------------------------------------------------------- small helpers


def test_norm_layer_factory_returns_eps_1e5_layers_and_rejects_others() -> None:
    """Norm factory: "layer" is nn.LayerNorm, "batch" nn.BatchNorm1d, both eps 1e-5."""
    layer = get_norm_layer(6, "layer")
    batch = get_norm_layer(6, "batch")
    assert type(layer) is nn.LayerNorm and layer.normalized_shape == (6,)
    assert type(batch) is nn.BatchNorm1d and batch.num_features == 6
    assert layer.eps == 1e-5 and batch.eps == 1e-5
    with pytest.raises(ValueError, match="Unsupported norm type: group"):
        get_norm_layer(6, "group")


def test_weight_l2_norm_is_the_root_sum_of_squares_over_trainable_params() -> None:
    """Linear(2, 1) with W = [3, 4], b = [12] -> sqrt(9 + 16 + 144) = 13; a frozen
    parameter is excluded, so freezing b leaves sqrt(25) = 5.
    """
    linear = nn.Linear(2, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[3.0, 4.0]]))
        linear.bias.copy_(torch.tensor([12.0]))
    assert calculate_weight_l2_norm(linear) == pytest.approx(13.0)
    linear.bias.requires_grad_(False)
    assert calculate_weight_l2_norm(linear) == pytest.approx(5.0)


def test_rolling_correlation_windows_and_the_constant_window_zero() -> None:
    """Window 3 over 4 points gives 2 values. x = [1, 2, 3, 4], y = [2, 4, 6, 5]:
    window 1 is perfectly linear (r = 1); window 2 is x = [2, 3, 4], y = [4, 6, 5],
    centered x [-1, 0, 1], y [-1, 1, 0], so r = 1 / sqrt(2 * 2) = 0.5. A constant window
    reports 0.0 instead of NaN; a series shorter than the window returns [].
    """
    values = calculate_rolling_correlation([1, 2, 3, 4], [2, 4, 6, 5], window=3)
    assert values == pytest.approx([1.0, 0.5])
    assert calculate_rolling_correlation([1, 1, 1], [1, 2, 3], window=3) == [0.0]
    assert calculate_rolling_correlation([1, 2], [1, 2], window=3) == []


def test_preprocessor_shares_one_norm_module_across_its_layers() -> None:
    """Finding: ``PreProcessor`` builds ``norm_layer`` once and appends the SAME module in
    every layer, so a 3-layer MLP has one set of norm parameters: Linear(5, 8) 48 +
    2 * Linear(8, 8) 144 + LayerNorm 16 = 208, not 240.

    Contract (issue #540): the activation is the ``act_register`` module of that name,
    so "gelu" builds GELU and the forward is gelu(norm(linear(.))) per layer; an
    unregistered name is refused by name.
    """
    torch.manual_seed(0)
    pre = PreProcessor(5, 8, num_layers=3, dropout=0.0, activation="gelu")
    norms = [m for m in pre.mlp if isinstance(m, nn.LayerNorm)]
    assert len(norms) == 3 and norms[0] is norms[1] is norms[2]
    assert sum(p.numel() for p in pre.parameters()) == 208
    assert type(pre.act) is nn.GELU and pre.act is act_register["gelu"]
    assert type(PreProcessor(5, 8, activation="tanh").act) is nn.Tanh
    x = torch.randn(3, 5)
    expected = x
    for linear in [pre.mlp[0], pre.mlp[4], pre.mlp[8]]:
        expected = nn.functional.gelu(norms[0](linear(expected)))
    torch.testing.assert_close(pre(x), expected)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unknown activation: silu; expected one of "
            "['gelu', 'leaky_relu', 'relu', 'sigmoid', 'tanh']"
        ),
    ):
        PreProcessor(5, 8, activation="silu")


# ---------------------------------------------------------------- conv construction


def test_gatv2_layer_splits_the_width_across_heads() -> None:
    """Out 8 with heads 2 -> GATv2Conv(.., 4, heads=2, concat=True, no self loops); the
    wrapper's expected width is heads * out = 8, so its projection is Identity; with
    concat False the conv emits 4 and the wrapper adds Linear(4, 8).
    """
    conv = create_conv_layer("gatv2", 8, 8, {"heads": 2})
    assert isinstance(conv, GATv2Conv)
    assert (conv.heads, conv.out_channels, conv.concat) == (2, 4, True)
    assert conv.add_self_loops is False
    assert isinstance(AttentionConvWrapper(conv, 8).proj, nn.Identity)
    flat = create_conv_layer("gatv2", 8, 8, {"heads": 2, "concat": False})
    proj = AttentionConvWrapper(flat, 8).proj
    assert isinstance(proj, nn.Linear) and (proj.in_features, proj.out_features) == (
        4,
        8,
    )


def test_gin_layer_mlp_depth_and_a_one_layer_mlp_that_ignores_out_channels() -> None:
    """3 GIN layers: Linear(8, 16), ReLU, Dropout, Linear(16, 16), ReLU, Dropout,
    Linear(16, 8); ``train_eps`` makes eps a parameter.

    Finding: with ``gin_num_layers=1`` only the first branch runs
    (hetero_cell_bipartite_dango_gi.py:703-706), so the MLP is Linear(8, 16), ReLU,
    Dropout and the conv emits ``gin_hidden_dim`` = 16, not ``out_channels`` = 8; the
    wrapper then has to add Linear(16, 8).
    """
    deep = create_conv_layer("gin", 8, 8, {"gin_hidden_dim": 16, "gin_num_layers": 3})
    assert isinstance(deep, GINConv) and isinstance(deep.eps, nn.Parameter)
    shapes = [
        (m.in_features, m.out_features) for m in deep.nn if isinstance(m, nn.Linear)
    ]
    assert shapes == [(8, 16), (16, 16), (16, 8)]
    shallow = create_conv_layer(
        "gin", 8, 8, {"gin_hidden_dim": 16, "gin_num_layers": 1}
    )
    assert isinstance(shallow, GINConv) and isinstance(shallow.nn, nn.Sequential)
    assert [type(m) for m in shallow.nn] == [nn.Linear, nn.ReLU, nn.Dropout]
    assert shallow(torch.randn(3, 8), _edge_index([(0, 1)])).shape == (3, 16)
    proj = AttentionConvWrapper(shallow, 8).proj
    assert isinstance(proj, nn.Linear) and (proj.in_features, proj.out_features) == (
        16,
        8,
    )
    with pytest.raises(ValueError, match="Unknown encoder type: sage"):
        create_conv_layer("sage", 8, 8, {})


def test_wrapper_width_rules_for_other_convs_and_its_norm_options() -> None:
    """GCNConv(8, 5): width from ``out_channels`` -> Linear(5, 8). A GINConv whose nn is
    not a Sequential falls back to ``target_dim`` (Identity). norm "batch" -> PyG
    BatchNorm, "layer" -> PyG LayerNorm, anything else or None -> no norm; dropout 0 ->
    no dropout module.

    ``activation=None`` (the default) means no activation, like ``norm=None``: the act
    is Identity. A name maps through ``act_register`` ("gelu" -> GELU).
    """
    gcn = AttentionConvWrapper(GCNConv(8, 5), 8, norm="batch", dropout=0.0)
    assert isinstance(gcn.proj, nn.Linear) and gcn.proj.in_features == 5
    assert type(gcn.norm).__name__ == "BatchNorm"
    bare_gin = AttentionConvWrapper(GINConv(nn.Linear(8, 8)), 8, norm="layer")
    assert isinstance(bare_gin.proj, nn.Identity)
    assert isinstance(bare_gin.norm, PygLayerNorm)
    assert AttentionConvWrapper(GCNConv(8, 8), 8, norm="group").norm is None
    assert AttentionConvWrapper(GCNConv(8, 8), 8, norm=None).norm is None
    assert gcn.dropout is None
    assert type(gcn.act) is nn.Identity
    assert type(AttentionConvWrapper(GCNConv(8, 8), 8, activation="gelu").act) is (
        nn.GELU
    )


def test_wrapper_forward_is_act_of_norm_of_proj_of_conv() -> None:
    """eval: out = relu(LayerNorm_graph(W_proj conv(x, e) + b)), checked term by term."""
    torch.manual_seed(0)
    wrapper = AttentionConvWrapper(
        GCNConv(4, 3), 6, norm="layer", activation="relu", dropout=0.5
    ).eval()
    x = torch.randn(5, 4)
    edges = _edge_index([(0, 1), (1, 2), (3, 4)])
    inner = wrapper.proj(wrapper.conv(x, edges))
    expected = torch.relu(wrapper.norm(inner))
    torch.testing.assert_close(wrapper(x, edges), expected)
    assert (wrapper(x, edges) >= 0).all()


# ---------------------------------------------------------------- graph aggregation


def test_self_attention_aggregation_parameters_and_the_one_graph_closed_form() -> None:
    """MultiheadAttention(d) holds 4d^2 + 4d = 288 parameters at d = 8, plus G * d = 16
    graph embeddings: 304. With one graph each node attends only to itself (weight 1), so
    aggregated = out_proj(v_proj(x + e_0)); empty input gives (None, None).
    """
    torch.manual_seed(0)
    agg = SelfAttentionGraphAggregation(HIDDEN, num_graphs=2, num_heads=2).eval()
    assert sum(p.numel() for p in agg.parameters()) == 304
    x = torch.randn(3, HIDDEN)
    out, weights = agg({"physical": x})
    attn = agg.multihead_attn
    w_v = attn.in_proj_weight[2 * HIDDEN :]
    b_v = attn.in_proj_bias[2 * HIDDEN :]
    v = (x + agg.graph_embeddings[0]) @ w_v.t() + b_v
    torch.testing.assert_close(out, attn.out_proj(v))
    assert weights is not None and torch.equal(weights, torch.ones(3, 1, 1))
    assert agg({}) == (None, None)


def test_self_attention_aggregation_is_node_equivariant_and_key_order_free() -> None:
    """Attention runs across graphs per node, so permuting nodes permutes the output;
    graph names are sorted, so dict insertion order changes nothing. Weights rows sum
    to 1 over the 2 graphs.
    """
    torch.manual_seed(0)
    agg = SelfAttentionGraphAggregation(HIDDEN, num_graphs=2, num_heads=2).eval()
    a, b = torch.randn(4, HIDDEN), torch.randn(4, HIDDEN)
    out, weights = agg({"physical": a, "regulatory": b})
    swapped, _ = agg({"regulatory": b, "physical": a})
    assert torch.equal(out, swapped)
    perm = torch.tensor([2, 0, 3, 1])
    permuted, _ = agg({"physical": a[perm], "regulatory": b[perm]})
    torch.testing.assert_close(permuted, out[perm])
    assert weights is not None and weights.shape == (4, 2, 2)
    torch.testing.assert_close(weights.sum(-1), torch.ones(4, 2))


def test_pairwise_aggregation_parameters_keys_and_the_single_graph_case() -> None:
    """Two graphs give G(G+1)/2 = 3 pair MLPs of 2d*d + d + d*d + d = 208 each (624) and
    an attention head d*(d/4) + d/4 + d/4 + 1 = 21: 645. With only "physical" present the
    one interaction has weight 1, so the output is mlp_physical_physical(cat(x, x)).
    """
    torch.manual_seed(0)
    agg = PairwiseGraphAggregation(HIDDEN, ["regulatory", "physical"]).eval()
    assert list(agg.interaction_mlps) == [
        "physical_physical",
        "physical_regulatory",
        "regulatory_regulatory",
    ]
    assert sum(p.numel() for p in agg.parameters()) == 645
    x = torch.randn(3, HIDDEN)
    out, weights = agg({"physical": x})
    mlp = agg.interaction_mlps["physical_physical"]
    torch.testing.assert_close(out, mlp(torch.cat([x, x], dim=-1)))
    assert weights is not None and torch.equal(weights, torch.ones(3, 1))


def test_pairwise_aggregation_is_the_attention_weighted_sum_of_three_pairs() -> None:
    """Out = sum_k softmax_k(att(I_k)) I_k with I = [pp(a,a), pr(a,b), rr(b,b)]."""
    torch.manual_seed(0)
    agg = PairwiseGraphAggregation(HIDDEN, ["physical", "regulatory"]).eval()
    a, b = torch.randn(2, HIDDEN), torch.randn(2, HIDDEN)
    out, weights = agg({"physical": a, "regulatory": b})
    mlps = agg.interaction_mlps
    stacked = torch.stack(
        [
            mlps["physical_physical"](torch.cat([a, a], -1)),
            mlps["physical_regulatory"](torch.cat([a, b], -1)),
            mlps["regulatory_regulatory"](torch.cat([b, b], -1)),
        ],
        dim=1,
    )
    expected_weights = torch.softmax(agg.attention(stacked).squeeze(-1), dim=-1)
    assert weights is not None
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(out, (stacked * expected_weights.unsqueeze(-1)).sum(1))


def test_pairwise_aggregation_falls_back_to_the_mean_for_unknown_graphs() -> None:
    """No known graph name -> (mean of the inputs, None); empty input -> (None, None)."""
    agg = PairwiseGraphAggregation(HIDDEN, ["physical"])
    a, b = torch.ones(2, HIDDEN), 3 * torch.ones(2, HIDDEN)
    out, weights = agg({"x": a, "y": b})
    assert torch.equal(out, 2 * torch.ones(2, HIDDEN)) and weights is None
    assert agg({}) == (None, None)


class _Scale(nn.Module):
    """A parameter-free 'conv' that returns c * x (ignores edges) for exact aggregation."""

    def __init__(self, c: float) -> None:
        super().__init__()
        self.c = c

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.c * x


def test_hetero_conv_sum_and_mean_are_exact_and_missing_edge_types_are_skipped() -> (
    None
):
    """Convs 2x (physical) and 3x (regulatory): sum = 5x, mean = 2.5x, no attention.
    Only "physical" in the edge dict -> 2x. No matching edges -> empty output.
    """
    convs: dict[Any, nn.Module] = {PHYS: _Scale(2.0), REG: _Scale(3.0)}
    x = {"gene": torch.arange(6.0).reshape(3, 2)}
    edges = {PHYS: _edge_index([(0, 1)]), REG: _edge_index([(1, 2)])}
    summed, attn = HeteroConvAggregator(convs, 2, "sum")(x, edges)
    assert torch.equal(summed["gene"], 5 * x["gene"]) and attn is None
    mean, _ = HeteroConvAggregator(convs, 2, "mean")(x, edges)
    assert torch.equal(mean["gene"], 2.5 * x["gene"])
    only, _ = HeteroConvAggregator(convs, 2, "sum")(x, {PHYS: edges[PHYS]})
    assert torch.equal(only["gene"], 2 * x["gene"])
    assert HeteroConvAggregator(convs, 2, "sum")(x, {}) == ({}, None)
    with pytest.raises(ValueError, match="Unknown aggregation method: max"):
        HeteroConvAggregator(convs, 2, "max")


def test_hetero_conv_learned_aggregators_return_their_weights() -> None:
    """cross_attention -> {"gene": [N, 2, 2]}; pairwise -> {"gene": [N, 3]}; each output
    equals the aggregator applied to {rel: conv(x)}.
    """
    torch.manual_seed(0)
    convs: dict[Any, nn.Module] = {PHYS: _Scale(2.0), REG: _Scale(3.0)}
    x = {"gene": torch.randn(3, HIDDEN)}
    edges = {PHYS: _edge_index([(0, 1)]), REG: _edge_index([(1, 2)])}
    per_graph = {"physical": 2 * x["gene"], "regulatory": 3 * x["gene"]}
    for method, shape in [
        ("cross_attention", (3, 2, 2)),
        ("pairwise_interaction", (3, 3)),
    ]:
        layer = HeteroConvAggregator(
            convs, HIDDEN, method, {"num_heads": 2, "dropout": 0.0}
        ).eval()
        out, attn = layer(x, edges)
        assert attn is not None and attn["gene"] is not None
        assert tuple(attn["gene"].shape) == shape
        assert layer.aggregator is not None
        expected, _ = layer.aggregator(per_graph)
        torch.testing.assert_close(out["gene"], expected)


def test_attentional_pooling_is_a_per_group_softmax_weighted_sum() -> None:
    """pool[g] = sum_{i in g} softmax_g(gate(x_i)) * transform(x_i); invariant to the
    order of nodes within a group.
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


# ---------------------------------------------------------------- Dango local head


def test_hypersagnn_rezero_starts_at_001_and_beta_zero_is_the_identity() -> None:
    """Each layer's beta is initialized to exactly 0.01; with beta = 0 the dynamic
    embeddings equal the input bit for bit, and static = relu(W x + b) always.
    """
    torch.manual_seed(0)
    sagnn = DangoLikeHyperSAGNN(HIDDEN, num_heads=2, num_layers=2, dropout=0.0)
    assert [float(b) for b in sagnn.beta_params] == [
        pytest.approx(0.01),
        pytest.approx(0.01),
    ]
    with torch.no_grad():
        for beta in sagnn.beta_params:
            beta.zero_()
    x = torch.randn(3, HIDDEN)
    static, dynamic = sagnn(x)
    assert torch.equal(dynamic, x)
    torch.testing.assert_close(static, torch.relu(sagnn.static_embedding[0](x)))
    with pytest.raises(AssertionError, match="must be divisible by num_heads 3"):
        DangoLikeHyperSAGNN(HIDDEN, num_heads=3)


def test_hypersagnn_two_genes_attend_only_to_each_other() -> None:
    """The diagonal is masked, so with 2 genes every head puts weight 1 on the other
    gene: dynamic_i = x_i + beta * out_proj(v_proj(x_j)) for j != i (one layer).
    A single gene has nothing to attend to and passes through unchanged.
    """
    torch.manual_seed(0)
    sagnn = DangoLikeHyperSAGNN(HIDDEN, num_heads=2, num_layers=1, dropout=0.0)
    layer = sagnn.attention_layers[0]
    assert isinstance(layer, nn.ModuleDict)
    x = torch.randn(2, HIDDEN)
    _, dynamic = sagnn(x)
    other = layer["out_proj"](layer["v_proj"](x.flip(0)))
    torch.testing.assert_close(dynamic, x + sagnn.beta_params[0] * other)
    one = torch.randn(1, HIDDEN)
    assert torch.equal(sagnn(one)[1], one)


def test_hypersagnn_batches_are_independent_and_equivariant_within_a_batch() -> None:
    """With a batch vector each group is processed alone: rows of group b equal the
    unbatched call on that group, and permuting a group's rows permutes its output.
    """
    torch.manual_seed(0)
    sagnn = DangoLikeHyperSAGNN(HIDDEN, num_heads=2, num_layers=2, dropout=0.0)
    x = torch.randn(5, HIDDEN)
    batch = torch.tensor([0, 0, 0, 1, 1])
    _, dynamic = sagnn(x, batch)
    torch.testing.assert_close(dynamic[:3], sagnn(x[:3])[1])
    torch.testing.assert_close(dynamic[3:], sagnn(x[3:])[1])
    perm = torch.tensor([2, 0, 1])
    torch.testing.assert_close(sagnn(x[:3][perm])[1], dynamic[:3][perm])


def test_interaction_predictor_scores_are_group_means_of_squared_differences() -> None:
    """score_b = mean_{i in b} w . (dyn_i - static_i)^2 + c; with no batch vector the one
    score has shape [1, 1]. A group id with no genes (id 1 in [0, 0, 2, 2]) scores 0.
    """
    torch.manual_seed(0)
    predictor = GeneInteractionPredictor(HIDDEN, num_heads=2, num_layers=1, dropout=0.0)
    x = torch.randn(4, HIDDEN)
    batch = torch.tensor([0, 0, 2, 2])
    scores = predictor(x, batch)
    static, dynamic = predictor.hyper_sagnn(x, batch)
    gene_scores = predictor.prediction_layer((dynamic - static) ** 2).squeeze(-1)
    expected = torch.stack(
        [gene_scores[:2].mean(), torch.tensor(0.0), gene_scores[2:].mean()]
    )
    torch.testing.assert_close(scores, expected.unsqueeze(-1))
    single = predictor(x[:2])
    assert single.shape == (1, 1)
    torch.testing.assert_close(single[0, 0], scores[0, 0])


# ---------------------------------------------------------------- the full model


def test_tiny_model_parameter_count_matches_the_hand_derivation() -> None:
    """Component counts from the module docstring; total 1120."""
    model = _tiny()
    assert model.graph_names == ["physical", "regulatory"]
    assert model.num_parameters == {
        "gene_embedding": 32,
        "preprocessor": 160,
        "convs": 322,
        "gene_interaction_predictor": 370,
        "global_aggregator": 113,
        "global_interaction_predictor": 81,
        "gate_mlp": 42,
        "total": 1120,
    }
    assert model.num_parameters["total"] == sum(p.numel() for p in model.parameters())


def test_concat_mode_drops_the_gate_mlp_from_the_count() -> None:
    """combination_method "concat" builds no gate: 1120 - 42 = 1078."""
    model = _tiny(
        local_predictor_config={
            "combination_method": "concat",
            "num_heads": 2,
            "num_attention_layers": 1,
        }
    )
    assert model.gate_mlp is None
    assert "gate_mlp" not in model.num_parameters
    assert model.num_parameters["total"] == 1078


def test_init_zeroes_every_linear_bias_and_misses_the_gatv2_attributes() -> None:
    """``_init_weights`` sets every nn.Linear bias to 0 and nn.LayerNorm to (1, 0).

    Finding: its GATv2 branch tests ``lin_src``/``lin_dst``/``att_src``/``att_dst``
    (hetero_cell_bipartite_dango_gi.py:877-893), which are GATConv names; GATv2Conv has
    ``lin_l``, ``lin_r`` and ``att``, so the branch never re-initializes anything and the
    default encoder keeps PyG's own init.
    """
    model = _tiny(gene_encoder_config={"encoder_type": "gatv2"})
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert all(torch.equal(m.bias, torch.zeros_like(m.bias)) for m in linears)
    norm = model.preprocessor.mlp[1]
    assert isinstance(norm, nn.LayerNorm)
    assert torch.equal(norm.weight, torch.ones(HIDDEN))
    gat = [m for m in model.modules() if isinstance(m, GATv2Conv)]
    assert len(gat) == 2
    for conv in gat:
        names = {n.split(".")[0] for n, _ in conv.named_parameters()}
        assert names == {"att", "bias", "lin_l", "lin_r"}
        assert not any(hasattr(conv, a) for a in ["lin_src", "lin_dst", "att_src"])


def test_forward_outputs_satisfy_the_gating_and_difference_identities() -> None:
    """Two samples perturbing {0, 1} and {2, 3} (eval mode):

    * z_p = z_w - z_i exactly, z_w of shape [1, 8], z_i of shape [2, 8]
    * pert_gene_embs are rows [0, 1, 2, 3] of the wildtype encoder output
    * local = predictor(pert_gene_embs, [0, 0, 1, 1]); global = MLP(z_p)
    * gate weights are a softmax (rows sum to 1) of gate_mlp(cat(global, local)) and the
      prediction is sum(cat(global, local) * gate) per row
    """
    model = _tiny().eval()
    cell_graph, batch = _cell_graph(), _batch([[0, 1], [2, 3]])
    with torch.no_grad():
        pred, out = model(cell_graph, batch)
        z_w_nodes = model.forward_single(cell_graph)
        local = model.gene_interaction_predictor(z_w_nodes, torch.tensor([0, 0, 1, 1]))
        stack = torch.cat([out["global_interaction"], out["local_interaction"]], dim=1)
        gate = torch.softmax(model.gate_mlp(stack), dim=1)  # type: ignore[misc]
    assert pred.shape == (2, 1) and torch.isfinite(pred).all()
    assert out["z_w"].shape == (1, HIDDEN) and out["z_i"].shape == (2, HIDDEN)
    assert torch.equal(out["z_p"], out["z_w"].expand(2, -1) - out["z_i"])
    torch.testing.assert_close(out["pert_gene_embs"], z_w_nodes[[0, 1, 2, 3]])
    torch.testing.assert_close(out["local_interaction"], local)
    torch.testing.assert_close(
        out["global_interaction"], model.global_interaction_predictor(out["z_p"])
    )
    torch.testing.assert_close(out["gate_weights"], gate)
    torch.testing.assert_close(out["gate_weights"].sum(1), torch.ones(2))
    torch.testing.assert_close(pred, (stack * gate).sum(1, keepdim=True))
    assert out["graph_attention_weights"] == []


def test_concat_mode_averages_global_and_local_with_fixed_half_gates() -> None:
    """Combination "concat": prediction = 0.5 * global + 0.5 * local, gates all 0.5."""
    model = _tiny(
        local_predictor_config={
            "combination_method": "concat",
            "num_heads": 2,
            "num_attention_layers": 1,
        }
    ).eval()
    with torch.no_grad():
        pred, out = model(_cell_graph(), _batch([[0, 1], [2, 3]]))
    torch.testing.assert_close(
        pred, 0.5 * out["global_interaction"] + 0.5 * out["local_interaction"]
    )
    assert torch.equal(out["gate_weights"], torch.full((2, 2), 0.5))


def test_an_unknown_combination_builds_and_fails_only_at_forward() -> None:
    """Finding: ``combination_method`` is not validated in ``__init__`` (any value other
    than "gating" just skips the gate MLP, line 847-855); the ValueError comes at the
    end of the first forward (line 1112), after the whole encoder has run.
    """
    model = _tiny(local_predictor_config={"combination_method": "product"})
    assert model.gate_mlp is None
    with pytest.raises(ValueError, match="Unknown combination method: product"):
        model(_cell_graph(), _batch([[0, 1], [2, 3]]))


def test_forward_is_seeded_and_every_parameter_receives_gradient() -> None:
    """Same seed -> identical predictions (train mode, dropout 0); backward from the
    summed prediction gives every one of the model's parameters a finite gradient.
    """
    cell_graph, batch = _cell_graph(), _batch([[0, 1], [2, 3]])
    model = _tiny(seed=3)
    pred, _ = model(cell_graph, batch)
    again, _ = _tiny(seed=3)(cell_graph, batch)
    assert torch.equal(pred, again)
    pred.sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_prediction_is_invariant_to_relabeling_the_genes() -> None:
    """Relabel gene g as pi(g) with pi = [2, 0, 3, 1]: permute the embedding rows, map
    every wildtype edge and every perturbation through pi, rebuild the samples. The
    predictions and gates are unchanged (eval mode) and the perturbed-gene embeddings
    are the same rows in the same order.
    """
    pi = [2, 0, 3, 1]
    model = _tiny().eval()
    relabeled = _tiny().eval()
    inverse = torch.tensor([pi.index(g) for g in range(N_GENES)])
    with torch.no_grad():
        relabeled.gene_embedding.weight.copy_(model.gene_embedding.weight[inverse])
    perts = [[0, 1], [2, 3]]
    mapped_edges = {
        name: [(pi[s], pi[t]) for s, t in pairs]
        for name, pairs in WILDTYPE_EDGES.items()
    }
    with torch.no_grad():
        pred, out = model(_cell_graph(), _batch(perts))
        pred_pi, out_pi = relabeled(
            _cell_graph(mapped_edges),
            _batch([[pi[g] for g in p] for p in perts], mapped_edges),
        )
    torch.testing.assert_close(pred_pi, pred)
    torch.testing.assert_close(out_pi["gate_weights"], out["gate_weights"])
    torch.testing.assert_close(out_pi["pert_gene_embs"], out["pert_gene_embs"])


def test_eval_prediction_depends_on_the_other_samples_under_layer_norm() -> None:
    """Finding: ``AttentionConvWrapper`` calls PyG ``LayerNorm`` (default mode "graph")
    without a batch vector (hetero_cell_bipartite_dango_gi.py:654-655), so one mean and
    variance are taken over every node of every sample in the batch. Sample [0, 1]
    therefore gets a different eval-mode prediction next to [2, 3] than next to [1, 2].
    With norm "batch" the running statistics are used in eval and the prediction for
    [0, 1] is the same in both batches.
    """
    for norm, depends in [("layer", True), ("batch", False)]:
        model = _tiny(norm=norm).eval()
        with torch.no_grad():
            first, _ = model(_cell_graph(), _batch([[0, 1], [2, 3]]))
            second, _ = model(_cell_graph(), _batch([[0, 1], [1, 2]]))
        same = torch.allclose(first[0], second[0], atol=1e-6, rtol=0.0)
        assert same is not depends, norm


def test_the_three_local_assignment_paths_agree_when_every_sample_has_perturbations() -> (
    None
):
    """``perturbation_indices_ptr`` (from follow_batch) and a stored
    ``perturbation_indices_batch`` give identical [2, 1] local scores.

    Finding: with neither field the local head pools all four perturbed genes of both
    samples into ONE score, placed in sample 0, and sample 1's local score is 0
    (hetero_cell_bipartite_dango_gi.py:1019-1051).
    """
    model = _tiny().eval()
    cell_graph = _cell_graph()
    with_ptr = _batch([[0, 1], [2, 3]])
    plain = Batch.from_data_list([_sample([0, 1]), _sample([2, 3])])
    assert not hasattr(plain["gene"], "perturbation_indices_ptr")
    with_batch = Batch.from_data_list([_sample([0, 1]), _sample([2, 3])])
    with_batch["gene"].perturbation_indices_batch = torch.tensor([0, 0, 1, 1])
    with torch.no_grad():
        _, out_ptr = model(cell_graph, with_ptr)
        _, out_batch = model(cell_graph, with_batch)
        _, out_none = model(cell_graph, plain)
        pooled = model.gene_interaction_predictor(out_ptr["pert_gene_embs"], None)
    torch.testing.assert_close(
        out_batch["local_interaction"], out_ptr["local_interaction"]
    )
    assert out_none["local_interaction"].shape == (2, 1)
    torch.testing.assert_close(out_none["local_interaction"][0], pooled[0])
    assert out_none["local_interaction"][1].item() == 0.0


def test_a_trailing_sample_without_perturbations_misplaces_the_local_scores() -> None:
    """Finding: when the local head returns fewer rows than samples, the re-expansion
    (hetero_cell_bipartite_dango_gi.py:1043-1050) writes local row i at
    ``batch_assign[i]``, the sample of the i-th perturbed GENE, not sample i. Samples
    {0, 1}, {2}, {} give batch_assign [0, 0, 1] and 2 local rows: row 0 goes to sample
    0, then row 1 (sample 1's score) overwrites sample 0, and samples 1 and 2 get 0.
    """
    model = _tiny().eval()
    batch = _batch([[0, 1], [2], []])
    assert batch["gene"].perturbation_indices_ptr.tolist() == [0, 2, 3, 3]
    with torch.no_grad():
        _, out = model(_cell_graph(), batch)
        per_sample = model.gene_interaction_predictor(
            out["pert_gene_embs"], torch.tensor([0, 0, 1])
        )
    assert per_sample.shape == (2, 1)
    local = out["local_interaction"]
    assert local.shape == (3, 1)
    torch.testing.assert_close(local[0], per_sample[1])
    assert local[1:].flatten().tolist() == [0.0, 0.0]


def test_a_batch_without_pert_mask_indexes_the_embedding_with_batch_node_ids() -> None:
    """Finding: without ``pert_mask`` the batched path embeds ``arange(num_nodes)`` over
    the WHOLE batch (hetero_cell_bipartite_dango_gi.py:916-919), so two full samples
    (8 nodes) index past the 4-row embedding and raise IndexError.
    """
    model = _tiny()
    samples = [_sample([]), _sample([])]
    for sample in samples:
        del sample["gene"].pert_mask
    batch = Batch.from_data_list(samples)
    batch["gene"].perturbation_indices = torch.tensor([0, 1])
    with pytest.raises(IndexError, match="index out of range"):
        model(_cell_graph(), batch)


def test_default_gatv2_cross_attention_model_reports_attention_per_layer() -> None:
    """Defaults (GATv2, cross_attention): one {"gene": [n, 2, 2]} weight dict per conv
    layer, stored from the LAST ``forward_single`` call (the perturbed batch, whose two
    samples {0} and {2, 3} keep 3 + 2 = 5 nodes, not the wildtype's 4). Each row is a
    softmax over the 2 graphs.
    """
    torch.manual_seed(0)
    model = GeneInteractionDango(
        N_GENES,
        HIDDEN,
        num_layers=2,
        gene_multigraph=_multigraph(),
        dropout=0.0,
        local_predictor_config={"num_heads": 2, "num_attention_layers": 1},
    ).eval()
    with torch.no_grad():
        pred, out = model(_cell_graph(), _batch([[0], [2, 3]]))
    weights = out["graph_attention_weights"]
    assert [sorted(w) for w in weights] == [["gene"], ["gene"]]
    for layer in weights:
        assert layer["gene"].shape == (5, 2, 2)
        torch.testing.assert_close(layer["gene"].sum(-1), torch.ones(5, 2))
    assert pred.shape == (2, 1) and torch.isfinite(pred).all()


@pytest.mark.parametrize(
    ("poison", "message"),
    [
        ("gene_embedding.weight", r"wildtype embeddings \(z_w\)"),
        (
            "global_aggregator.gate_nn.3.bias",
            r"global wildtype embeddings \(z_w_global\)",
        ),
        (
            "gene_interaction_predictor.prediction_layer.bias",
            "local interaction predictions",
        ),
        ("global_interaction_predictor.3.bias", "global interaction predictions"),
        ("gate_mlp.3.bias", "gate logits"),
    ],
)
def test_a_nan_parameter_is_named_by_the_first_stage_it_reaches(
    poison: str, message: str
) -> None:
    """Setting one parameter to NaN raises the RuntimeError of the first checked stage
    downstream of it, so the message localizes the fault.
    """
    model = _tiny().eval()
    with torch.no_grad():
        dict(model.named_parameters())[poison].fill_(float("nan"))
    with pytest.raises(RuntimeError, match=f"^NaN or inf detected in {message}$"):
        model(_cell_graph(), _batch([[0, 1], [2, 3]]))


# ---------------------------------------------------------------- Phase 17: config flags


def _grad_model(encoder: str, aggregation: str) -> GeneInteractionDango:
    return _tiny(
        gene_encoder_config={
            "encoder_type": encoder,
            "graph_aggregation_method": aggregation,
            "heads": 2,
            "graph_aggregation_config": {"num_heads": 2},
        }
    )


@pytest.mark.parametrize(
    ("encoder", "aggregation", "total"),
    [
        ("gin", "sum", 1120),
        ("gin", "mean", 1120),
        ("gin", "cross_attention", 1120 + 304),
        ("gin", "pairwise_interaction", 1120 + 645),
        ("gatv2", "sum", 1120 + 30),
        ("gatv2", "mean", 1120 + 30),
        ("gatv2", "cross_attention", 1120 + 30 + 304),
        ("gatv2", "pairwise_interaction", 1120 + 30 + 645),
    ],
)
def test_encoder_and_aggregation_flags_add_their_exact_parameter_counts(
    encoder: str, aggregation: str, total: int
) -> None:
    """Totals from the module docstring's 1120 plus each flag's own modules.

    * cross_attention adds one ``SelfAttentionGraphAggregation`` per conv layer: 304
      (MultiheadAttention 4d^2 + 4d = 288 plus 2 graph embeddings of d = 16).
    * pairwise_interaction adds one ``PairwiseGraphAggregation``: 3 pair MLPs of 208
      plus the attention head 21 = 645. sum and mean add nothing.
    * gatv2 with heads 2 replaces each GIN conv (161 with its LayerNorm) by
      GATv2Conv(8, 4, heads=2): lin_l 72 + lin_r 72 + att 2 * 4 = 8 + bias 8 = 160, plus
      the wrapper LayerNorm 16 = 176, so +15 per graph and +30 for two graphs.

    One backward from the summed prediction reaches every parameter (no ``None``
    gradient) and every gradient is finite.
    """
    model = _grad_model(encoder, aggregation)
    assert model.num_parameters["total"] == total
    pred, _ = model(_cell_graph(), _batch([[0, 1], [2, 3]]))
    pred.sum().backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    assert [n for n, g in grads.items() if g is None] == []
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())


def test_depth_flags_add_one_conv_block_or_one_attention_block_each() -> None:
    """``num_layers`` 2 adds a second hetero conv layer: two GIN wrappers of 161 = 322,
    total 1120 + 322 = 1442. ``num_attention_layers`` 2 adds one HyperSAGNN layer, q/k/v/
    out Linear(8, 8) 4 * 72 = 288 plus its ReZero beta 1 = 289, total 1409, with every
    beta initialized to exactly 0.01.
    """
    deep_conv = _tiny(num_layers=2)
    assert len(deep_conv.convs) == 2
    assert deep_conv.num_parameters["convs"] == 644
    assert deep_conv.num_parameters["total"] == 1442
    deep_head = _tiny(
        local_predictor_config={"num_heads": 2, "num_attention_layers": 2}
    )
    assert deep_head.num_parameters["gene_interaction_predictor"] == 370 + 289
    assert deep_head.num_parameters["total"] == 1409
    betas = deep_head.gene_interaction_predictor.hyper_sagnn.beta_params
    assert [b.item() for b in betas] == [pytest.approx(0.01)] * 2


def test_single_candidate_softmaxes_give_exactly_zero_gradient_to_their_logits() -> (
    None
):
    """Two masked-softmax identities.

    HyperSAGNN masks the diagonal, so in a two-gene sample each gene attends to the
    other with weight exactly 1 whatever q and k are: q_proj and k_proj get a gradient
    of exactly zero. A three-gene sample {0, 1, 2} has two candidates per gene and the
    same parameters get a nonzero gradient.

    GATv2 (no self loops) normalizes over each node's incoming edges. In the wildtype
    and both samples every node has in-degree at most 1, so the attention vector ``att``
    and the target transform ``lin_r``, which only enter the logits, get exactly zero
    gradient. Adding physical edge 3 -> 1 gives node 1 two in-edges in the wildtype and
    both gradients become nonzero.
    """
    qk = [
        "gene_interaction_predictor.hyper_sagnn.attention_layers.0.q_proj.weight",
        "gene_interaction_predictor.hyper_sagnn.attention_layers.0.k_proj.weight",
    ]
    gat = [
        "convs.0.convs.('gene', 'physical', 'gene').conv.att",
        "convs.0.convs.('gene', 'physical', 'gene').conv.lin_r.weight",
    ]

    def grad_sums(
        model: GeneInteractionDango, cell_graph: HeteroData, batch: Batch
    ) -> dict[str, float]:
        pred, _ = model(cell_graph, batch)
        pred.sum().backward()
        params = dict(model.named_parameters())
        sums = {}
        for name in qk + gat:
            grad = params[name].grad
            assert grad is not None
            sums[name] = float(grad.abs().sum())
        return sums

    pairs = grad_sums(
        _grad_model("gatv2", "sum"), _cell_graph(), _batch([[0, 1], [2, 3]])
    )
    assert pairs == dict.fromkeys(qk + gat, 0.0)

    triple = grad_sums(_grad_model("gatv2", "sum"), _cell_graph(), _batch([[0, 1, 2]]))
    assert all(triple[name] > 0.0 for name in qk)

    fan_in = {**WILDTYPE_EDGES, "physical": [*WILDTYPE_EDGES["physical"], (3, 1)]}
    merged = grad_sums(
        _grad_model("gatv2", "sum"), _cell_graph(fan_in), _batch([[0, 1], [2, 3]])
    )
    assert all(merged[name] > 0.0 for name in gat)


def test_permuting_the_genotypes_in_a_batch_permutes_the_predictions() -> None:
    """Samples are independent apart from the shared LayerNorm statistics, which are a
    mean and variance over all nodes and so do not depend on sample order. Reordering
    the genotypes [{0, 1}, {2, 3}, {1}] as [{1}, {0, 1}, {2, 3}] reorders predictions,
    gate weights and z_i rows the same way (eval mode, default layer norm).
    """
    model = _tiny().eval()
    perts = [[0, 1], [2, 3], [1]]
    order = [2, 0, 1]
    with torch.no_grad():
        pred, out = model(_cell_graph(), _batch(perts))
        pred_perm, out_perm = model(_cell_graph(), _batch([perts[i] for i in order]))
    torch.testing.assert_close(pred_perm, pred[order])
    torch.testing.assert_close(out_perm["gate_weights"], out["gate_weights"][order])
    torch.testing.assert_close(out_perm["z_i"], out["z_i"][order])
    torch.testing.assert_close(out_perm["z_w"], out["z_w"])


def test_shipped_006_encoder_flags_reach_the_modules_they_name() -> None:
    """The 006 config sets ``activation: "gelu"``, ``graph_aggregation_config:
    {aggregation_norm: null, dropout: 0.0}`` next to the model ``dropout``.

    Contract (issue #540): "gelu" builds GELU in the preprocessor and in every conv
    wrapper; the aggregation config's own ``dropout`` (0.0) wins over the model dropout
    (0.25) in the MultiheadAttention, and without its own key the aggregation inherits
    the model dropout. ``aggregation_norm`` null builds no norm: the cross_attention
    model has 1424 parameters, the count it had when the key was ignored, and the
    config's "layer" it shipped until 2026.10.01 is now refused (tested below).
    """
    encoder = {
        "encoder_type": "gin",
        "graph_aggregation_method": "cross_attention",
        "graph_aggregation_config": {
            "num_heads": 2,
            "aggregation_norm": None,
            "dropout": 0.0,
        },
    }
    model = _tiny(dropout=0.25, activation="gelu", gene_encoder_config=encoder)
    assert model.aggregation_norm is None
    wrappers = [m for m in model.modules() if isinstance(m, AttentionConvWrapper)]
    assert [type(m.act) for m in [model.preprocessor, *wrappers]] == [nn.GELU] * 3
    assert model.num_parameters["total"] == 1424
    aggregator = model.convs[0].aggregator
    assert isinstance(aggregator, SelfAttentionGraphAggregation)
    assert aggregator.multihead_attn.dropout == 0.0

    inherit = {**encoder, "graph_aggregation_config": {"num_heads": 2}}
    inherited = _tiny(dropout=0.25, gene_encoder_config=inherit).convs[0].aggregator
    assert isinstance(inherited, SelfAttentionGraphAggregation)
    assert inherited.multihead_attn.dropout == 0.25


PAIRWISE_AGGREGATOR_KEYS = {
    "convs.0.aggregator.attention.0.bias",
    "convs.0.aggregator.attention.0.weight",
    "convs.0.aggregator.attention.2.bias",
    "convs.0.aggregator.attention.2.weight",
    *(
        f"convs.0.aggregator.interaction_mlps.{pair}.{layer}.{name}"
        for pair in (
            "physical_physical",
            "physical_regulatory",
            "regulatory_regulatory",
        )
        for layer in (0, 3)
        for name in ("bias", "weight")
    ),
}


def _encoder(method: str, **aggregation: Any) -> dict[str, Any]:
    return {
        "encoder_type": "gin",
        "graph_aggregation_method": method,
        "graph_aggregation_config": aggregation,
    }


def _refusal(value: object) -> str:
    """The exact refusal message for ``value``, anchored for ``pytest.raises``."""
    message = (
        f"aggregation_norm={value!r} is not implemented in "
        "torchcell.models.hetero_cell_bipartite_dango_gi: its graph aggregators "
        "build no normalization layer. Set aggregation_norm to null, or use "
        "torchcell.models.hetero_cell_bipartite_dango_gi_lazy, whose "
        "pairwise_interaction aggregator builds it."
    )
    return f"^{re.escape(message)}$"


@pytest.mark.parametrize(
    ("method", "total", "aggregator_keys"),
    [("sum", 1120, set()), ("pairwise_interaction", 1765, PAIRWISE_AGGREGATOR_KEYS)],
)
def test_null_aggregation_norm_builds_exactly_the_module_main_built(
    method: str, total: int, aggregator_keys: set[str]
) -> None:
    """Contract (issue #540): ``aggregation_norm`` null, and the key absent, build the
    module of main (8774914fe) exactly: same parameter count, same state_dict keys,
    and no norm key under ``convs.0.aggregator``.

    Counts on main: "sum" 1120 (module docstring) and "pairwise_interaction"
    1120 + 645, the 645 being three pair MLPs of Linear(16, 8) 136 + Linear(8, 8) 72
    (624) and the attention head Linear(8, 2) 18 + Linear(2, 1) 3 (21). The pairwise
    model's keys are the "sum" model's 56 plus the 16 aggregator keys listed in
    ``PAIRWISE_AGGREGATOR_KEYS`` (72 in all). Weights built under one seed are equal.
    """
    absent = _tiny(gene_encoder_config=_encoder(method))
    null = _tiny(gene_encoder_config=_encoder(method, aggregation_norm=None))
    base_keys = set(_tiny(gene_encoder_config=_encoder("sum")).state_dict())
    assert len(base_keys) == 56
    for model in (absent, null):
        assert model.aggregation_norm is None
        assert sum(p.numel() for p in model.parameters()) == total
        assert set(model.state_dict()) == base_keys | aggregator_keys
        assert len(model.state_dict()) == 56 + len(aggregator_keys)
    for key, tensor in absent.state_dict().items():
        torch.testing.assert_close(null.state_dict()[key], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("value", ["layer", "batch", "none"])
@pytest.mark.parametrize("method", ["sum", "cross_attention", "pairwise_interaction"])
def test_a_non_null_aggregation_norm_is_refused_by_name(
    method: str, value: str
) -> None:
    """Contract (issue #540): any value other than null, including the string "none",
    raises ``AggregationNormNotImplementedError`` (a ValueError) with the exact message
    naming the value and the ``_lazy`` module that builds the norm, for every
    aggregation method, from the model and from ``HeteroConvAggregator`` given the
    value either as its argument or inside its ``aggregation_config``.
    The caller's config dict is not mutated by the model reading the key.
    """
    encoder = _encoder(method, aggregation_norm=value)
    with pytest.raises(AggregationNormNotImplementedError, match=_refusal(value)):
        _tiny(gene_encoder_config=encoder)
    assert encoder["graph_aggregation_config"] == {"aggregation_norm": value}
    assert issubclass(AggregationNormNotImplementedError, ValueError)

    conv = AttentionConvWrapper(GINConv(nn.Linear(HIDDEN, HIDDEN)), HIDDEN)
    with pytest.raises(AggregationNormNotImplementedError, match=_refusal(value)):
        HeteroConvAggregator(
            {PHYS: conv}, HIDDEN, aggregation_method=method, aggregation_norm=value
        )
    with pytest.raises(AggregationNormNotImplementedError, match=_refusal(value)):
        HeteroConvAggregator(
            {PHYS: conv},
            HIDDEN,
            aggregation_method=method,
            aggregation_config={"aggregation_norm": value},
        )


# ---------------------------------------------------------------- Phase 17: wrapper edges


def test_wrapper_without_norm_is_act_of_proj_of_conv() -> None:
    """Norm None skips the norm step: out = relu(W_proj conv(x, e) + b) exactly."""
    torch.manual_seed(0)
    wrapper = AttentionConvWrapper(GCNConv(4, 3), 6, norm=None, activation="relu")
    wrapper.eval()
    x = torch.randn(5, 4)
    edges = _edge_index([(0, 1), (1, 2), (3, 4)])
    expected = torch.relu(wrapper.proj(wrapper.conv(x, edges)))
    assert torch.equal(wrapper(x, edges), expected)


def test_wrapper_around_a_gin_mlp_with_no_linear_is_refused_by_name() -> None:
    """Contract (issue #540): a GIN ``nn`` Sequential with no nn.Linear has no output
    width to read, so the wrapper raises a ValueError naming GINConv instead of an
    UnboundLocalError. A nested Sequential still finds its last Linear (width 5 ->
    Linear(5, 8) projection).
    """
    with pytest.raises(
        ValueError,
        match=re.escape(
            "AttentionConvWrapper cannot infer the output width of GINConv: its "
            "nn.Sequential contains no nn.Linear"
        ),
    ):
        AttentionConvWrapper(GINConv(nn.Sequential(nn.ReLU())), 8)
    nested = GINConv(nn.Sequential(nn.Sequential(nn.Linear(8, 5)), nn.ReLU()))
    proj = AttentionConvWrapper(nested, 8).proj
    assert isinstance(proj, nn.Linear) and (proj.in_features, proj.out_features) == (
        5,
        8,
    )


# ---------------------------------------------------------------- Phase 17: non-finite


@pytest.mark.parametrize(
    ("poisons", "combination", "message"),
    [
        (
            {"gene_embedding.weight": float("inf")},
            "gating",
            r"wildtype embeddings \(z_w\)",
        ),
        (
            {"global_aggregator.transform_nn.0.bias": float("inf")},
            "gating",
            r"global wildtype embeddings \(z_w_global\)",
        ),
        ({"gate_mlp.3.bias": float("inf")}, "gating", "gate logits"),
        (
            {"global_interaction_predictor.3.bias": float("inf")},
            "gating",
            "global interaction predictions",
        ),
        (
            {"global_interaction_predictor.3.bias": float("inf")},
            "concat",
            "global interaction predictions",
        ),
        (
            {"gene_interaction_predictor.prediction_layer.bias": float("-inf")},
            "concat",
            "local interaction predictions",
        ),
        (
            {
                "global_interaction_predictor.3.bias": float("inf"),
                "gene_interaction_predictor.prediction_layer.bias": float("-inf"),
            },
            "concat",
            "local interaction predictions",
        ),
    ],
)
def test_an_infinite_parameter_is_named_by_the_first_stage_it_reaches(
    poisons: dict[str, float], combination: str, message: str
) -> None:
    """Contract (issue #540): every stage check is ``torch.isfinite``, so +/-inf is
    caught where it first appears, with the same RuntimeError shape as NaN:

    * gene embedding +inf: z_w (the first check).
    * transform bias +inf: z_w_global (a softmax-weighted sum of +inf rows is +inf); it
      used to pass until z_p = inf - inf = NaN.
    * gate bias +inf: the gate logits [inf, inf]; it used to pass until the softmax.
    * a +inf global head: the global interaction check under both combinations; under
      concat it used to return +inf with no error.
    * a -inf local head: the local interaction check, which runs before the global one,
      so it names the local stage even when the global head is +inf as well.
    """
    model = _tiny(
        local_predictor_config={
            "combination_method": combination,
            "num_heads": 2,
            "num_attention_layers": 1,
        }
    ).eval()
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in poisons.items():
            params[name].fill_(value)
    with pytest.raises(RuntimeError, match=f"^NaN or inf detected in {message}$"):
        model(_cell_graph(), _batch([[0, 1], [2, 3]]))


# ---------------------------------------------------------------- Phase 17: the script


MAIN_PERTS = [[0, 1], [2, 3], [1, 3]]
MAIN_Y = [0.1, -0.2, 0.3]


def _main_batch() -> Batch:
    batch = _batch(MAIN_PERTS)
    batch["gene"].phenotype_values = torch.tensor(MAIN_Y)
    return batch


def _main_cfg(
    loss: str,
    epochs: int = 4,
    plot_every: int = 2,
    lr: float = 1e-3,
    scheduler: dict[str, Any] | None = None,
    seed: int = 0,
) -> DictConfig:
    """The tiny model of ``_tiny`` expressed as the script's config (CPU accelerator)."""
    regression_task: dict[str, Any] = {
        "loss": loss,
        "lambda_dist": 0.1,
        "lambda_supcr": 0.001,
        "loss_config": {
            "min_samples_for_dist": 2,
            "min_samples_for_supcr": 2,
            "min_samples_for_wasserstein": 2,
            "buffer_size": 8,
            "use_ddp_gather": False,
        },
        "optimizer": {"lr": lr, "weight_decay": 0.0},
        "clip_grad_norm": True,
        "clip_grad_norm_max_norm": 10.0,
        "plot_every_n_epochs": plot_every,
    }
    if scheduler is not None:
        regression_task["lr_scheduler"] = scheduler
    return OmegaConf.create(
        {
            "seed": seed,
            "trainer": {"accelerator": "cpu", "max_epochs": epochs},
            "data_module": {"batch_size": 3, "num_workers": 0},
            "cell_dataset": {
                "graphs": ["physical", "regulatory"],
                "learnable_embedding_input_channels": HIDDEN,
            },
            "model": {
                "gene_num": N_GENES,
                "hidden_channels": HIDDEN,
                "num_layers": 1,
                "dropout": 0.0,
                "norm": "layer",
                "activation": "relu",
                "gene_encoder_config": {
                    "encoder_type": "gin",
                    "graph_aggregation_method": "sum",
                },
                "local_predictor_config": {
                    "num_heads": 2,
                    "num_attention_layers": 1,
                    "combination_method": "gating",
                },
            },
            "regression_task": regression_task,
        }
    )


@pytest.fixture
def fake_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Serve ``main`` the four-gene fixture without the genome, Neo4j or disk writes.

    ``main`` imports its helpers at call time, so the module attributes are replaced:
    ``load_sample_data_batch`` returns the wildtype graph and a 3-sample batch,
    the genome and graph classes record their keyword arguments,
    ``build_gene_multigraph`` returns the two-graph fixture, ``timestamp`` is "TS",
    ``load_dotenv`` is a no-op and ``plt.savefig`` records the file name instead of
    rendering. ``ASSET_IMAGES_DIR`` is ``tmp_path``.
    """
    import matplotlib.pyplot as plt

    import torchcell.graph.graph as graph_module
    import torchcell.sequence.genome.scerevisiae.s288c as s288c_module
    import torchcell.timestamp as timestamp_module

    calls: dict[str, Any] = {"saved": []}
    loader = types.ModuleType("torchcell.scratch.load_batch_005")

    def load_sample_data_batch(**kwargs: Any) -> tuple[Any, Batch, None, None]:
        calls["loader"] = kwargs
        return (
            types.SimpleNamespace(cell_graph=_cell_graph()),
            _main_batch(),
            None,
            None,
        )

    def genome(**kwargs: Any) -> str:
        calls["genome"] = kwargs
        return "genome"

    def graph(**kwargs: Any) -> str:
        calls["graph"] = kwargs
        return "graph"

    def build_gene_multigraph(graph: Any, graph_names: list[str]) -> GeneMultiGraph:
        calls["multigraph"] = (graph, list(graph_names))
        return _multigraph()

    loader.__dict__["load_sample_data_batch"] = load_sample_data_batch
    monkeypatch.setitem(sys.modules, "torchcell.scratch.load_batch_005", loader)
    monkeypatch.setattr(s288c_module, "SCerevisiaeGenome", genome)
    monkeypatch.setattr(graph_module, "SCerevisiaeGraph", graph)
    monkeypatch.setattr(graph_module, "build_gene_multigraph", build_gene_multigraph)
    monkeypatch.setattr(timestamp_module, "timestamp", lambda: "TS")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(
        plt, "savefig", lambda path, **_: calls["saved"].append(osp.basename(path))
    )
    monkeypatch.setenv("ASSET_IMAGES_DIR", str(tmp_path))
    return calls


def test_main_at_lr_zero_plots_on_schedule_and_reports_the_untrained_model(
    fake_main: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """LogCosh, 4 epochs, ``plot_every_n_epochs`` 2, lr 0, config ``seed`` 0.

    Contract (issue #540): ``main`` seeds torch, numpy and random from the config's
    ``seed`` (printing ``seed: 0``) and the model build is the first RNG consumer, so
    even with the caller's RNG left at seed 123 the printed metrics equal a
    ``_tiny(seed=0)`` reference model.

    Plot schedule: ``epoch % 2 == 0 or epoch == 3`` for epochs 0..3 fires at 0, 2 and 3,
    saved as ``training_epoch_0001``, ``_0003``, ``_0004``, then ``final_results_TS``;
    the epoch banner prints only for those epochs. The plot directory is
    ``hetero_cell_bipartite_dango_gi_training_TS`` and stays empty (``savefig`` records).

    The genome is built with ``overwrite=False`` under ``$DATA_ROOT`` (memory rule: never
    ``overwrite=True``), the graph from that genome, and the multigraph from the config's
    graph names.

    At lr 0 AdamW moves nothing (step and decay both scale by lr), so the final eval
    forward is the forward of a fresh model built under the same seed: the printed MSE,
    MAE, RMSE and Pearson are those of that reference, and the printed final loss is
    its LogCosh. The model is the 1120-parameter ``_tiny`` configuration.
    """
    torch.manual_seed(123)
    dango_module.main(_main_cfg("logcosh", epochs=4, plot_every=2, lr=0.0, seed=0))
    out = capsys.readouterr().out
    assert "seed: 0" in out.splitlines()

    assert fake_main["saved"] == [
        "training_epoch_0001.png",
        "training_epoch_0003.png",
        "training_epoch_0004.png",
        "final_results_TS.png",
    ]
    assert [p.name for p in tmp_path.iterdir()] == [
        "hetero_cell_bipartite_dango_gi_training_TS"
    ]
    assert (
        list((tmp_path / "hetero_cell_bipartite_dango_gi_training_TS").iterdir()) == []
    )
    assert re.findall(r"^Epoch (\d+)/4$", out, flags=re.M) == ["1", "3", "4"]
    assert re.findall(r"^LR: (.*)$", out, flags=re.M) == ["0.00e+00"] * 3

    root = os.environ["DATA_ROOT"]
    assert fake_main["loader"] == {
        "batch_size": 3,
        "num_workers": 0,
        "config": "hetero_cell_bipartite",
        "is_dense": False,
    }
    assert fake_main["genome"] == {
        "genome_root": osp.join(root, "data/sgd/genome"),
        "go_root": osp.join(root, "data/go"),
        "overwrite": False,
    }
    assert fake_main["graph"] == {
        "sgd_root": osp.join(root, "data/sgd/genome"),
        "string_root": osp.join(root, "data/string"),
        "tflink_root": osp.join(root, "data/tflink"),
        "genome": "genome",
    }
    assert fake_main["multigraph"] == ("graph", ["physical", "regulatory"])
    assert "Parameter count: 1120\n" in out and "Using LogCosh loss\n" in out

    reference = _tiny(seed=0).eval()
    with torch.no_grad():
        pred, _ = reference(_cell_graph(), _main_batch())
    y = torch.tensor(MAIN_Y)
    pred_np, y_np = pred.squeeze().numpy(), y.numpy()
    mse = np.mean((pred_np - y_np) ** 2)
    expected = [
        f"Final Pearson Correlation: {np.corrcoef(pred_np, y_np)[0, 1]:.6f}",
        f"Final MSE: {mse:.6f}",
        f"Final MAE: {np.mean(np.abs(pred_np - y_np)):.6f}",
        f"Final RMSE: {np.sqrt(mse):.6f}",
        f"Final LOGCOSH Loss: {LogCoshLoss()(pred.squeeze(), y).item():.6f}",
    ]
    lines = out.splitlines()
    assert [line for line in expected if line not in lines] == []


def test_main_steps_the_warmup_scheduler_once_per_epoch_after_the_optimizer(
    fake_main: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """With ``lr_scheduler`` of type CosineAnnealingWarmupRestarts (4-step cycle, 1
    warmup step, max 1e-2, min 1e-4, ``cycle_mult`` and ``gamma`` absent), the printed
    LR after epochs 1, 3 and 4 is the lr of the same scheduler replayed on a dummy
    optimizer after 1, 3 and 4 steps; the banner reports the two absent keys at their
    1.0 defaults.
    """
    schedule = {
        "type": "CosineAnnealingWarmupRestarts",
        "first_cycle_steps": 4,
        "max_lr": 1e-2,
        "min_lr": 1e-4,
        "warmup_steps": 1,
    }
    dango_module.main(_main_cfg("logcosh", scheduler=schedule))
    out = capsys.readouterr().out

    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    replay = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=4,
        cycle_mult=1.0,
        max_lr=1e-2,
        min_lr=1e-4,
        warmup_steps=1,
        gamma=1.0,
    )
    lrs = []
    for _ in range(4):
        optimizer.step()
        replay.step()
        lrs.append(f"{optimizer.param_groups[0]['lr']:.2e}")
    assert re.findall(r"^LR: (.*)$", out, flags=re.M) == [lrs[0], lrs[2], lrs[3]]
    assert len(set(lrs)) > 1
    banner = out.split("Using CosineAnnealingWarmupRestarts scheduler with:\n")[1]
    assert banner.splitlines()[:6] == [
        "  - first_cycle_steps: 4",
        "  - cycle_mult: 1.0",
        "  - max_lr: 0.01",
        "  - min_lr: 0.0001",
        "  - warmup_steps: 1",
        "  - gamma: 1.0",
    ]


def test_main_refuses_an_unknown_loss_before_it_creates_the_plot_directory(
    fake_main: dict[str, Any], tmp_path: Path
) -> None:
    """The loss dispatch raises ``Unknown loss type: huber`` after the model is built
    and before the plot directory exists, so nothing is written or plotted.
    """
    with pytest.raises(ValueError, match="^Unknown loss type: huber$"):
        dango_module.main(_main_cfg("huber"))
    assert list(tmp_path.iterdir()) == []
    assert fake_main["saved"] == []


@pytest.mark.parametrize(
    ("loss", "name", "banner", "saved"),
    [
        (
            "icloss",
            "ICLoss",
            "Using ICLoss with lambda_dist=0.1, lambda_supcr=0.001",
            [
                "training_epoch_0001.png",
                "training_epoch_0002.png",
                "loss_components_evolution_TS.png",
                "final_results_TS.png",
            ],
        ),
        (
            "mle_dist_supcr",
            "MleDistSupCR",
            "Using MleDistSupCR with lambda_mse=1.0, lambda_dist=0.1, "
            "lambda_supcr=0.001",
            [
                "training_epoch_0001.png",
                "training_epoch_0002.png",
                "loss_components_evolution_TS.png",
                "final_results_TS.png",
            ],
        ),
    ],
)
def test_main_composite_losses_print_their_components_every_epoch(
    fake_main: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    loss: str,
    name: str,
    banner: str,
    saved: list[str],
) -> None:
    """ICLoss and MleDistSupCR take z_p as a third argument and return (loss, dict); the
    script prints one component line per epoch (2 of 2) and plots every epoch at
    ``plot_every_n_epochs`` 1.

    Contract (issue #540): every composite loss, not only "icloss", saves the final
    ``loss_components_evolution`` figure before ``final_results``.
    """
    cfg = _main_cfg(loss, epochs=2, plot_every=1)
    cfg.regression_task.is_weighted_phenotype_loss = True
    dango_module.main(cfg)
    out = capsys.readouterr().out
    assert banner in out.splitlines()
    component = rf"^  {name} components: mse=\d+\.\d{{4}}, dist=-?\d+\.\d{{4}}, supcr=-?\d+\.\d{{4}}$"
    assert len(re.findall(component, out, flags=re.M)) == 2
    assert fake_main["saved"] == saved


def test_main_with_the_shipped_wasserstein_loss_prints_its_own_components(
    fake_main: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """Contract (issue #540): "mle_wass_supcr", the loss the shipped
    ``hetero_cell_bipartite_dango_gi.yaml`` selects, trains. The component line reads
    the keys ``MleWassSupCR`` emits (``weighted_wasserstein``), labeled
    ``wasserstein=``, and it used to raise KeyError on ``weighted_dist`` in epoch 1.

    Epoch 1 is the first forward of the seed-0 model, so its line equals a fresh
    ``MleWassSupCR`` (same arguments as ``main`` builds from the config) applied to a
    ``_tiny(seed=0)`` forward in train mode (dropout 0). Two epochs print two lines and
    save both epoch plots, the components figure and the final figure.
    """
    dango_module.main(_main_cfg("mle_wass_supcr", epochs=2, plot_every=1, lr=0.0))
    out = capsys.readouterr().out
    assert (
        "Using MleWassSupCR with lambda_mse=1.0, lambda_wasserstein=0.1, "
        "lambda_supcr=0.001" in out.splitlines()
    )

    reference = _tiny(seed=0).train()
    batch = _main_batch()
    pred, reps = reference(_cell_graph(), batch)
    criterion = MleWassSupCR(
        lambda_mse=1.0,
        lambda_wasserstein=0.1,
        lambda_supcr=0.001,
        embedding_dim=HIDDEN,
        buffer_size=8,
        min_samples_for_wasserstein=2,
        min_samples_for_supcr=2,
        use_ddp_gather=False,
        weights=None,
        max_epochs=2,
    )
    y = batch["gene"].phenotype_values
    _, parts = criterion(pred.squeeze().unsqueeze(1), y.unsqueeze(1), reps["z_p"])
    first = (
        f"  MleWassSupCR components: mse={parts['mse_loss']:.4f}, "
        f"wasserstein={parts['weighted_wasserstein']:.4f}, "
        f"supcr={parts['weighted_supcr']:.4f}"
    )
    lines = re.findall(r"^  MleWassSupCR components: .*$", out, flags=re.M)
    assert len(lines) == 2 and lines[0] == first
    assert parts["weighted_wasserstein"] != 0.0
    assert fake_main["saved"] == [
        "training_epoch_0001.png",
        "training_epoch_0002.png",
        "loss_components_evolution_TS.png",
        "final_results_TS.png",
    ]


def _pairwise_main_cfg(value: str | None) -> DictConfig:
    cfg = _main_cfg("logcosh", epochs=1, plot_every=1, lr=0.0)
    cfg.model.gene_encoder_config = _encoder(
        "pairwise_interaction", aggregation_norm=value
    )
    return cfg


def test_main_passes_a_layer_aggregation_norm_through_to_the_refusal(
    fake_main: dict[str, Any], tmp_path: Path
) -> None:
    """Contract (issue #540): ``main`` hands ``model.gene_encoder_config.
    graph_aggregation_config.aggregation_norm`` to the model unchanged, so the key
    can never be dropped silently again: "layer" is refused by name, before the plot
    directory exists and before anything is saved.
    """
    with pytest.raises(AggregationNormNotImplementedError, match=_refusal("layer")):
        dango_module.main(_pairwise_main_cfg("layer"))
    assert list(tmp_path.iterdir()) == []
    assert fake_main["saved"] == []


def test_main_with_a_null_aggregation_norm_builds_the_main_pairwise_model(
    fake_main: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """Contract (issue #540): with ``aggregation_norm: null`` ``main`` builds the
    1765-parameter pairwise model of main (no norm module) and trains one epoch,
    saving the epoch figure and the final figure.
    """
    dango_module.main(_pairwise_main_cfg(None))
    assert "Parameter count: 1765" in capsys.readouterr().out.splitlines()
    assert fake_main["saved"] == ["training_epoch_0001.png", "final_results_TS.png"]
