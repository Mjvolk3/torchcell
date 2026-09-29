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
"""

from typing import Any

import networkx as nx
import pytest
import torch
import torch.nn as nn
from sortedcontainers import SortedDict
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv
from torch_geometric.nn import LayerNorm as PygLayerNorm

from torchcell.graph.graph import GeneGraph, GeneMultiGraph
from torchcell.models.hetero_cell_bipartite_dango_gi import (
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
    every layer (hetero_cell_bipartite_dango_gi.py:572-582), so a 3-layer MLP has one set
    of norm parameters: Linear(5, 8) 48 + 2 * Linear(8, 8) 144 + LayerNorm 16 = 208, not
    240. Any activation other than "relu" silently becomes SiLU (line 571).
    """
    torch.manual_seed(0)
    pre = PreProcessor(5, 8, num_layers=3, dropout=0.0, activation="gelu")
    norms = [m for m in pre.mlp if isinstance(m, nn.LayerNorm)]
    assert len(norms) == 3 and norms[0] is norms[1] is norms[2]
    assert sum(p.numel() for p in pre.parameters()) == 208
    assert type(pre.act) is nn.SiLU
    x = torch.randn(3, 5)
    expected = x
    for linear in [pre.mlp[0], pre.mlp[4], pre.mlp[8]]:
        expected = nn.functional.silu(norms[0](linear(expected)))
    torch.testing.assert_close(pre(x), expected)


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

    Finding: ``activation=None`` (the default) gives SiLU, not identity
    (hetero_cell_bipartite_dango_gi.py:645).
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
    assert type(gcn.act) is nn.SiLU


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
    with pytest.raises(RuntimeError, match=f"NaN detected in {message}"):
        model(_cell_graph(), _batch([[0, 1], [2, 3]]))
