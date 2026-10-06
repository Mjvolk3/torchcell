# tests/torchcell/models/test_hetero_cell_bipartite_dango.py
# [[tests.torchcell.models.test_hetero_cell_bipartite_dango]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_hetero_cell_bipartite_dango.py
"""The 003/005 predecessor ``HeteroCellBipartite`` with its DANGO interaction head
(torchcell/models/hetero_cell_bipartite_dango.py), as driven by
experiments/005-kuzmin2018-tmi/scripts/hetero_cell_bipartite_dango.py.

Fixture (the shape of ``tests/torchcell/models/test_hetero_cell_bipartite_dango_gi.py``'s
four-gene batch, restated because test modules cannot import each other here, plus the
metabolic relations this model hard-codes): 4 genes, 3 reactions, 2 metabolites;
``physical_interaction`` 0->1, 1->2, 2->3 and ``regulatory_interaction`` 3->0, 0->2;
``gpr`` hyperedges genes {0, 1} -> r0, {2} -> r1, {3} -> r2; ``(reaction, rmr,
metabolite)`` r0->m0, r1->m1, r2->m1 with stoichiometry [-1, 1, 2]. A perturbed sample
drops its genes, relabels the kept ones in index order, keeps edges among them, and
stores ``pert_mask`` for all three node types, ``cell_graph_idx_pert`` (wildtype ids; the
name has no "index" so collation does not shift it) and an ``x_pert`` row per perturbed
gene; samples are collated with ``follow_batch=["x_pert"]`` (the CellDataModule default
includes it) so ``x_pert_ptr`` exists.

Model: hidden 8, 1 conv layer, single-head GATv2 everywhere, dropout 0, LayerNorm.
Parameter count by hand (Linear(i, o) = i*o + o, LayerNorm(d) = 2d):

* embeddings 4*8 + 3*8 + 2*8 = 72; preprocessor 2 * 72 + ONE shared LayerNorm 16 = 160;
* each GATv2Conv(8, 8, heads 1): lin_l 72 + lin_r 72 + att 8 + bias 8 = 160, plus the
  wrapper's PyG LayerNorm 16 (proj is Identity at width 8): 176; the rmr conv adds
  lin_edge Linear(1, 8, no bias) 8: 184. Four relations: 3 * 176 + 184 = 712;
* interaction predictor: q, k, v, out 4 * 72 + beta 1 + prediction Linear(8, 1) 9 = 298;
* attentional pooling: gate Linear(8, 4) 36 + Linear(4, 1) 5 + transform 72 = 113;
* one-layer head Linear(8, 2) 18. Total 1373.
"""

import math
import re
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import BatchNorm as PygBatchNorm
from torch_geometric.nn import LayerNorm as PygLayerNorm

from torchcell.models.hetero_cell_bipartite_dango import (
    AttentionConvWrapper,
    GeneInteractionAttention,
    GeneInteractionPredictor,
    HeteroCellBipartite,
    PreProcessor,
    get_norm_layer,
)

G, R, M, D = 4, 3, 2, 8
PHYS = ("gene", "physical_interaction", "gene")
REG = ("gene", "regulatory_interaction", "gene")
GPR = ("gene", "gpr", "reaction")
RMR = ("reaction", "rmr", "metabolite")
GENE_EDGES = {PHYS: [(0, 1), (1, 2), (2, 3)], REG: [(3, 0), (0, 2)]}
GPR_PAIRS = [(0, 0), (1, 0), (2, 1), (3, 2)]
IDENTITY = list(range(G))


def _pairs(pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _graph(pert: list[int] | None = None, name: list[int] = IDENTITY) -> HeteroData:
    """Wildtype (pert None) or perturbed sample; wildtype gene g is renamed name[g]."""
    gene_edges = {
        et: [(name[s], name[t]) for s, t in pairs] for et, pairs in GENE_EDGES.items()
    }
    gpr = [(name[g], r) for g, r in GPR_PAIRS]
    pert_named = None if pert is None else [name[g] for g in pert]
    keep = [g for g in range(G) if g not in (pert_named or [])]
    new_id = {g: i for i, g in enumerate(keep)}
    data = HeteroData()
    data["gene"].num_nodes = len(keep)
    data["reaction"].num_nodes = R
    data["metabolite"].num_nodes = M
    for edge_type, pairs in gene_edges.items():
        data[edge_type].edge_index = _pairs(
            [(new_id[s], new_id[t]) for s, t in pairs if s in new_id and t in new_id]
        )
    data[GPR].hyperedge_index = _pairs([(new_id[g], r) for g, r in gpr if g in new_id])
    data[RMR].hyperedge_index = _pairs([(0, 0), (1, 1), (2, 1)])
    data[RMR].stoichiometry = torch.tensor([-1.0, 1.0, 2.0])
    if pert_named is not None:
        mask = torch.zeros(G, dtype=torch.bool)
        mask[pert_named] = True
        data["gene"].pert_mask = mask
        data["reaction"].pert_mask = torch.zeros(R, dtype=torch.bool)
        data["metabolite"].pert_mask = torch.zeros(M, dtype=torch.bool)
        data["gene"].cell_graph_idx_pert = torch.tensor(pert_named, dtype=torch.long)
        data["gene"].x_pert = torch.zeros(len(pert_named), 1)
    return data


def _batch(perts: list[list[int]], name: list[int] = IDENTITY) -> Batch:
    batch: Batch = Batch.from_data_list(
        [_graph(p, name) for p in perts], follow_batch=["x_pert"]
    )
    return batch


@pytest.fixture(autouse=True)
def _restore_global_rng() -> Iterator[None]:
    """Every test seeds the global torch RNG; restore it afterwards."""
    with torch.random.fork_rng(devices=[]):
        yield


def _model(seed: int = 0, **kwargs: Any) -> HeteroCellBipartite:
    torch.manual_seed(seed)
    config: dict[str, Any] = dict(
        gene_num=G,
        reaction_num=R,
        metabolite_num=M,
        hidden_channels=D,
        out_channels=2,
        num_layers=1,
        dropout=0.0,
    )
    config.update(kwargs)
    return HeteroCellBipartite(**config).eval()


def _set_identity(linear: nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.copy_(torch.eye(linear.weight.size(0)))
        linear.bias.zero_()


def _np_softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    e = np.exp(x - x.max())
    out: NDArray[np.float64] = e / e.sum()
    return out


# --- small parts ------------------------------------------------------------------ #
def test_norm_layer_factory_and_its_refusal() -> None:
    assert isinstance(get_norm_layer(5, "layer"), nn.LayerNorm)
    assert isinstance(get_norm_layer(5, "batch"), nn.BatchNorm1d)
    with pytest.raises(ValueError, match=re.escape("Unsupported norm type: instance")):
        get_norm_layer(5, "instance")


def test_any_activation_other_than_relu_builds_silu() -> None:
    """Finding: ``PreProcessor`` (line 214), ``AttentionConvWrapper`` (line 271) and the
    prediction head (line 418) map ``activation == "relu"`` to ReLU and EVERY other
    string, including "gelu" and typos, to SiLU, without a refusal. Pinned until an
    unknown activation is refused.
    """
    assert isinstance(PreProcessor(D, D, activation="gelu").act, nn.SiLU)
    assert isinstance(PreProcessor(D, D, activation="relu").act, nn.ReLU)
    wrapper = AttentionConvWrapper(_StubConv(D), D, activation="gleu")
    assert isinstance(wrapper.act, nn.SiLU)
    model = _model(prediction_head_config={"head_num_layers": 2, "activation": "tanh"})
    head = model.prediction_head
    assert isinstance(head, nn.Sequential)
    assert [type(m).__name__ for m in head] == [
        "Linear",
        "LayerNorm",
        "SiLU",
        "Dropout",
        "Linear",
    ]


def test_preprocessor_shares_one_norm_module_across_its_layers() -> None:
    """Finding: one ``norm_layer`` object is appended after both Linears (lines 215-223),
    so the two normalizations share weights: 2 * 72 + 16 = 160 parameters.
    """
    pre = PreProcessor(D, D, num_layers=2, dropout=0.0)
    assert pre.mlp[1] is pre.mlp[5]
    assert sum(p.numel() for p in pre.parameters()) == 160


class _StubConv(nn.Module):
    """A conv stand-in with ``out_channels`` (and optionally heads/concat)."""

    def __init__(
        self, out_channels: int, heads: int | None = None, concat: bool = True
    ):
        super().__init__()
        self.out_channels = out_channels
        if heads is not None:
            self.heads = heads
            self.concat = concat
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        width = self.out_channels * (
            getattr(self, "heads", 1) if getattr(self, "concat", False) else 1
        )
        return self.scale * x[:, :width]


def test_conv_wrapper_width_rules_norm_options_and_forward() -> None:
    """Expected width is heads * out_channels when the conv concatenates, else
    out_channels; a width other than the target adds Linear(width, target). Norm "batch"
    is PyG BatchNorm, "layer" PyG LayerNorm, any other string or None is no norm.
    Forward (no norm, dropout 0) is ReLU(proj(conv(x))) with the stub conv 2 * x[:, :w].
    """
    concat = AttentionConvWrapper(_StubConv(3, heads=2, concat=True), 8)
    assert isinstance(concat.proj, nn.Linear) and concat.proj.in_features == 6
    mean = AttentionConvWrapper(_StubConv(8, heads=2, concat=False), 8)
    assert isinstance(mean.proj, nn.Identity)
    plain = AttentionConvWrapper(_StubConv(5), 8, dropout=0.0, activation="relu")
    assert isinstance(plain.proj, nn.Linear) and plain.proj.in_features == 5
    assert plain.norm is None and plain.dropout is None
    assert isinstance(
        AttentionConvWrapper(_StubConv(8), 8, norm="batch").norm, PygBatchNorm
    )
    assert isinstance(
        AttentionConvWrapper(_StubConv(8), 8, norm="layer").norm, PygLayerNorm
    )
    assert AttentionConvWrapper(_StubConv(8), 8, norm="group").norm is None
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    proj = plain.proj
    assert isinstance(proj, nn.Linear)
    with torch.no_grad():
        expected = torch.relu((2.0 * x[:, :5]) @ proj.weight.T + proj.bias)
        assert torch.equal(plain(x, torch.zeros(2, 0, dtype=torch.long)), expected)


# --- DANGO interaction head ------------------------------------------------------- #
def test_interaction_attention_closed_form_single_head_scale_and_rezero_start() -> None:
    """Q = K = V = out = identity, dropout 0.

    Findings: ``beta`` starts at 0.1, not 0 as the comment on line 41 says; the scores
    are divided by sqrt(hidden_dim) = sqrt(8), not sqrt(head_dim), and ``num_heads`` is
    stored but never splits the features (single-head attention whatever num_heads is).
    Pinned until the comment and the head split agree with the code.

    Two genes each see only the other (self masked to -1e9): out_i = x_i + 0.1 x_j.
    Three genes: out_i = x_i + 0.1 * sum_{j != i} softmax(x_i . x_j / sqrt 8) x_j.
    One gene: returned unchanged.
    """
    attn = GeneInteractionAttention(D, num_heads=4, dropout=0.0).eval()
    assert attn.beta.item() == pytest.approx(0.1)
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj):
        _set_identity(proj)
    torch.manual_seed(1)
    x = torch.randn(3, D)
    with torch.no_grad():
        pair = attn(x[:2])
        torch.testing.assert_close(pair, x[:2] + 0.1 * x[:2].flip(0))
        assert torch.equal(attn(x[:1]), x[:1])
        triple = attn(x)
    xn = x.double().numpy()
    expected = xn.copy()
    for i in range(3):
        others = [j for j in range(3) if j != i]
        w = _np_softmax(np.array([xn[i] @ xn[j] / math.sqrt(D) for j in others]))
        expected[i] = xn[i] + 0.1 * sum(
            wj * xn[j] for wj, j in zip(w, others, strict=True)
        )
    torch.testing.assert_close(
        triple.double(), torch.from_numpy(expected), atol=1e-6, rtol=1e-6
    )


def test_interaction_attention_processes_each_set_independently() -> None:
    """With a batch vector, rows of set b equal the set-b-only call; a singleton set is
    copied through.
    """
    torch.manual_seed(0)
    attn = GeneInteractionAttention(D, dropout=0.0).eval()
    x = torch.randn(6, D)
    sets = torch.tensor([0, 0, 1, 2, 2, 2])
    with torch.no_grad():
        out = attn(x, sets)
        torch.testing.assert_close(out[:2], attn(x[:2]))
        assert torch.equal(out[2], x[2])
        torch.testing.assert_close(out[3:], attn(x[3:]))


def test_interaction_predictor_is_the_set_mean_of_linear_squared_dynamics() -> None:
    """With identity attention projections a pair gives diff_i = 0.1 x_j, so the set score
    is mean_i (w . (0.01 x_j^2) + b). Sets {0, 1}, {} (middle, empty), {2, 3}: the empty
    set scores 0 (the zeros it is initialized with). Without a batch vector all genes
    pool into one score of shape [1]. A trailing empty set gets no row (max + 1 = 3).
    """
    pred = GeneInteractionPredictor(D, dropout=0.0).eval()
    for proj in (
        pred.attention.q_proj,
        pred.attention.k_proj,
        pred.attention.v_proj,
        pred.attention.out_proj,
    ):
        _set_identity(proj)
    torch.manual_seed(2)
    x = torch.randn(4, D)
    w = pred.prediction_layer.weight.detach()[0]
    b = pred.prediction_layer.bias.detach()[0]

    def pair_score(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return ((w * 0.01 * c**2).sum() + b + (w * 0.01 * a**2).sum() + b) / 2

    with torch.no_grad():
        out = pred(x, torch.tensor([0, 0, 2, 2]))
    assert out.shape == (3, 1)
    torch.testing.assert_close(out[0, 0], pair_score(x[0], x[1]))
    assert out[1, 0].item() == 0.0
    torch.testing.assert_close(out[2, 0], pair_score(x[2], x[3]))
    with torch.no_grad():
        pooled = pred(x[:2])
    assert pooled.shape == (1,)
    torch.testing.assert_close(pooled[0], pair_score(x[0], x[1]))


# --- assembled model -------------------------------------------------------------- #
def test_parameter_counts_by_hand() -> None:
    """The module docstring derivation, total 1373.

    Finding: ``out_channels`` is accepted and ignored; the head is always built with 2
    outputs (line 399), so ``out_channels=1`` builds the same 18-parameter head.
    Pinned until the argument is used or removed.
    """
    model = _model()
    assert model.num_parameters == {
        "gene_embedding": 32,
        "reaction_embedding": 24,
        "metabolite_embedding": 16,
        "preprocessor": 160,
        "convs": 712,
        "global_aggregator": 113,
        "gene_interaction_predictor": 298,
        "prediction_head": 18,
        "total": 1373,
    }
    assert sum(p.numel() for p in model.parameters()) == 1373
    one = _model(out_channels=1).prediction_head
    assert isinstance(one, nn.Sequential)
    last = one[-1]
    assert isinstance(last, nn.Linear) and last.out_features == 2


def test_encoder_config_heads_and_unread_shipped_keys() -> None:
    """Heads 3 on the gene relations: GATv2Conv(8, 8 // 3 = 2, heads 3, concat) gives a
    width of 6, so the wrapper adds Linear(6, 8). The shipped 005 config also sets
    ``gene_encoder_config.bias / share_weights / num_attention_layers`` and
    ``metabolism_config.is_stoich_gated / use_attention``; none is read, so adding them
    leaves every parameter name and shape unchanged (finding: config keys read by no
    code, pinned until they are wired or dropped from the config).
    """
    model = _model(gene_encoder_config={"heads": 3})
    wrapper = dict(model.named_modules())[
        "convs.0.convs.<gene___physical_interaction___gene>"
    ]
    assert isinstance(wrapper, AttentionConvWrapper)
    assert isinstance(wrapper.conv, nn.Module)
    assert wrapper.conv.heads == 3 and wrapper.conv.out_channels == 2
    assert isinstance(wrapper.proj, nn.Linear) and wrapper.proj.in_features == 6
    plain = _model()
    noisy = _model(
        gene_encoder_config={
            "bias": False,
            "share_weights": True,
            "num_attention_layers": 5,
        },
        metabolism_config={"is_stoich_gated": False, "use_attention": False},
    )
    assert {k: tuple(v.shape) for k, v in plain.state_dict().items()} == {
        k: tuple(v.shape) for k, v in noisy.state_dict().items()
    }


def test_predictions_are_blended_by_a_frozen_epoch_alpha_of_one_twentieth() -> None:
    """Finding: ``predictions = [(1 - alpha) fitness, alpha gene_interaction]`` with
    ``alpha = min(1, (current_epoch + 1) / 20)`` (lines 583-587), but nothing advances
    ``current_epoch`` (the increment is commented out and no trainer sets it), so alpha
    stays 1/20: the fitness column is 0.95 * fitness and the gene-interaction column is
    0.05 * the DANGO score for the whole run. Setting current_epoch to 19 gives alpha 1.
    Pinned until alpha is driven by the trainer or removed.
    """
    model = _model()
    batch = _batch([[0, 1], [2]])
    with torch.no_grad():
        pred, out = model(_graph(), batch)
    assert model.current_epoch == 0
    torch.testing.assert_close(pred[:, 0:1], 0.95 * out["fitness"])
    torch.testing.assert_close(pred[:, 1:2], 0.05 * out["gene_interaction"])
    assert torch.equal(out["gene_interaction"], out["interaction_component"])
    model.current_epoch = 19
    with torch.no_grad():
        late, _ = model(_graph(), batch)
    assert torch.equal(late[:, 0], torch.zeros(2))
    torch.testing.assert_close(late[:, 1:2], out["gene_interaction"])


def test_fitness_and_interaction_wiring() -> None:
    """Fitness = head(z_w - z_i)[:, 0:1] with z_w the pooled wildtype genes and z_i the
    per-sample pooled perturbed genes; the interaction score reads the WILDTYPE gene
    states at ``cell_graph_idx_pert`` (not the perturbed-graph states), grouped by
    ``x_pert_ptr`` [0, 2, 3] into sets {0, 1} and {2}.
    """
    model = _model()
    batch = _batch([[0, 1], [2]])
    assert batch["gene"].x_pert_ptr.tolist() == [0, 2, 3]
    with torch.no_grad():
        _, out = model(_graph(), batch)
        z_w = model.forward_single(_graph())
        z_w_pool = model.global_aggregator(
            z_w, index=torch.zeros(G, dtype=torch.long), dim_size=1
        )
        z_i = model.global_aggregator(
            model.forward_single(batch), index=batch["gene"].batch
        )
        interaction = model.gene_interaction_predictor(
            z_w[[0, 1, 2]], torch.tensor([0, 0, 1])
        )
    torch.testing.assert_close(out["z_w"], z_w_pool)
    torch.testing.assert_close(out["z_i"], z_i)
    torch.testing.assert_close(out["z_p"], z_w_pool.expand(2, -1) - z_i)
    torch.testing.assert_close(
        out["fitness"], model.prediction_head(out["z_p"])[:, 0:1]
    )
    torch.testing.assert_close(out["gene_interaction"], interaction)


def test_the_x_pert_batch_path_matches_the_ptr_path_and_no_vector_fails() -> None:
    """Without ``x_pert_ptr`` the stored ``x_pert_batch`` gives the same predictions.
    With neither, the predictor pools every perturbed gene into one [1] score and the
    concatenation with the [2, 1] fitness raises.
    """
    model = _model()
    batch = _batch([[0, 1], [2]])
    with torch.no_grad():
        ref, _ = model(_graph(), batch)
        del batch["gene"].x_pert_ptr
        via_batch, _ = model(_graph(), batch)
        del batch["gene"].x_pert_batch
        with pytest.raises(
            RuntimeError,
            match=re.escape("Tensors must have same number of dimensions: got 2 and 1"),
        ):
            model(_graph(), batch)
    torch.testing.assert_close(via_batch, ref)


def test_the_current_processor_field_name_is_not_read() -> None:
    """Finding: since commit f72cabc7a (2025-06-03) ``SubgraphRepresentation`` stores the
    perturbed wildtype ids as ``gene.perturbation_indices``; this model still reads
    ``gene.cell_graph_idx_pert`` (line 543), so a batch from the current processor fails
    with AttributeError. Pinned until the model reads ``perturbation_indices``.
    """
    model = _model()
    samples = [_graph(p) for p in ([0, 1], [2])]
    for sample in samples:
        sample["gene"].perturbation_indices = sample["gene"].cell_graph_idx_pert
        del sample["gene"].cell_graph_idx_pert
    batch = Batch.from_data_list(samples, follow_batch=["x_pert"])
    with pytest.raises(
        AttributeError, match=re.escape("has no attribute 'cell_graph_idx_pert'")
    ):
        model(_graph(), batch)


def test_fitness_depends_on_the_batch_partner_under_layer_norm() -> None:
    """Finding: the conv wrapper calls PyG ``LayerNorm`` (mode "graph") with no batch
    vector (line 283), so one mean and variance span every gene of every sample. In eval
    the fitness of sample {0, 1} differs next to {2} and next to {3}; the interaction
    column (wildtype states only) is unchanged. With norm "batch" (running statistics in
    eval) the fitness is the same in both batches. The successor GI model has the same
    pinned behavior. Pinned until the norm receives the batch vector.
    """
    for norm, depends in [("layer", True), ("batch", False)]:
        model = _model(norm=norm)
        with torch.no_grad():
            first, _ = model(_graph(), _batch([[0, 1], [2]]))
            second, _ = model(_graph(), _batch([[0, 1], [3]]))
        assert (abs(first[0, 0].item() - second[0, 0].item()) > 1e-5) is depends, norm
        assert first[0, 1].item() == pytest.approx(second[0, 1].item(), abs=1e-7)


def test_relabeling_genes_leaves_the_predictions() -> None:
    """Model B is model A with embedding row NAME[g] = A's row g, run on the graph and
    batch with every gene g renamed NAME[g]. GATv2, PyG LayerNorm, attentional pooling and
    the set attention are all order-free over nodes, so the predictions agree.
    """
    name = [2, 0, 3, 1]
    model_a = _model()
    model_b = _model()
    with torch.no_grad():
        model_b.gene_embedding.weight[name] = model_a.gene_embedding.weight
        pred_a, _ = model_a(_graph(), _batch([[0, 1], [2]]))
        pred_b, _ = model_b(_graph(name=name), _batch([[0, 1], [2]], name=name))
    torch.testing.assert_close(pred_b, pred_a, atol=1e-5, rtol=1e-5)


def test_the_metabolic_branch_and_the_second_head_output_never_learn() -> None:
    """Gradient of predictions.sum() (train mode, dropout 0).

    Finding: the model returns gene states only and no relation sends reaction or
    metabolite states to genes, so the reaction and metabolite embeddings and every gpr
    and rmr conv get no gradient (None); the prediction head has 2 outputs but only
    column 0 is read (``[:, 0:1]``, line 539), so row 1 of its weight and entry 1 of its
    bias get a gradient of exactly 0. Pinned until the metabolic branch reaches the
    genes and the head width matches its use.
    """
    model = _model().train()
    pred, _ = model(_graph(), _batch([[0, 1], [2]]))
    pred.sum().backward()
    no_grad = sorted(n for n, p in model.named_parameters() if p.grad is None)
    gpr = "convs.0.convs.<gene___gpr___reaction>."
    rmr = "convs.0.convs.<reaction___rmr___metabolite>."
    gat = [
        "conv.att",
        "conv.bias",
        "conv.lin_l.bias",
        "conv.lin_l.weight",
        "conv.lin_r.bias",
        "conv.lin_r.weight",
    ]
    expected = (
        [gpr + p for p in [*gat, "norm.bias", "norm.weight"]]
        + [rmr + p for p in [*gat, "conv.lin_edge.weight", "norm.bias", "norm.weight"]]
        + ["metabolite_embedding.weight", "reaction_embedding.weight"]
    )
    assert no_grad == sorted(expected)
    head = model.prediction_head
    assert isinstance(head, nn.Sequential)
    last = head[-1]
    assert isinstance(last, nn.Linear)
    assert last.weight.grad is not None and last.bias.grad is not None
    assert torch.equal(last.weight.grad[1], torch.zeros(D))
    assert last.bias.grad[1].item() == 0.0
    assert last.weight.grad[0].abs().max().item() > 0.0


def test_seeded_construction_is_deterministic() -> None:
    batch = _batch([[0, 1], [2]])
    a, b, c = _model(0), _model(0), _model(1)
    with torch.no_grad():
        pa, pb, pc = (m(_graph(), batch)[0] for m in (a, b, c))
    assert torch.equal(pa, pb)
    assert not torch.equal(pa, pc)


def test_head_depth_and_norm_options_and_a_column_stoichiometry() -> None:
    """head_num_layers 0 is the identity, so fitness is column 0 of z_p itself;
    head_norm None drops the norm (Linear, ReLU, Dropout, Linear); the rmr conv receives
    the stoichiometry as a [3, 1] edge_attr whether it is stored as [3] (unsqueezed) or
    already as a column;
    a wrapper dropout > 0 adds Dropout(p) and is inert in eval.
    """
    model = _model(prediction_head_config={"head_num_layers": 0})
    assert isinstance(model.prediction_head, nn.Identity)
    batch = _batch([[0, 1], [2]])
    with torch.no_grad():
        _, out = model(_graph(), batch)
    assert torch.equal(out["fitness"], out["z_p"][:, 0:1])
    no_norm = _model(prediction_head_config={"head_num_layers": 2, "head_norm": None})
    head = no_norm.prediction_head
    assert isinstance(head, nn.Sequential)
    assert [type(m).__name__ for m in head] == ["Linear", "ReLU", "Dropout", "Linear"]
    model = _model()
    rmr_conv = dict(model.named_modules())[
        "convs.0.convs.<reaction___rmr___metabolite>.conv"
    ]
    seen: list[torch.Tensor] = []

    def hook(module: nn.Module, args: Any, kwargs: dict[str, Any], output: Any) -> None:
        seen.append(kwargs["edge_attr"])

    handle = rmr_conv.register_forward_hook(hook, with_kwargs=True)
    column_graph = _graph()
    column_graph[RMR].stoichiometry = column_graph[RMR].stoichiometry.unsqueeze(1)
    with torch.no_grad():
        model.forward_single(_graph())
        model.forward_single(column_graph)
    handle.remove()
    expected_attr = torch.tensor([[-1.0], [1.0], [2.0]])
    assert len(seen) == 2
    assert all(torch.equal(attr, expected_attr) for attr in seen)
    dropped = _model(dropout=0.3)
    wrapper = dict(dropped.named_modules())[
        "convs.0.convs.<gene___physical_interaction___gene>"
    ]
    assert isinstance(wrapper, AttentionConvWrapper)
    assert isinstance(wrapper.dropout, nn.Dropout) and wrapper.dropout.p == 0.3
    with torch.no_grad():
        dropped_pred, _ = dropped(_graph(), batch)
        reference = _model(dropout=0.0)
        reference.load_state_dict(dropped.state_dict())
        reference_pred, _ = reference(_graph(), batch)
    assert torch.equal(dropped_pred, reference_pred)
