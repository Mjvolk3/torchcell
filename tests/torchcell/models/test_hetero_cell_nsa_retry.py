# tests/torchcell/models/test_hetero_cell_nsa_retry.py
# [[tests.torchcell.models.test_hetero_cell_nsa_retry]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_hetero_cell_nsa_retry.py
"""The 006 NSA retry model ``HeteroCellNSA`` (torchcell/models/hetero_cell_nsa_retry.py).

Fixture, built the way experiments/006-kuzmin-tmi/scripts/hetero_cell_nsa_retry.py feeds
the model: 4 genes, 3 reactions, 2 metabolites, then ``HeteroToDenseMask`` with the
model's node counts (the script's mandatory dense transform). The wildtype graph has
``physical_interaction`` 0->1, 1->2, 2->3, ``regulatory_interaction`` 3->0, ``gpr``
hyperedges genes {0, 1} -> r0, {2} -> r1, {3} -> r2, and the bipartite
``(reaction, rmr, metabolite)`` relation r0->m0, r1->m1, r2->m1 that the shipped config's
``incidence_graphs: [metabolism_bipartite]`` produces. A perturbed sample drops its
perturbed genes, relabels the kept genes in index order, keeps only edges among them and
stores ``gene.pert_mask`` (length 4); samples are collated with ``Batch.from_data_list``.
Model: hidden 8, 2 heads, dropout 0, layer norm, so a forward in eval is pure.

The NSA blocks themselves (torchcell/nn/hetero_nsa.py) have their own tests; here the
model's use of them is pinned: what it hands them, what it does with what comes back,
which parameters can learn, and whether a genotype's prediction depends on the other
genotypes in the batch. Parameter counts by hand for d = 8, 2 heads (Linear(i, o) =
i*o + o, LayerNorm(d) = 2d):

* SelfAttentionBlock (one per node type): norm1 16 + norm2 16 + q, k, v, out
  4 * 72 = 288 + feed-forward MLP (8*32 + 32) + (32*8 + 8) = 552 (the MLP alone), block
  total 872;
* NodeSelfAttention (one per edge type): the same 872 plus an edge-attribute MLP per head
  (Linear(1, 16) 32 + Linear(16, 1) 17 = 49) * 2 heads = 98, block total 970;
* default graph names give 4 edge types (2 gene-gene, gpr, metabolite-reaction), so the
  NSA layer is 4 * 970 + 3 * 872 = 6496;
* embeddings 4*8 + 3*8 + 2*8 = 72; preprocessor 2 * 72 + ONE shared LayerNorm 16 = 160;
  per-type norms 3 * 16 = 48; attentional pooling gate Linear(8, 4) 36 + Linear(4, 1) 5
  + transform Linear(8, 8) 72 = 113; one-layer head Linear(8, 1) 9. Total 6898.
"""

import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import nn
from torch_geometric.data import Batch, HeteroData

from torchcell.models.act import act_register
from torchcell.models.hetero_cell_nsa_retry import (
    AttentionalGraphAggregation,
    HeteroCellNSA,
    PreProcessor,
    get_norm_layer,
)
from torchcell.nn.masked_attention_block import NodeSelfAttention
from torchcell.transforms.hetero_to_dense_mask import HeteroToDenseMask

G, R, M, D, HEADS = 4, 3, 2, 8, 2
PHYS = ("gene", "physical_interaction", "gene")
REG = ("gene", "regulatory_interaction", "gene")
GPR = ("gene", "gpr", "reaction")
RMR = ("reaction", "rmr", "metabolite")
GENE_EDGES = {PHYS: [(0, 1), (1, 2), (2, 3)], REG: [(3, 0)]}
GPR_PAIRS = [(0, 0), (1, 0), (2, 1), (3, 2)]


def _pairs(pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _graph(
    pert: list[int] | None = None,
    gene_edges: dict[tuple[str, str, str], list[tuple[int, int]]] | None = None,
) -> HeteroData:
    """The wildtype graph (pert None) or a perturbed sample, dense-masked."""
    edges = GENE_EDGES if gene_edges is None else gene_edges
    keep = [g for g in range(G) if g not in (pert or [])]
    new_id = {g: i for i, g in enumerate(keep)}
    data = HeteroData()
    data["gene"].num_nodes = len(keep)
    data["reaction"].num_nodes = R
    data["metabolite"].num_nodes = M
    if pert is not None:
        mask = torch.zeros(G, dtype=torch.bool)
        mask[pert] = True
        data["gene"].pert_mask = mask
    for edge_type, pairs in edges.items():
        kept = [(new_id[s], new_id[t]) for s, t in pairs if s in new_id and t in new_id]
        data[edge_type].edge_index = _pairs(kept)
    data[GPR].hyperedge_index = _pairs(
        [(new_id[g], r) for g, r in GPR_PAIRS if g in new_id]
    )
    data[RMR].hyperedge_index = _pairs([(0, 0), (1, 1), (2, 1)])
    data[RMR].stoichiometry = torch.tensor([-1.0, 1.0, 2.0])
    dense: HeteroData = HeteroToDenseMask({"gene": G, "reaction": R, "metabolite": M})(
        data
    )
    return dense


def _batch(perts: list[list[int]], **kwargs: Any) -> Batch:
    batch: Batch = Batch.from_data_list([_graph(p, **kwargs) for p in perts])
    return batch


@pytest.fixture(autouse=True)
def _restore_global_rng() -> Iterator[None]:
    """Every test seeds the global torch RNG; restore it afterwards."""
    with torch.random.fork_rng(devices=[]):
        yield


def _model(seed: int = 0, **kwargs: Any) -> HeteroCellNSA:
    torch.manual_seed(seed)
    config: dict[str, Any] = dict(
        gene_num=G,
        reaction_num=R,
        metabolite_num=M,
        hidden_channels=D,
        out_channels=1,
        num_heads=HEADS,
        dropout=0.0,
    )
    config.update(kwargs)
    return HeteroCellNSA(**config).eval()


# --- helpers ---------------------------------------------------------------------- #
def test_norm_layer_factory_and_its_refusal() -> None:
    """Layer -> LayerNorm(d), batch -> BatchNorm1d(d), anything else raises by name."""
    layer = get_norm_layer(6, "layer")
    assert isinstance(layer, nn.LayerNorm) and layer.normalized_shape == (6,)
    batch = get_norm_layer(6, "batch")
    assert isinstance(batch, nn.BatchNorm1d) and batch.num_features == 6
    with pytest.raises(ValueError, match=re.escape("Unsupported norm type: group")):
        get_norm_layer(6, "group")


def test_preprocessor_shares_one_norm_module_across_its_layers() -> None:
    """Finding: ``PreProcessor`` builds ``norm_layer`` once (hetero_cell_nsa_retry.py:69)
    and appends the SAME module after both Linears (lines 72 and 77), so the two
    normalizations share one weight and bias: 2 * 72 + 16 = 160 parameters, not 176.
    Forward is Linear -> LN -> ReLU -> Linear -> LN -> ReLU (dropout 0).
    Pinned until each layer gets its own norm.
    """
    torch.manual_seed(0)
    pre = PreProcessor(D, D, num_layers=2, dropout=0.0)
    assert pre.mlp[1] is pre.mlp[5]
    assert sum(p.numel() for p in pre.parameters()) == 160
    l1, ln, l2 = pre.mlp[0], pre.mlp[1], pre.mlp[4]
    assert isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear)
    assert isinstance(ln, nn.LayerNorm)
    x = torch.randn(5, D)
    with torch.no_grad():
        ln.weight.copy_(torch.linspace(0.5, 2.0, D))
        ln.bias.copy_(torch.linspace(-1.0, 1.0, D))
        h = torch.relu(nn.functional.layer_norm(l1(x), (D,), ln.weight, ln.bias))
        expected = torch.relu(nn.functional.layer_norm(l2(h), (D,), ln.weight, ln.bias))
        assert torch.equal(pre(x), expected)
    assert isinstance(pre.mlp[2], type(act_register["relu"]))


def test_attentional_pooling_is_a_per_group_softmax_weighted_sum() -> None:
    """out_g = sum_{i in g} softmax_g(gate(x_i)) * ReLU(W x_i + b), gate = Linear ->
    ReLU -> Linear; numpy oracle on the module's weights, groups {0, 1, 2} and {3, 4}.
    """
    torch.manual_seed(0)
    pool = AttentionalGraphAggregation(D, D, dropout=0.0).eval()
    x = torch.randn(5, D)
    index = torch.tensor([0, 0, 0, 1, 1])
    out = pool(x, index)
    g1, g2, t = pool.gate_nn[0], pool.gate_nn[3], pool.transform_nn[0]
    assert isinstance(g1, nn.Linear) and isinstance(g2, nn.Linear)
    assert isinstance(t, nn.Linear)

    def np_linear(lin: nn.Linear, v: NDArray[np.float64]) -> NDArray[np.float64]:
        out_np: NDArray[np.float64] = (
            v @ lin.weight.detach().double().numpy().T
            + lin.bias.detach().double().numpy()
        )
        return out_np

    xn = x.double().numpy()
    gate = np_linear(g2, np.maximum(np_linear(g1, xn), 0.0))[:, 0]
    value = np.maximum(np_linear(t, xn), 0.0)
    expected = []
    for rows in ([0, 1, 2], [3, 4]):
        w = np.exp(gate[rows] - gate[rows].max())
        expected.append((w / w.sum()) @ value[rows])
    torch.testing.assert_close(
        out.detach().double(),
        torch.from_numpy(np.stack(expected)),
        atol=1e-6,
        rtol=1e-6,
    )


# --- construction ----------------------------------------------------------------- #
def test_heads_must_divide_the_width() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Hidden dimension (10) must be divisible by number of heads (4)"
        ),
    ):
        _model(hidden_channels=10, num_heads=4)


def test_graph_names_build_the_edge_types_and_the_pattern_builds_the_blocks() -> None:
    """Default graph names are the suffixed relations; any list replaces them. gpr and the
    metabolite hyperedge relation are always added. Pattern ["M", "S"] -> two blocks.
    """
    default = _model()
    assert default.graph_names == ["physical_interaction", "regulatory_interaction"]
    assert default.nsa_layer.edge_types == {
        PHYS,
        REG,
        GPR,
        ("metabolite", "reaction", "metabolite"),
    }
    assert [b.layer_type for b in default.nsa_layer.blocks] == ["M", "S"]
    assert default.nsa_layer.aggregation == "sum"
    named = _model(
        graph_names=["physical", "tflink"], attention_pattern=["S", "M", "S"]
    )
    assert named.nsa_layer.edge_types == {
        ("gene", "physical", "gene"),
        ("gene", "tflink", "gene"),
        GPR,
        ("metabolite", "reaction", "metabolite"),
    }
    assert [b.layer_type for b in named.nsa_layer.blocks] == ["S", "M", "S"]


def test_parameter_counts_by_hand() -> None:
    """The module docstring derivation: 6898 in total. Nine graph names (the shipped
    006 list) give 11 edge types: 6496 + 7 * 970 = 13286 for the NSA layer.
    """
    model = _model()
    assert model.num_parameters == {
        "gene_embedding": 32,
        "reaction_embedding": 24,
        "metabolite_embedding": 16,
        "preprocessor": 160,
        "nsa_layer": 6496,
        "layer_norms": 48,
        "global_aggregator": 113,
        "prediction_head": 9,
        "total": 6898,
    }
    assert sum(p.numel() for p in model.parameters()) == 6898
    nine = _model(graph_names=[f"g{i}" for i in range(9)])
    assert nine.num_parameters["nsa_layer"] == 13286


def test_prediction_head_config_keys_build_the_head() -> None:
    """head_num_layers 3, hidden 5, head_norm batch, gelu: Linear(8, 5) BN GELU Dropout
    Linear(5, 5) BN GELU Dropout Linear(5, 1); 0 layers is the identity; the head's
    dropout key wins over the model dropout.
    """
    model = _model(
        dropout=0.0,
        prediction_head_config={
            "hidden_channels": 5,
            "head_num_layers": 3,
            "dropout": 0.25,
            "activation": "gelu",
            "head_norm": "batch",
        },
    )
    head = model.prediction_head
    assert isinstance(head, nn.Sequential)
    kinds = [type(m).__name__ for m in head]
    assert kinds == [
        "Linear",
        "BatchNorm1d",
        "GELU",
        "Dropout",
        "Linear",
        "BatchNorm1d",
        "GELU",
        "Dropout",
        "Linear",
    ]
    shapes = [(m.in_features, m.out_features) for m in head if isinstance(m, nn.Linear)]
    assert shapes == [(8, 5), (5, 5), (5, 1)]
    assert [m.p for m in head if isinstance(m, nn.Dropout)] == [0.25, 0.25]
    assert isinstance(
        _model(prediction_head_config={"head_num_layers": 0}).prediction_head,
        nn.Identity,
    )


# --- forward ---------------------------------------------------------------------- #
def test_wildtype_path_is_layer_norm_of_nsa_plus_residual() -> None:
    """A graph with no ``batch`` attribute takes the single-graph path: genes are
    preprocessor(embedding), reactions and metabolites are raw embeddings, and the
    returned gene states are LN_gene(nsa(x)["gene"] + x["gene"]).
    """
    model = _model()
    cell_graph = _graph()
    with torch.no_grad():
        out = model.forward_single(cell_graph)
        x = {
            "gene": model.preprocessor(model.gene_embedding.weight),
            "reaction": model.reaction_embedding.weight,
            "metabolite": model.metabolite_embedding.weight,
        }
        nsa = model.nsa_layer(x, cell_graph, {})
        expected = model.layer_norms["gene"](nsa["gene"] + x["gene"])
    assert torch.equal(out, expected)


def test_forward_is_the_pooled_wildtype_minus_pooled_perturbed_through_the_head() -> (
    None
):
    """z_w = pool(wildtype genes) [1, 8]; z_i = pool per sample [2, 8]; z_p = z_w - z_i;
    predictions = head(z_p) [2, 1] and are returned again as gene_interaction.
    """
    model = _model()
    batch = _batch([[0], [3]])
    with torch.no_grad():
        pred, out = model(_graph(), batch)
        z_i = model.global_aggregator(
            model.forward_single(batch), index=batch["gene"].batch
        )
    assert out["z_w"].shape == (1, D)
    torch.testing.assert_close(out["z_i"], z_i)
    assert torch.equal(out["z_p"], out["z_w"].expand(2, D) - out["z_i"])
    assert torch.equal(pred, model.prediction_head(out["z_p"]))
    assert torch.equal(out["gene_interaction"], pred)
    assert pred.shape == (2, 1)


@pytest.mark.parametrize("pattern", [["S"], ["M"]])
def test_a_genotype_prediction_depends_on_the_other_genotypes_in_the_batch(
    pattern: list[str],
) -> None:
    """Finding: the batched gene tokens of all samples go to the NSA blocks as ONE set
    (hetero_cell_nsa_retry.py:249-263 build one flat token set and line 322
    hands it over; ``batch_idx`` is passed but no block reads it). An S block attends over every gene of every sample; an M block receives
    the collated adj_mask [2 * 4, 4] for 6 gene tokens, which NodeSelfAttention silently
    pads and crops to [6, 6], so sample 1's genes read rows meant for sample 0 and
    attend to sample 0's tokens. In eval, sample {0}'s prediction therefore changes when
    its batch partner changes from {3} to {1}, and the batched prediction differs from the
    single-sample run. Pinned until the blocks attend within a sample.

    Tolerance: the init depends on PYTHONHASHSEED (see
    ``test_init_depends_on_the_python_hash_seed``), and across 101 hash seeds the
    smallest gaps were 8.2e-4 (S, partner), 1.06e-4 (M, partner) and 2.55e-5 (M, alone).
    Float noise is about 1e-7, so ``atol=1e-6`` separates a batch-dependent model from a
    batch-independent one (which gives about 1e-7 and fails this test) on every seed.
    """
    model = _model(attention_pattern=pattern)
    cell_graph = _graph()
    with torch.no_grad():
        with_3, _ = model(cell_graph, _batch([[0], [3]]))
        with_1, _ = model(cell_graph, _batch([[0], [1]]))
        alone, _ = model(cell_graph, _batch([[0]]))
    assert not torch.allclose(with_3[0], with_1[0], rtol=0.0, atol=1e-6)
    assert not torch.allclose(with_3[0], alone[0], rtol=0.0, atol=1e-6)


def test_the_m_block_receives_a_non_square_mask_for_batches_and_for_gpr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spy on NodeSelfAttention: for a 2-sample batch (6 gene tokens) the gene-gene
    blocks receive tokens [1, 6, 8] with the collated mask [1, 8, 4], and the gpr gene
    side receives the [8, 3] gene-by-reaction incidence as a gene-by-gene mask (the
    reaction side gets its transpose for 6 reaction tokens). NodeSelfAttention pads and
    crops these to square without a warning, so a gene "attends" to the gene whose index
    equals a reaction it is annotated to. Finding, pinned with the batch finding.
    """
    seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original: Callable[..., torch.Tensor] = NodeSelfAttention.forward

    def spy(
        self: NodeSelfAttention, x: torch.Tensor, adj_mask: torch.Tensor, *a: Any
    ) -> torch.Tensor:
        seen.append((tuple(x.shape), tuple(adj_mask.shape)))
        return original(self, x, adj_mask, *a)

    monkeypatch.setattr(NodeSelfAttention, "forward", spy)
    model = _model(attention_pattern=["M"])
    with torch.no_grad():
        model.forward_single(_batch([[0], [3]]))
    assert sorted(seen) == sorted(
        [
            ((1, 6, D), (1, 8, 4)),  # physical_interaction
            ((1, 6, D), (1, 8, 4)),  # regulatory_interaction
            ((1, 6, D), (1, 8, 3)),  # gpr, gene side
            ((1, 6, D), (1, 3, 8)),  # gpr, reaction side
        ]
    )


def test_bare_graph_names_against_suffixed_relations_are_silently_unused() -> None:
    """Finding: the shipped config (experiments/006-kuzmin-tmi/conf/
    hetero_cell_nsa_retry.yaml) passes ``graphs: [physical, regulatory, ...]`` as
    ``graph_names``, while ``to_cell_data`` emits ``physical_interaction`` and
    ``regulatory_interaction``. The M block skips an edge type the data lacks
    (torchcell/nn/hetero_nsa.py ``if (src, rel, dst) not in data.edge_types: continue``),
    so with the bare names the gene states are exactly those of a graph with no
    gene-gene relation at all. Pinned until the model refuses a graph name the data
    does not carry.
    """
    model = _model(graph_names=["physical", "regulatory"], attention_pattern=["M"])
    with torch.no_grad():
        suffixed = model.forward_single(_graph())
        no_gene_edges = model.forward_single(_graph(gene_edges={}))
        default = _model(attention_pattern=["M"]).forward_single(_graph())
    assert torch.equal(suffixed, no_gene_edges)
    assert not torch.allclose(default, no_gene_edges)


def test_only_gene_paths_learn_and_the_metabolism_relation_never_runs() -> None:
    """Gradient of predictions.sum() on a 2-sample batch (train mode, dropout 0).

    Finding: the model returns gene states only (hetero_cell_nsa_retry.py:331), and no
    block lets a reaction or metabolite state reach a gene, so the reaction and
    metabolite embeddings, their layer norms and their S blocks get no gradient; the
    model's metabolism edge type ``(metabolite, reaction, metabolite)`` is never present
    in the bipartite data the 006 config builds (``(reaction, rmr, metabolite)``), so
    its M block never runs; every M block's edge-attribute MLPs never run (no
    ``edge_attr`` on gene relations); the gpr block runs on both sides, and only its
    gene side reaches the output. Pinned until the metabolic branch feeds the genes.
    """
    model = _model().train()
    pred, _ = model(_graph(), _batch([[0], [3]]))
    pred.sum().backward()
    no_grad = {n for n, p in model.named_parameters() if p.grad is None}
    prefix = "nsa_layer.blocks."
    dead_modules = {
        n.rsplit(".", 1)[0].removeprefix(prefix)
        for n in no_grad
        if n.startswith(prefix) and ".edge_attr_proj." not in n
    }
    assert dead_modules == {
        "0.masked_blocks.metabolite__reaction__metabolite.norm1",
        "0.masked_blocks.metabolite__reaction__metabolite.q_proj",
        "0.masked_blocks.metabolite__reaction__metabolite.k_proj",
        "0.masked_blocks.metabolite__reaction__metabolite.v_proj",
        "0.masked_blocks.metabolite__reaction__metabolite.out_proj",
        "0.masked_blocks.metabolite__reaction__metabolite.norm2",
        "0.masked_blocks.metabolite__reaction__metabolite.mlp.0",
        "0.masked_blocks.metabolite__reaction__metabolite.mlp.2",
        *(
            f"1.self_blocks.{nt}.{part}"
            for nt in ("reaction", "metabolite")
            for part in (
                "norm1",
                "norm2",
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "mlp.0",
                "mlp.3",
            )
        ),
    }
    edge_attr = {n for n in no_grad if ".edge_attr_proj." in n}
    assert (
        len(edge_attr) == 4 * HEADS * 4
    )  # 4 blocks x 2 heads x (2 weights + 2 biases)
    others = sorted(n for n in no_grad if not n.startswith(prefix))
    assert others == [
        "layer_norms.metabolite.bias",
        "layer_norms.metabolite.weight",
        "layer_norms.reaction.bias",
        "layer_norms.reaction.weight",
        "metabolite_embedding.weight",
        "reaction_embedding.weight",
    ]
    for name in (
        "gene_embedding.weight",
        "preprocessor.mlp.0.weight",
        "prediction_head.0.weight",
    ):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and grad.abs().max().item() > 0.0


def test_seeded_construction_is_deterministic() -> None:
    """Same seed: identical state dicts and predictions; seed 1 differs.

    This holds within ONE process only: across processes the init also depends on
    PYTHONHASHSEED (``test_init_depends_on_the_python_hash_seed``).
    """
    batch = _batch([[0], [3]])
    a, b, c = _model(0), _model(0), _model(1)
    with torch.no_grad():
        pa, pb, pc = (m(_graph(), batch)[0] for m in (a, b, c))
    assert all(
        torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()
    )
    assert torch.equal(pa, pb)
    assert not torch.equal(pa, pc)


def test_an_unbatched_perturbed_sample_embeds_genes_zero_to_n_minus_one() -> None:
    """Finding: a perturbed sample passed without a ``batch`` vector takes the
    single-graph path (hetero_cell_nsa_retry.py:299-311), which embeds
    ``arange(min(num_nodes, gene_num))`` and ignores ``pert_mask``. Deleting gene 0 or
    gene 3 both leave 3 genes, so both samples embed genes 0, 1, 2; with an S-only
    pattern (no edges read) their gene states are identical, and the prediction differs
    from the same sample collated as a batch of one (which embeds genes 1, 2, 3).
    Pinned until the single path honors ``pert_mask``.
    """
    model = _model(attention_pattern=["S"])
    with torch.no_grad():
        drop_0 = model.forward_single(_graph([0]))
        drop_3 = model.forward_single(_graph([3]))
        alone, _ = model(_graph(), _graph([0]))
        collated, _ = model(_graph(), _batch([[0]]))
    assert torch.equal(drop_0, drop_3)
    assert alone.shape == (1, 1)
    assert abs(alone.item() - collated.item()) > 1e-4


def test_batched_reaction_and_metabolite_pert_masks_select_embedding_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``reaction.pert_mask`` [F, F, T] and ``metabolite.pert_mask`` [F, T] on both
    samples, the tokens handed to the NSA layer are reaction embedding rows [0, 1, 0, 1]
    and metabolite rows [0, 0] (raw, not preprocessed); genes are the preprocessed rows of
    the kept genes [1, 2, 3, 0, 1, 2].
    """
    model = _model()
    samples = [_graph([0]), _graph([3])]
    for sample in samples:
        sample["reaction"].pert_mask = torch.tensor([False, False, True])
        sample["metabolite"].pert_mask = torch.tensor([False, True])
    batch = Batch.from_data_list(samples)
    seen: dict[str, torch.Tensor] = {}

    def record(
        x_dict: dict[str, torch.Tensor],
        data: HeteroData,
        batch_idx: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        seen.update(x_dict)
        return dict(x_dict)

    monkeypatch.setattr(model.nsa_layer, "forward", record)
    with torch.no_grad():
        model.forward_single(batch)
        genes = model.preprocessor(model.gene_embedding.weight[[1, 2, 3, 0, 1, 2]])
    assert torch.equal(seen["reaction"], model.reaction_embedding.weight[[0, 1, 0, 1]])
    assert torch.equal(seen["metabolite"], model.metabolite_embedding.weight[[0, 0]])
    torch.testing.assert_close(seen["gene"], genes)


def test_an_nsa_failure_is_printed_and_reraised(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """forward_single prints ``Error in NSA layer: <message>`` and re-raises the error."""
    model = _model()

    def boom(*args: Any) -> dict[str, torch.Tensor]:
        raise RuntimeError("boom")

    monkeypatch.setattr(model.nsa_layer, "forward", boom)
    with pytest.raises(RuntimeError, match="^boom$"):
        model.forward_single(_graph())
    assert capsys.readouterr().out == "Error in NSA layer: boom\n"


def test_a_batch_without_pert_mask_tiles_the_full_gene_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two unperturbed samples with no ``pert_mask``: the batched path takes the first
    ``num_nodes`` = 8 rows of the tiled table, i.e. genes [0, 1, 2, 3, 0, 1, 2, 3]
    (preprocessed). A head built with ``head_norm: None`` has no norm layer.
    """
    model = _model(prediction_head_config={"head_num_layers": 2, "head_norm": None})
    head = model.prediction_head
    assert isinstance(head, nn.Sequential)
    assert [type(m).__name__ for m in head] == ["Linear", "ReLU", "Dropout", "Linear"]
    batch = Batch.from_data_list([_graph(), _graph()])
    seen: dict[str, torch.Tensor] = {}

    def record(
        x_dict: dict[str, torch.Tensor],
        data: HeteroData,
        batch_idx: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        seen.update(x_dict)
        return dict(x_dict)

    monkeypatch.setattr(model.nsa_layer, "forward", record)
    with torch.no_grad():
        model.forward_single(batch)
        expected = model.preprocessor(model.gene_embedding.weight.repeat(2, 1))
    torch.testing.assert_close(seen["gene"], expected)


_HASH_SEED_PROBE = """
import torch
from torchcell.models.hetero_cell_nsa_retry import HeteroCellNSA
torch.manual_seed(0)
model = HeteroCellNSA(4, 3, 2, 8, 1, num_heads=2, dropout=0.0)
for name, tensor in model.state_dict().items():
    print(name, repr(float(tensor.double().sum())))
"""


def _state_sums(hash_seed: str) -> dict[str, float]:
    root = Path(__file__).resolve().parents[3]
    env = {k: v for k, v in os.environ.items() if k != "DATA_ROOT"}
    env.update(
        PYTHONHASHSEED=hash_seed,
        PYTHONPATH=str(root),
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", _HASH_SEED_PROBE],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    sums: dict[str, float] = {}
    for line in result.stdout.splitlines():
        name, value = line.split(" ")
        sums[name] = float(value)
    return sums


def test_init_depends_on_the_python_hash_seed() -> None:
    """Finding: ``HeteroCellNSA`` passes Python sets of node and edge types
    (hetero_cell_nsa_retry.py:140-151) to ``HeteroNSA``, which builds each
    ``ModuleDict`` by iterating the set (torchcell/nn/hetero_nsa.py:46), so the order in
    which blocks draw from the RNG follows PYTHONHASHSEED. Two processes with
    ``torch.manual_seed(0)`` and hash seeds 1 and 2 build models with the same parameter
    names but different weights (the audit saw the gpr ``q_proj.weight`` sum at -0.31147
    and 1.76596). Seeded NSA runs (hetero_cell_nsa_retry.yaml) are therefore not
    reproducible across processes. Pinned until the node and edge types are sorted
    before the ModuleDict is built.
    """
    one, two = _state_sums("1"), _state_sums("2")
    assert sorted(one) == sorted(two)
    key = "nsa_layer.blocks.0.masked_blocks.gene__gpr__reaction.q_proj.weight"
    assert key in one
    assert abs(one[key] - two[key]) > 1e-3
    assert one == _state_sums("1")
