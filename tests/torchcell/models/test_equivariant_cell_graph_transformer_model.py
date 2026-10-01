# tests/torchcell/models/test_equivariant_cell_graph_transformer_model.py
# [[tests.torchcell.models.test_equivariant_cell_graph_transformer_model]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_equivariant_cell_graph_transformer_model.py
"""The assembled ``CellGraphTransformer``: constructor options, forward identities, loss.

Fixture: the conftest ``cell_graph`` (8 genes; a ``physical`` gene-gene relation with
edges 0->1, 1->2, 2->3, 3->4; gpr genes {0,1}->r0, {2}->r1, {3,4}->r2, {5}->r3; rmr
m0<-{r0,r1}, m1<-{r2}, m2<-{r3,r0}) and ``batch`` (3 genotypes perturbing {1, 2}, {3},
{0, 4, 5}). Model: hidden 16, 4 heads, dropout 0 so a forward is a pure function.

A randomly initialized network carries no float contract (Decision 12 of
[[plan.test-suite-buildout.2026.09.25]]), so what is pinned here is structural:

* relabeling the genes permutes every per-gene output and leaves every pooled output;
* the ReZero / zero-init options are exact identities at init (propagation, Perceiver
  mixing, Hadamard ``add``, the response basis);
* parameter counts written out by hand for a 1-layer, hidden-16 model: a Linear(i, o) has
  i*o + o parameters and a LayerNorm(d) has 2d, so the encoder layer is 4 * 272 (Q, K, V,
  out) + 2 * 32 (norms) + (16*64 + 64) + (64*16 + 16) (FFN) = 3280;
* the graph-regularization KL in closed form: with the regularized layer's Q and K
  projections zeroed every score is 0, so each gene row attends 1/9 to each of the 9
  tokens; a degree-1 row of the row-normalized adjacency contributes
  1 * (log 1 - log(1/9)) = log 9, the ``physical`` graph has 4 such rows and 4 edges, so
  the per-graph term is lambda * 4 log 9 / (4 / 8) = 8 lambda log 9;
* every constructor and forward ValueError with its exact message.
"""

import math
import re
from typing import Any

import pytest
import torch
from torch import nn
from torch_geometric.data import Data, HeteroData

from torchcell.losses.distributional import DistHead
from torchcell.models.equivariant_cell_graph_transformer import (
    CellGraphTransformer,
    EquivariantPerturbationTransform,
    GraphRegularizedTransformerLayer,
    MaskedMultitaskLoss,
    PerGeneHead,
)

N, D, HEADS, B = 8, 16, 4, 3
PERM = torch.tensor([7, 3, 5, 0, 1, 6, 2, 4])  # new position j holds old gene PERM[j]
INVERSE = torch.argsort(PERM)  # old gene g sits at new position INVERSE[g]


def _model(
    cell_graph: HeteroData, seed: int = 0, **kwargs: Any
) -> CellGraphTransformer:
    torch.manual_seed(seed)
    config: dict[str, Any] = dict(
        gene_num=N,
        hidden_channels=D,
        num_transformer_layers=1,
        num_attention_heads=HEADS,
        cell_graph=cell_graph,
        dropout=0.0,
    )
    config.update(kwargs)
    return CellGraphTransformer(**config)


def _no_grad_parameters(model: nn.Module) -> list[str]:
    return sorted(
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    )


def _relabeled(
    cell_graph: HeteroData, batch: HeteroData
) -> tuple[HeteroData, HeteroData]:
    """The same cell and batch with gene g renamed INVERSE[g] everywhere."""
    graph = cell_graph.clone()
    for edge_type in graph.edge_types:
        src, _, dst = edge_type
        edge_index = graph[edge_type].edge_index.clone()
        if src == "gene":
            edge_index[0] = INVERSE[edge_index[0]]
        if dst == "gene":
            edge_index[1] = INVERSE[edge_index[1]]
        graph[edge_type].edge_index = edge_index
    relabeled = batch.clone()
    relabeled["gene"].perturbation_indices = INVERSE[batch["gene"].perturbation_indices]
    return graph, relabeled


# --- relabeling ------------------------------------------------------------------- #
def test_relabeling_genes_permutes_per_gene_outputs_and_leaves_pooled_ones(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """With the embedding table permuted to match, every output follows the relabeling.

    The encoder has no positional encoding and the perturbation operator is equivariant,
    so ``H_genes`` and ``H_genes_pert`` permute; the interaction prediction, the global
    head and the per-metabolite head (whose gpr incidence is relabeled with the genes)
    are unchanged; the per-gene head permutes. ``h_CLS`` is unchanged.
    """
    heads = {"global": {"output_dim": 5}, "per_gene": {}, "per_metabolite": {}}
    model = _model(cell_graph, num_transformer_layers=2, heads_config=heads).eval()
    graph_p, batch_p = _relabeled(cell_graph, batch)
    model_p = _model(graph_p, num_transformer_layers=2, heads_config=heads).eval()
    model_p.load_state_dict(model.state_dict())
    assert model_p.gene_embedding is not None
    with torch.no_grad():
        model_p.gene_embedding.weight.copy_(model_p.gene_embedding.weight[PERM])
        pred, reps = model(cell_graph, batch)
        pred_p, reps_p = model_p(graph_p, batch_p)

    torch.testing.assert_close(pred_p, pred, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(reps_p["h_CLS"], reps["h_CLS"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        reps_p["H_genes"], reps["H_genes"][PERM], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        reps_p["H_genes_pert"], reps["H_genes_pert"][:, PERM], atol=1e-5, rtol=1e-5
    )
    heads_out, heads_p = reps["head_outputs"], reps_p["head_outputs"]
    torch.testing.assert_close(
        heads_p["global"], heads_out["global"], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        heads_p["per_gene"], heads_out["per_gene"][:, PERM], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        heads_p["per_metabolite"], heads_out["per_metabolite"], atol=1e-5, rtol=1e-5
    )
    assert heads_out["global"].shape == (B, 5)
    assert heads_out["per_gene"].shape == (B, N)
    assert heads_out["per_metabolite"].shape == (B, 3)


def test_perturbation_order_within_a_genotype_does_not_change_the_prediction(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """The perturbed set is a set: listing {0, 4, 5} as {5, 0, 4} changes nothing."""
    model = _model(cell_graph).eval()
    shuffled = batch.clone()
    shuffled["gene"].perturbation_indices = torch.tensor([2, 1, 3, 5, 0, 4])
    with torch.no_grad():
        pred, reps = model(cell_graph, batch)
        pred_s, reps_s = model(cell_graph, shuffled)
    torch.testing.assert_close(pred_s, pred, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        reps_s["H_genes_pert"], reps["H_genes_pert"], atol=1e-5, rtol=1e-5
    )


# --- parameter arithmetic ---------------------------------------------------------- #
def test_parameter_counts_are_the_hand_computed_layer_sums(
    cell_graph: HeteroData,
) -> None:
    """1 layer, hidden 16, every head on.

    gene_embedding 8*16 = 128; cls 16; encoder layer 3280 (module docstring);
    perturbation transform: MultiheadAttention in_proj 3*16*16 + 48 = 816 and out_proj
    272, FFN 16->64->16 = 2128, two LayerNorms 64, total 3280; perturbation head
    Linear(32, 16) + Linear(16, 1) = 528 + 17 = 545; global head (pooled, 5 outputs)
    528 + (16*5 + 5) = 613; per-gene head 272 + 17 = 289; per-metabolite head 272 + 17 =
    289. Total 128 + 16 + 3280 + 3280 + 545 + 613 + 289 + 289 = 8440, and it equals the
    trainable-parameter count of the module.
    """
    heads = {"global": {"output_dim": 5}, "per_gene": {}, "per_metabolite": {}}
    model = _model(cell_graph, heads_config=heads)
    assert model.num_parameters == {
        "gene_embedding": 128,
        "embedding_preprocessor": 0,
        "cls_token": 16,
        "transformer_layers": 3280,
        "perturbation_transform": 3280,
        "perturbation_head": 545,
        "global_head": 613,
        "per_gene_head": 289,
        "per_metabolite_head": 289,
        "total": 8440,
    }
    assert sum(p.numel() for p in model.parameters()) == 8440


def test_precomputed_embeddings_build_the_two_layer_preprocessor(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """A 5-wide precomputed embedding and no learnable table.

    ``learnable_embedding_config`` is None and the precomputed width is 5 > 0, so the
    learnable table is off and 5 != 16 builds the default 2-layer preprocessor:
    Linear(5, (5 + 16) // 2 = 10) = 60, LayerNorm(10) = 20, Linear(10, 16) = 176,
    LayerNorm(16) = 32, total 288. The forward reads ``cell_graph["gene"].x``, so a
    different ``x`` changes the prediction. With the first Linear's bias zeroed, ``x``
    reaches the model only through LayerNorm(W x), so scaling ``x`` by 3 leaves the
    prediction unchanged.
    """
    node_embeddings = {
        "toy": [Data(embeddings={"a": torch.zeros(1, 3), "b": torch.zeros(1, 2)})]
    }
    graph = cell_graph.clone()
    torch.manual_seed(1)
    graph["gene"].x = torch.randn(N, 5)
    model = _model(graph, node_embeddings=node_embeddings).eval()
    assert model.gene_embedding is None
    counts = model.num_parameters
    assert (counts["gene_embedding"], counts["embedding_preprocessor"]) == (0, 288)
    assert [type(m).__name__ for m in model.embedding_preprocessor or []] == [
        "Linear",
        "LayerNorm",
        "GELU",
        "Linear",
        "LayerNorm",
    ]  # dropout 0.0 adds no Dropout modules
    assert model.embedding_preprocessor is not None
    first = model.embedding_preprocessor[0]
    assert isinstance(first, nn.Linear)
    with torch.no_grad():
        pred, _ = model(graph, batch)
        other = graph.clone()
        other["gene"].x = other["gene"].x + 1.0
        pred_other, _ = model(other, batch)
        # with the first Linear's bias zeroed, x enters only as LayerNorm(W x), which is
        # invariant to scaling x by 3 (up to LayerNorm's eps), so the prediction is too
        first.bias.zero_()
        pred_unbiased, _ = model(graph, batch)
        scaled = graph.clone()
        scaled["gene"].x = scaled["gene"].x * 3.0
        pred_scaled, _ = model(scaled, batch)
    assert pred.shape == (B, 1)
    assert not torch.allclose(pred, pred_other)
    torch.testing.assert_close(pred_scaled, pred_unbiased, atol=1e-4, rtol=1e-4)


def test_learnable_and_precomputed_embeddings_concatenate(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Learnable width 4 plus precomputed width 5 is 9 wide; a 1-layer preprocessor is
    Linear(9, 16) + LayerNorm(16) = 160 + 32 = 192, and every parameter trains.
    """
    graph = cell_graph.clone()
    graph["gene"].x = torch.ones(N, 5)
    model = _model(
        graph,
        node_embeddings={"toy": [Data(embeddings={"a": torch.zeros(1, 5)})]},
        learnable_embedding_config={
            "enabled": True,
            "size": 4,
            "preprocessor": {"num_layers": 1, "dropout": 0.0},
        },
    )
    counts = model.num_parameters
    assert (counts["gene_embedding"], counts["embedding_preprocessor"]) == (32, 192)
    pred, _ = model(graph, batch)
    pred.sum().backward()
    assert _no_grad_parameters(model) == []


def test_learnable_table_narrower_than_hidden_is_projected(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Size 4 alone: 8 * 4 = 32 table entries; default preprocessor Linear(4, 10) = 50,
    LayerNorm(10) = 20, Linear(10, 16) = 176, LayerNorm(16) = 32, total 278.
    """
    model = _model(cell_graph, learnable_embedding_config={"enabled": True, "size": 4})
    counts = model.num_parameters
    assert (counts["gene_embedding"], counts["embedding_preprocessor"]) == (32, 278)
    assert model(cell_graph, batch)[0].shape == (B, 1)


def test_no_embedding_source_is_an_error_at_forward(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Learnable disabled and no precomputed embeddings leaves nothing to encode."""
    model = _model(cell_graph, learnable_embedding_config={"enabled": False})
    assert model.gene_embedding is None and model.embedding_preprocessor is None
    with pytest.raises(
        ValueError,
        match=re.escape(
            "No gene embeddings available (neither learnable nor pre-computed)"
        ),
    ):
        model(cell_graph, batch)


def test_gene_count_mismatch_between_graph_and_model_is_an_error(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """A 9-gene graph fed to an 8-gene model raises instead of scoring shifted genes."""
    model = _model(cell_graph)
    graph = cell_graph.clone()
    graph["gene"].num_nodes = 9
    with pytest.raises(
        ValueError,
        match=re.escape(
            "cell graph has 9 gene nodes but this model was built for gene_num=8."
        ),
    ):
        model(graph, batch)


# --- graph regularization ----------------------------------------------------------- #
def _graph_reg_model(cell_graph: HeteroData, **graph_reg: Any) -> CellGraphTransformer:
    config = {
        "regularized_heads": {"physical": {"layer": [0], "head": 1, "lambda": 0.5}}
    }
    config.update(graph_reg)
    model = _model(cell_graph, graph_reg_lambda=0.5, graph_regularization_config=config)
    layer = model.transformer_layers[0]
    assert isinstance(layer, GraphRegularizedTransformerLayer)
    with torch.no_grad():
        for proj in (layer.q_proj, layer.k_proj):
            proj.weight.zero_()
            proj.bias.zero_()
    return model


def test_graph_regularization_kl_has_the_closed_form_value(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Uniform attention against the 4-edge ``physical`` graph: 8 * 0.5 * log 9.

    Arithmetic in the module docstring; the edge count is cached once per graph. The
    loss is computed during a plain forward (no ``return_attention``) because lambda > 0
    needs the weights, and they are not returned.
    """
    model = _graph_reg_model(cell_graph)
    _, reps = model(cell_graph, batch)
    assert reps["graph_reg_loss"].item() == pytest.approx(
        8 * 0.5 * math.log(9), rel=1e-6
    )
    assert reps["attention_weights"] is None
    assert model._edge_count_cache == {"physical": 4}
    reps["graph_reg_loss"].backward()
    # the KL reads only the attention weights, so its gradient path is the embeddings
    # plus the Q and K projections of the regularized layer and nothing else
    reached = sorted(n for n, p in model.named_parameters() if p.grad is not None)
    assert reached == [
        "cls_token",
        "gene_embedding.weight",
        "transformer_layers.0.k_proj.bias",
        "transformer_layers.0.k_proj.weight",
        "transformer_layers.0.q_proj.bias",
        "transformer_layers.0.q_proj.weight",
    ]


def test_row_sampling_scales_the_kl_by_the_rows_kept(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Rate 0.25 keeps int(0.25 * 8) = 2 of the 4 positive-degree rows, each worth
    log 9, so the term halves to 0.5 * 2 log 9 / (4 / 8) = 2 log 9 whichever two are
    drawn. Rate 0.5 keeps int(4) = 4, not more than the 4 positive rows, so all stay.
    """
    torch.manual_seed(3)
    quarter = _graph_reg_model(cell_graph, row_sampling_rate=0.25)
    assert quarter(cell_graph, batch)[1]["graph_reg_loss"].item() == pytest.approx(
        2 * math.log(9), rel=1e-6
    )
    half = _graph_reg_model(cell_graph, row_sampling_rate=0.5)
    assert half(cell_graph, batch)[1]["graph_reg_loss"].item() == pytest.approx(
        4 * math.log(9), rel=1e-6
    )


def test_graph_regularization_skips_other_layers_and_edgeless_graphs(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """A head configured for layer 1 of a 1-layer model contributes nothing; a sampled
    graph with no edges has no positive rows and is skipped, so both totals are 0.
    """
    elsewhere = _graph_reg_model(
        cell_graph,
        regularized_heads={"physical": {"layer": 1, "head": 0, "lambda": 0.5}},
    )
    assert elsewhere(cell_graph, batch)[1]["graph_reg_loss"].item() == 0.0

    graph = cell_graph.clone()
    graph["gene", "empty", "gene"].edge_index = torch.zeros(2, 0, dtype=torch.long)
    edgeless = _model(
        graph,
        graph_reg_lambda=0.5,
        graph_regularization_config={
            "regularized_heads": {"empty": {"layer": 0, "head": 0, "lambda": 0.5}},
            "row_sampling_rate": 0.5,
        },
    )
    assert edgeless(graph, batch)[1]["graph_reg_loss"].item() == 0.0


def test_unknown_regularized_graph_raises_and_the_interaction_suffix_is_aliased(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """``physical_interaction`` is keyed under both spellings; a missing name raises.

    Rows of the dense normalized matrix are 1 / (degree + 1e-10), so a degree-1 row
    stores 1 / (1 + 1e-10), which is 1.0 in float32.
    """
    graph = HeteroData()
    graph["gene"].num_nodes = N
    graph["gene", "physical_interaction", "gene"].edge_index = torch.tensor(
        [[0, 0, 1], [1, 2, 2]]
    )
    model = _model(
        graph,
        graph_reg_lambda=0.5,
        graph_regularization_config={
            "regularized_heads": {"physical": {"layer": 0, "head": 0, "lambda": 0.5}}
        },
    )
    assert model.adjacency_matrices is not None
    assert sorted(model.adjacency_matrices) == ["physical", "physical_interaction"]
    expected = torch.zeros(N, N)
    expected[0, 1] = expected[0, 2] = 0.5
    expected[1, 2] = 1.0
    assert torch.equal(model.adjacency_matrices["physical"], expected)
    assert (
        model.adjacency_matrices["physical"]
        is model.adjacency_matrices["physical_interaction"]
    )

    missing = _model(
        cell_graph,
        graph_reg_lambda=0.5,
        graph_regularization_config={
            "regularized_heads": {"regulatory": {"layer": 0, "head": 0, "lambda": 0.5}}
        },
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "regularized head names 'regulatory' but cell_graph has no such gene-gene "
            "relation (available: ['physical']). to_cell_data SUFFIXES two of them: "
            "physical -> physical_interaction, regulatory -> regulatory_interaction."
        ),
    ):
        missing(cell_graph, batch)


def test_graph_regularization_is_zero_and_unconfigured_when_lambda_is_zero(
    cell_graph: HeteroData,
) -> None:
    """Lambda 0 ignores the config: no adjacency, no heads, sampling rate 1.0, and the
    loss short-circuits to exactly 0 for any attention tensor.
    """
    model = _model(
        cell_graph, graph_regularization_config={"regularized_heads": {"physical": {}}}
    )
    assert model.adjacency_matrices is None
    assert model.regularized_head_config is None
    assert model.row_sampling_rate == 1.0
    loss = model.compute_graph_regularization_loss(torch.rand(2, HEADS, N, N), 0)
    assert loss.item() == 0.0


# --- attention masking -------------------------------------------------------------- #
def test_attention_mask_burns_one_relation_into_one_head(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Head 0 follows ``physical`` symmetrized with self-loops and a free CLS row and
    column; heads 1-3 stay fully free. With weights requested, head 0 puts exactly 0 on
    every gene pair outside the mask.
    """
    model = _model(
        cell_graph,
        attention_mask_config={"enabled": True, "head_graphs": {"0": "physical"}},
    ).eval()
    mask = model.attention_head_mask
    assert mask is not None and mask.shape == (HEADS, N + 1, N + 1)
    expected = torch.eye(N + 1, dtype=torch.bool)
    expected[0, :] = True
    expected[:, 0] = True
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        expected[i + 1, j + 1] = expected[j + 1, i + 1] = True
    assert torch.equal(mask[0], expected)
    assert bool(mask[1:].all())
    assert model.attention_mask_layers == []

    with torch.no_grad():
        _, reps = model(cell_graph, batch, return_attention=True)
    head0 = reps["attention_weights"][0][0, 0]  # [N, N] gene block of head 0
    assert torch.all(head0[~expected[1:, 1:]] == 0.0)
    assert torch.all(head0[expected[1:, 1:]] > 0.0)
    assert reps["residual_update_ratios"] is not None
    assert len(reps["residual_update_ratios"]) == 1


def test_attention_mask_applies_only_to_listed_layers(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """layers=[1] masks layer 1 and leaves layer 0 dense (nonzero off-graph weight)."""
    model = _model(
        cell_graph,
        num_transformer_layers=2,
        attention_mask_config={
            "enabled": True,
            "layers": [1],
            "head_graphs": {0: "physical"},
        },
    ).eval()
    with torch.no_grad():
        _, reps = model(cell_graph, batch, return_attention=True)
    layer0, layer1 = reps["attention_weights"]
    assert layer0[0, 0, 0, 7].item() > 0.0  # genes 0 and 7 share no edge
    assert layer1[0, 0, 0, 7].item() == 0.0


def test_attention_mask_validation_messages(cell_graph: HeteroData) -> None:
    """An out-of-range layer and an unknown relation both raise, with their messages."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "attention_mask.layers=[0, 5] names layer(s) [5] outside the 1-layer encoder "
            "(use [] to mask every layer)"
        ),
    ):
        _model(cell_graph, attention_mask_config={"enabled": True, "layers": [0, 5]})
    with pytest.raises(
        ValueError,
        match=re.escape(
            "attention_mask head 2 names graph 'regulatory', which is not a gene-gene "
            "relation in cell_graph (available: ['physical']). The KL path skipped "
            "this silently; masking does not."
        ),
    ):
        _model(
            cell_graph,
            attention_mask_config={"enabled": True, "head_graphs": {"2": "regulatory"}},
        )


# --- decoder options that are exact identities at init ------------------------------ #
def test_closed_rezero_options_reproduce_the_plain_model(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Propagation and Perceiver mixing are built AFTER every plain-model parameter and
    start with a closed ReZero gate, so at the same seed the prediction is bit-identical
    to the plain model's.
    """
    plain = _model(cell_graph).eval()
    with torch.no_grad():
        expected, _ = plain(cell_graph, batch)
    options: list[dict[str, Any]] = [
        {"perturbation_propagation_config": {"enabled": True, "hops": 1}},
        {"post_perturbation_mixing_config": {"enabled": True, "num_latents": 4}},
    ]
    for option in options:
        model = _model(cell_graph, **option).eval()
        with torch.no_grad():
            got, _ = model(cell_graph, batch)
        assert torch.equal(got, expected), option


def test_propagation_uses_every_gene_relation_by_default_and_rejects_unknown_ones(
    cell_graph: HeteroData,
) -> None:
    """``graphs`` empty means every gene-gene relation; the transposed adjacency puts
    1 / deg(i) at (j, i) for each edge i -> j (all out-degrees are 1 here).
    """
    model = _model(cell_graph, perturbation_propagation_config={"enabled": True})
    assert list(model.adjacency_T_sparse) == ["physical"]
    expected = torch.zeros(N, N)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        expected[j, i] = 1.0
    assert torch.equal(model.adjacency_T_sparse["physical"].to_dense(), expected)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "perturbation_propagation.graphs ['string'] are not gene-gene relations in "
            "cell_graph (available: ['physical'])"
        ),
    ):
        _model(
            cell_graph,
            perturbation_propagation_config={"enabled": True, "graphs": ["string"]},
        )


def test_response_basis_leaves_the_per_gene_output_unchanged_at_init(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """The response basis is built after the per-gene head and its amplitude layer is
    zero-initialized, so the per-gene output equals the head-only model's exactly.
    """
    base = _model(cell_graph, heads_config={"per_gene": {}}).eval()
    with_basis = _model(
        cell_graph, heads_config={"per_gene": {"response_basis_rank": 3}}
    ).eval()
    with torch.no_grad():
        expected = base(cell_graph, batch)[1]["head_outputs"]["per_gene"]
        got = with_basis(cell_graph, batch)[1]["head_outputs"]["per_gene"]
    assert torch.equal(got, expected)


def test_hadamard_add_is_the_additive_operator_at_init() -> None:
    """``add`` = h + c + h * gamma(c) with gamma's last layer zero, so loading the ``off``
    operator's weights gives the ``off`` output exactly; ``replace`` drops the additive
    context, so at init every genotype sees the same (unperturbed) rows.
    """
    torch.manual_seed(0)
    h = torch.randn(N, D)
    idx, assign = torch.tensor([1, 2, 3, 0, 4, 5]), torch.tensor([0, 0, 1, 2, 2, 2])
    off = EquivariantPerturbationTransform(D, num_heads=HEADS, dropout=0.0).eval()
    add = EquivariantPerturbationTransform(
        D, num_heads=HEADS, dropout=0.0, hadamard="add"
    )
    missing, unexpected = add.load_state_dict(off.state_dict(), strict=False)
    assert sorted(missing) == [
        "hadamard_gamma.0.bias",
        "hadamard_gamma.0.weight",
        "hadamard_gamma.2.bias",
        "hadamard_gamma.2.weight",
    ]
    assert unexpected == []
    with torch.no_grad():
        expected, _ = off(h, idx, assign, B)
        got, _ = add.eval()(h, idx, assign, B)
        replaced, context = EquivariantPerturbationTransform(
            D, num_heads=HEADS, dropout=0.0, hadamard="replace"
        ).eval()(h, idx, assign, B)
    torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)
    assert torch.equal(replaced[0], replaced[1]) and torch.equal(
        replaced[1], replaced[2]
    )
    assert torch.equal(context, torch.zeros(B, N, D))
    with pytest.raises(
        ValueError,
        match=re.escape("hadamard must be 'off' | 'replace' | 'add', got 'mul'"),
    ):
        EquivariantPerturbationTransform(D, num_heads=HEADS, hadamard="mul")


def test_null_sink_scale_and_inert_sham() -> None:
    """Magnitude matching divides by sigmoid(-bias_init): 1 / sigmoid(4) = 1.0183157;
    a -20 frozen sink is numerically inert (weight ~2e-9) and matches the no-sink
    operator loaded with the same weights to 1e-6.
    """
    matched = EquivariantPerturbationTransform(
        D, num_heads=HEADS, null_sink=True, null_sink_magnitude_match=True
    )
    assert matched.null_scale == pytest.approx(1.0183156679659933, rel=1e-7)
    assert matched.null_bias.requires_grad
    torch.manual_seed(0)
    h = torch.randn(N, D)
    idx, assign = torch.tensor([1, 2, 3, 0, 4, 5]), torch.tensor([0, 0, 1, 2, 2, 2])
    ref = EquivariantPerturbationTransform(D, num_heads=HEADS, dropout=0.0).eval()
    sham = EquivariantPerturbationTransform(
        D,
        num_heads=HEADS,
        dropout=0.0,
        null_sink=True,
        null_sink_bias_init=-20.0,
        null_sink_trainable=False,
    ).eval()
    assert sham.load_state_dict(ref.state_dict(), strict=False).missing_keys == [
        "null_bias"
    ]
    assert not sham.null_bias.requires_grad and sham.null_scale == 1.0
    with torch.no_grad():
        torch.testing.assert_close(
            sham(h, idx, assign, B)[0], ref(h, idx, assign, B)[0], atol=1e-6, rtol=1e-6
        )
        open_sink = EquivariantPerturbationTransform(
            D, num_heads=HEADS, dropout=0.0, null_sink=True, null_sink_bias_init=0.0
        ).eval()
        open_sink.load_state_dict(ref.state_dict(), strict=False)
        assert not torch.allclose(
            open_sink(h, idx, assign, B)[0], ref(h, idx, assign, B)[0]
        )

        # matching rescales the attended context and nothing else: same weights, same
        # bias, the matched context is the unmatched one times 1 / sigmoid(4)
        unmatched = EquivariantPerturbationTransform(
            D, num_heads=HEADS, dropout=0.0, null_sink=True
        ).eval()
        matched.load_state_dict(unmatched.state_dict())
        torch.testing.assert_close(
            matched.eval()(h, idx, assign, B)[1],
            unmatched(h, idx, assign, B)[1] * matched.null_scale,
        )


def test_a_genotype_with_no_perturbation_gets_a_zero_context() -> None:
    """Assignments [0, 0, 2] leave sample 1 empty: its context is exactly zero and its
    rows are the operator applied to the wildtype with no attended input.
    """
    torch.manual_seed(0)
    h = torch.randn(N, D)
    module = EquivariantPerturbationTransform(D, num_heads=HEADS, dropout=0.0).eval()
    with torch.no_grad():
        out, context = module(h, torch.tensor([1, 2, 3]), torch.tensor([0, 0, 2]), 3)
        expected = module._apply_residual(h, torch.zeros(N, D), 0)
    assert out.shape == (3, N, D)
    assert torch.equal(context[1], torch.zeros(N, D))
    torch.testing.assert_close(out[1], expected)
    assert not torch.equal(context[0], torch.zeros(N, D))


def test_attention_only_extra_layers_and_the_rezero_identity() -> None:
    """``extra_layer_ffn_mult=0`` makes layers after the first attention-only
    (``nn.Identity``): for the single-gene genotype {3}, the 2-layer post-LN output is
    layer 0's full block followed by ``norm1_layers[1](h1 + attention(h1))`` with no FFN,
    recomputed here from the module's own sublayers. Rezero starts both betas at 0, so a
    2-layer rezero operator returns the wildtype rows for every genotype exactly.
    """
    torch.manual_seed(0)
    h = torch.randn(N, D)
    idx, assign = torch.tensor([1, 2, 3, 0, 4, 5]), torch.tensor([0, 0, 1, 2, 2, 2])
    postln = EquivariantPerturbationTransform(
        D, num_heads=HEADS, dropout=0.0, num_layers=2, extra_layer_ffn_mult=0
    )
    assert postln.ffn_mults == [4, 0]
    assert isinstance(postln.ffn_layers[1], nn.Identity)
    with torch.no_grad():
        out_postln, _ = postln.eval()(h, idx, assign, B)
        # genotype 1 perturbs {3} alone: layer 0 is the full post-LN block, layer 1 is
        # norm1(x + attention) with no FFN term
        key0 = h[[3]].unsqueeze(0)
        att0 = postln.cross_attn_layers[0](h.unsqueeze(0), key0, key0)[0].squeeze(0)
        h1 = postln._apply_residual(h, att0, 0)
        key1 = h1[[3]].unsqueeze(0)
        att1 = postln.cross_attn_layers[1](h1.unsqueeze(0), key1, key1)[0].squeeze(0)
        expected = postln.norm1_layers[1](h1 + att1)
    torch.testing.assert_close(out_postln[1], expected)
    rezero = EquivariantPerturbationTransform(
        D,
        num_heads=HEADS,
        dropout=0.0,
        residual="rezero",
        num_layers=2,
        extra_layer_ffn_mult=0,
    )
    out, _ = rezero(h, idx, assign, B)
    assert torch.equal(out, h.unsqueeze(0).expand(B, -1, -1))


# --- head options ------------------------------------------------------------------- #
def test_every_decoder_option_trains_and_the_unused_ones_are_named(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Kitchen-sink config: cross-attention global head with a distributional output,
    per-gene head with concat context, perturbed-set context, FiLM, bilinear features,
    free gene rows and a response basis, plus cross-gene and Perceiver mixing.

    Shapes: global [B, 5, 2] (param_dim 2), per_gene [B, N]. Backward through the sum of
    every output reaches every parameter except the observed-label encoder, which is
    only called when ``observed_values`` is passed.
    """
    model = _model(
        cell_graph,
        heads_config={
            "global": {
                "decoder": "s3_xattn",
                "output_dim": 5,
                "param_dim": 2,
                "use_ffn": False,
            },
            "per_gene": {
                "concat_context": True,
                "pert_set_context": True,
                "film_on_pert_set": True,
                "bilinear_rank": 2,
                "free_gene_dim": 3,
                "response_basis_rank": 2,
            },
        },
        perturbation_head_config={"pooling": "mean"},
        cross_gene_config={"enabled": True, "rank": 4},
        post_perturbation_mixing_config={
            "enabled": True,
            "num_latents": 4,
            "gate_mode": "on",
        },
        observed_label_config={"enabled": True},
    )
    assert model.pert_pooling == "mean"
    assert model.per_gene_head is not None and model.per_gene_head.in_mult == 5
    pred, reps = model(cell_graph, batch)
    heads = reps["head_outputs"]
    assert heads["global"].shape == (B, 5, 2)
    assert heads["per_gene"].shape == (B, N)
    (pred.sum() + heads["global"].sum() + heads["per_gene"].sum()).backward()
    assert _no_grad_parameters(model) == [
        "observed_label_encoder.proj.0.bias",
        "observed_label_encoder.proj.0.weight",
        "observed_label_encoder.proj.3.bias",
        "observed_label_encoder.proj.3.weight",
    ]

    model.zero_grad(set_to_none=True)
    observed = torch.zeros(B, N)
    observed_mask = torch.zeros(B, N, dtype=torch.bool)
    observed[0, 2], observed_mask[0, 2] = 0.7, True
    pred, reps = model(
        cell_graph, batch, observed_values=observed, observed_mask=observed_mask
    )
    heads = reps["head_outputs"]
    (pred.sum() + heads["global"].sum() + heads["per_gene"].sum()).backward()
    assert _no_grad_parameters(model) == []


def test_per_gene_vector_outputs_reshape_to_features_and_params() -> None:
    """The head's output is its MLP output, re-viewed, element for element.

    output_dim 3 with param_dim 2: the MLP emits 3 * 2 = 6 values per gene and
    ``view(batch, N, 3, 2)`` lays them out row-major, so ``out[..., k, p]`` is
    ``mlp(h)[..., 2k + p]``. output_dim 3 alone keeps the MLP's [B, N, 3] as is. A free
    gene row widens the MLP input to D + 2, and the head equals
    ``mlp(cat(h, free_gene_embedding expanded over the batch))`` squeezed to [B, N].
    Heads are in eval mode, so dropout (default 0.1) is the identity.
    """
    torch.manual_seed(0)
    h = torch.randn(B, N, D)
    with torch.no_grad():
        vector = PerGeneHead(D, output_dim=3, param_dim=2).eval()
        out, raw = vector(h), vector.mlp(h)
        assert out.shape == (B, N, 3, 2)
        for k in range(3):
            for p in range(2):
                assert torch.equal(out[..., k, p], raw[..., 2 * k + p])

        plain = PerGeneHead(D, output_dim=3).eval()
        assert torch.equal(plain(h), plain.mlp(h))

        free = PerGeneHead(D, free_gene_dim=2, num_genes=N).eval()
        assert free.free_gene_embedding is not None
        assert free.free_gene_embedding.shape == (N, 2)
        first = free.mlp[0]
        assert isinstance(first, nn.Linear) and first.in_features == D + 2
        widened = torch.cat(
            [h, free.free_gene_embedding.unsqueeze(0).expand(B, -1, -1)], dim=-1
        )
        assert torch.equal(free(h), free.mlp(widened).squeeze(-1))


def test_head_configuration_errors(cell_graph: HeteroData) -> None:
    """Unknown global decoder, unknown pooling, and a metabolite head with no rmr edges."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "unknown global head decoder 's2_mlp' (expected 's1_pool' or 's3_xattn')"
        ),
    ):
        _model(cell_graph, heads_config={"global": {"decoder": "s2_mlp"}})
    with pytest.raises(
        ValueError, match=re.escape("pooling must be 'sum' or 'mean', got 'max'")
    ):
        _model(cell_graph, perturbation_head_config={"pooling": "max"})
    graph = HeteroData()
    graph["gene"].num_nodes = N
    graph["gene", "gpr", "reaction"].edge_index = torch.tensor([[0], [0]])
    with pytest.raises(
        ValueError, match="per_metabolite head requested but cell_graph"
    ):
        _model(graph, heads_config={"per_metabolite": {}})


def test_metabolic_incidence_counts_nodes_from_edges_when_stores_lack_them() -> None:
    """Without reaction / metabolite ``num_nodes`` the counts come from the largest
    index + 1 (4 reactions, 3 metabolites), and each row is a mean over its members:
    r0 <- {0, 1} at 0.5, r1 <- {2} at 1, r2 <- {3, 4} at 0.5, r3 <- {5} at 1;
    m0 <- {r0, r1} at 0.5, m1 <- {r2} at 1, m2 <- {r3, r0} at 0.5.
    """
    graph = HeteroData()
    graph["gene"].num_nodes = N
    graph["gene", "gpr", "reaction"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 2, 3]]
    )
    graph["metabolite", "reaction", "metabolite"].edge_index = torch.tensor(
        [[0, 0, 1, 2, 2], [0, 1, 2, 3, 0]]
    )
    model = _model(graph, heads_config={"per_metabolite": {}})
    assert model.num_metabolites == 3
    gpr = torch.zeros(4, N)
    gpr[0, 0] = gpr[0, 1] = gpr[2, 3] = gpr[2, 4] = 0.5
    gpr[1, 2] = gpr[3, 5] = 1.0
    mr = torch.tensor(
        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.5, 0.0, 0.0, 0.5]]
    )
    gpr_incidence, mr_incidence = model.gpr_incidence_T, model.mr_incidence
    assert isinstance(gpr_incidence, torch.Tensor)
    assert isinstance(mr_incidence, torch.Tensor)
    assert torch.equal(gpr_incidence.to_dense(), gpr)
    assert torch.equal(mr_incidence.to_dense(), mr)
    assert "gpr_incidence_T" not in model.state_dict()  # non-persistent buffer


# --- masked multitask loss ---------------------------------------------------------- #
def test_masked_multitask_loss_paths_have_exact_values() -> None:
    """Predictions [[1, 2], [3, 4]] against targets [[0, 0], [0, 0]].

    * l1 over everything: (1 + 2 + 3 + 4) / 4 = 2.5;
    * l1 with feature mask [[T, T], [F, F]]: (1 + 2) / 2 = 1.5, which differs from the
      unmasked 2.5, so an ignored mask fails; mse with it: (1 + 4) / 2 = 2.5;
    * a head with no target is skipped; with no graph_reg_loss the anchor is 0, so the
      total is the weighted head loss: 2 * 2.5 = 5;
    * a ``point`` DistHead scores (params - target)^2 over the supervised row 1 only:
      (9 + 16) / 2 = 12.5.
    """
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.zeros(2, 2)
    fmask = torch.tensor([[True, True], [False, False]])

    total, per_head = MaskedMultitaskLoss(loss_fn="l1")(
        {"a": pred, "unsupervised": pred}, {"a": target}
    )
    assert total.item() == 2.5 and list(per_head) == ["a"]
    total, _ = MaskedMultitaskLoss(loss_fn="l1")(
        {"a": pred}, {"a": target}, feature_masks={"a": fmask}
    )
    assert total.item() == 1.5
    total, per_head = MaskedMultitaskLoss(head_weights={"a": 2.0})(
        {"a": pred}, {"a": target}, feature_masks={"a": fmask}
    )
    assert (total.item(), per_head["a"].item()) == (5.0, 2.5)

    dist = MaskedMultitaskLoss(dist_heads={"a": DistHead("point")})
    total, per_head = dist(
        {"a": pred}, {"a": target}, masks={"a": torch.tensor([False, True])}
    )
    assert (total.item(), per_head["a"].item()) == (12.5, 12.5)
