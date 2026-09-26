# tests/torchcell/models/test_equivariant_cell_graph_transformer_components.py
# [[tests.torchcell.models.test_equivariant_cell_graph_transformer_components]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_equivariant_cell_graph_transformer_components.py
"""Tensor-only components of the Cell Graph Transformer, one class each.

Sizes: 8 genes, hidden 16, 4 heads, a batch of 3 genotypes perturbing {1, 2}, {3} and
{0, 4, 5} (the conftest ``batch`` fixture). For a randomly initialized network exact
floats are not a contract across torch versions (Decision 12 of
[[plan.test-suite-buildout.2026.09.25]]); what is pinned: output shapes, finiteness, a
gradient on every parameter, seeded determinism, and the structural identities the
docstrings promise: ReZero gates start closed so the module is exactly the identity at
init, the perturbation transform is equivariant to a gene permutation, the two helper
functions have closed-form values.
"""

from typing import cast

import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

from torchcell.models.equivariant_cell_graph_transformer import (
    CellGraphTransformer,
    CrossAttnHead,
    CrossGeneMixing,
    EquivariantPerturbationTransform,
    GlobalHead,
    GraphRegularizedTransformerLayer,
    HyperSAGNN,
    LowRankBilinear,
    ObservedLabelEncoder,
    PerceiverMixing,
    PerGeneHead,
    PerMetaboliteHead,
    PerturbationGraphPropagation,
    PerturbationHead,
    ResponseBasisHead,
    calculate_weight_l2_norm,
    compute_smoothness,
)

N, D, HEADS, B = 8, 16, 4, 3
PERT_IDX = torch.tensor([1, 2, 3, 0, 4, 5])
PERT_BATCH = torch.tensor([0, 0, 1, 2, 2, 2])


def _h_genes(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(N, D)


def _h_pert(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, N, D)


def _all_params_get_grads(module: nn.Module, out: torch.Tensor) -> None:
    out.sum().backward()
    missing = [
        n for n, p in module.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert missing == []


def _row_normalized_transposed(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Sparse A_T with A[i, j] = 1 / deg(i) for each edge i -> j."""
    deg = torch.zeros(n).index_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
    values = 1.0 / deg[edge_index[0]]
    return torch.sparse_coo_tensor(edge_index.flip(0), values, (n, n)).coalesce()


# --- helpers ---------------------------------------------------------------------- #
def test_weight_l2_norm_is_the_euclidean_norm_of_all_trainable_weights() -> None:
    """Linear with weight [[3, 4]] and bias [0]: sqrt(9 + 16 + 0) = 5."""
    linear = nn.Linear(2, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[3.0, 4.0]]))
        linear.bias.zero_()
    assert calculate_weight_l2_norm(linear) == pytest.approx(5.0)
    linear.bias.requires_grad_(False)
    with torch.no_grad():
        linear.bias.fill_(12.0)  # frozen parameters are excluded
    assert calculate_weight_l2_norm(linear) == pytest.approx(5.0)


def test_smoothness_is_the_frobenius_norm_of_the_deviation_from_the_mean() -> None:
    """[[1, 1], [3, 3]] has mean [2, 2]; deviations are all +-1, so the norm is 2."""
    assert compute_smoothness(torch.tensor([[1.0, 1.0], [3.0, 3.0]])) == pytest.approx(
        2.0
    )
    assert compute_smoothness(torch.full((5, 3), 7.0)) == pytest.approx(0.0)


# --- encoder ---------------------------------------------------------------------- #
def test_transformer_layer_returns_gene_attention_without_the_cls_row() -> None:
    """Output keeps [B, N+1, D]; gene attention is [B, heads, N, N], rows summing to <= 1."""
    torch.manual_seed(0)
    layer = GraphRegularizedTransformerLayer(D, HEADS, dropout=0.0).eval()
    x = torch.randn(B, N + 1, D)
    out, attn = layer(x, return_attention=True)
    assert out.shape == (B, N + 1, D)
    assert attn is not None and attn.shape == (B, HEADS, N, N)
    # each gene row is a softmax over N+1 keys with the CLS column dropped
    assert torch.all(attn >= 0) and torch.all(attn.sum(-1) <= 1 + 1e-5)
    out_fast, none = layer(x, return_attention=False)
    assert none is None
    torch.testing.assert_close(out_fast, out, atol=1e-5, rtol=1e-5)


def test_transformer_layer_head_mask_blocks_attention_exactly() -> None:
    """A mask that allows only self-attention zeroes every off-diagonal weight."""
    torch.manual_seed(0)
    layer = GraphRegularizedTransformerLayer(D, HEADS, dropout=0.0).eval()
    x = torch.randn(B, N + 1, D)
    mask = torch.eye(N + 1, dtype=torch.bool).unsqueeze(0).expand(HEADS, -1, -1)
    _, attn = layer(x, return_attention=True, head_mask=mask)
    assert attn is not None
    off_diagonal = attn * (1 - torch.eye(N))
    assert off_diagonal.abs().max().item() == 0.0
    torch.testing.assert_close(attn.diagonal(dim1=-2, dim2=-1), torch.ones(B, HEADS, N))


def test_hyper_sagnn_at_init_is_the_set_mean_of_squared_static_residuals() -> None:
    """beta1 = beta2 = 0 at init, so the two attention branches vanish and the output is
    mean over each set of (embedding - static_embedding(embedding))^2; every parameter trains.
    """
    torch.manual_seed(0)
    module = HyperSAGNN(D, num_heads=HEADS)
    embeddings = torch.randn(PERT_IDX.numel(), D)
    out = module(embeddings, PERT_BATCH)
    assert out.shape == (B, D)
    with torch.no_grad():
        residual_sq = (embeddings - module.static_embedding(embeddings)) ** 2
        expected = torch.stack([residual_sq[PERT_BATCH == b].mean(0) for b in range(B)])
    torch.testing.assert_close(out.detach(), expected, atol=1e-6, rtol=1e-5)
    _all_params_get_grads(module, out)


def test_perturbation_transform_is_equivariant_to_a_gene_permutation() -> None:
    """Permuting the genes (and remapping the indices) permutes the output rows; the
    transform is not the identity and differs between genotypes.
    """
    torch.manual_seed(0)
    module = EquivariantPerturbationTransform(D, num_heads=HEADS, dropout=0.0).eval()
    h = _h_genes()
    with torch.no_grad():
        out, context = module(h, PERT_IDX, PERT_BATCH)
        perm = torch.tensor([7, 3, 5, 0, 1, 6, 2, 4])
        inverse = torch.argsort(perm)
        out_p, context_p = module(h[perm], inverse[PERT_IDX], PERT_BATCH)
    assert out.shape == (B, N, D) and context.shape == (B, N, D)
    torch.testing.assert_close(out_p, out[:, perm], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(context_p, context[:, perm], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(out[0], h)  # a real transformation of the wildtype rows
    assert not torch.allclose(out[0], out[1])  # different genotypes, different rows


def test_perturbation_transform_backward_and_rezero_variant() -> None:
    """Postln trains every parameter; rezero allocates two LayerNorms it never uses (a
    wart, pinned so a fix is noticed) and trains everything else.
    """
    torch.manual_seed(0)
    postln = EquivariantPerturbationTransform(D, num_heads=HEADS)
    out, _ = postln(_h_genes(), PERT_IDX, PERT_BATCH)
    _all_params_get_grads(postln, out)

    rezero = EquivariantPerturbationTransform(D, num_heads=HEADS, residual="rezero")
    out, _ = rezero(_h_genes(), PERT_IDX, PERT_BATCH)
    out.sum().backward()
    untrained = sorted(n for n, p in rezero.named_parameters() if p.grad is None)
    assert untrained == [
        "norm1_layers.0.bias",
        "norm1_layers.0.weight",
        "norm2_layers.0.bias",
        "norm2_layers.0.weight",
    ]
    assert rezero.beta_attn.grad is not None and rezero.beta_ffn.grad is not None
    with pytest.raises(ValueError, match="residual must be 'postln' or 'rezero'"):
        EquivariantPerturbationTransform(D, num_heads=HEADS, residual="preln")


def test_graph_propagation_rezero_gate_starts_as_the_identity() -> None:
    """Gate = 0 at init returns the input exactly; once open, the update is routed along
    the graph: in sample 1 (gene 3 perturbed, edge 3 -> 4) genes 3 and 4 receive their own
    updates and every other gene receives one identical bias-only update.
    """
    torch.manual_seed(0)
    module = PerturbationGraphPropagation(D, ["physical"], hops=1, dropout=0.0).eval()
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    adjacency = {"physical": _row_normalized_transposed(edge_index, N)}
    h = _h_pert()
    out = module(h, adjacency, PERT_IDX, PERT_BATCH)
    torch.testing.assert_close(out, h)
    with torch.no_grad():
        module.gate.fill_(1.0)
        update = module(h, adjacency, PERT_IDX, PERT_BATCH) - h  # [B, N, D]
    untouched = [0, 1, 2, 5, 6, 7]
    for gene in untouched[1:]:
        torch.testing.assert_close(update[1, gene], update[1, untouched[0]])
    assert not torch.allclose(update[1, 3], update[1, 0])
    assert not torch.allclose(update[1, 4], update[1, 0])
    assert not torch.allclose(update[1, 3], update[1, 4])
    opened = module(h, adjacency, PERT_IDX, PERT_BATCH)
    assert opened.shape == (B, N, D)
    _all_params_get_grads(module, opened)


def test_low_rank_bilinear_is_the_product_of_the_two_projections() -> None:
    """Rank 1 with u = e_0 and v = e_1 gives out[b, i, 0] = H[i, 0] * context[b, i, 1]."""
    torch.manual_seed(0)
    module = LowRankBilinear(D, rank=1)
    with torch.no_grad():
        module.u.weight.zero_()
        module.u.weight[0, 0] = 1.0
        module.v.weight.zero_()
        module.v.weight[0, 1] = 1.0
    h, context = _h_genes(), _h_pert()
    out = module(h, context)
    assert out.shape == (B, N, 1)
    expected = (h[:, 0].unsqueeze(0) * context[:, :, 1]).unsqueeze(-1)
    torch.testing.assert_close(out, expected)
    full = LowRankBilinear(D, rank=4)(h, context)
    assert full.shape == (B, N, 4)
    _all_params_get_grads(module, out)


def test_observed_label_encoder_changes_only_where_labels_are_observed() -> None:
    """Observing gene 2 of sample 0 changes only that row relative to the all-masked pass.

    Finding: the all-masked pass is NOT the identity the source docstring claims (the
    projection biases pass through the gate), so ``base`` differs from ``h`` everywhere.
    """
    torch.manual_seed(0)
    module = ObservedLabelEncoder(D, dropout=0.0).eval()
    h = _h_pert()
    values = torch.zeros(B, N)
    mask = torch.zeros(B, N)
    base = module(h, values, mask)
    assert torch.all((base - h).abs().sum(-1) > 1e-6)
    values[0, 2] = 0.7
    mask[0, 2] = 1.0
    observed = module(h, values, mask)
    changed = (observed - base).abs().sum(-1) > 1e-6  # [B, N]
    assert changed.tolist() == [[i == 2 for i in range(N)]] + [[False] * N] * 2
    assert observed.shape == (B, N, D)


def test_cross_gene_mixing_spreads_one_gene_to_every_gene_of_its_sample() -> None:
    """Shifting one gene row of sample 0 changes every row of sample 0 and none of samples 1, 2."""
    torch.manual_seed(0)
    module = CrossGeneMixing(D, rank=4, dropout=0.0).eval()
    h = _h_pert()
    with torch.no_grad():
        out = module(h)
        shifted = h.clone()
        shifted[0, 2] += 1.0
        out_shifted = module(shifted)
    assert out.shape == (B, N, D)
    changed = (out_shifted - out).abs().sum(-1) > 1e-6
    assert changed.tolist() == [[True] * N, [False] * N, [False] * N]
    _all_params_get_grads(module, module(h))


def test_response_basis_head_is_exactly_zero_at_init_then_responds() -> None:
    """The amplitude output layer is zero-initialized, so the head is exactly 0 at init for
    both output shapes; a nonzero amplitude turns it on.
    """
    torch.manual_seed(0)
    context = torch.randn(B, 2 * D)
    head = ResponseBasisHead(D, rank=4, dropout=0.0)
    out = head(_h_pert(), context)
    assert out.shape == (B, N)
    assert torch.equal(out, torch.zeros(B, N))
    dist = ResponseBasisHead(D, rank=4, param_dim=2, dropout=0.0)
    assert torch.equal(dist(_h_pert(), context), torch.zeros(B, N, 2))
    with torch.no_grad():
        cast(nn.Linear, head.amplitude[-1]).weight.normal_()
    out = head(_h_pert(), context)
    assert out.shape == (B, N) and not torch.equal(out, torch.zeros(B, N))
    _all_params_get_grads(head, out)


def test_perceiver_mixing_rezero_gate_starts_as_the_identity() -> None:
    """Gate = 0 returns the input exactly; gate_mode "on" mixes at once."""
    torch.manual_seed(0)
    closed = PerceiverMixing(D, num_latents=4, num_heads=HEADS, dropout=0.0)
    h = _h_pert()
    torch.testing.assert_close(closed(h), h)
    opened = PerceiverMixing(
        D, num_latents=4, num_heads=HEADS, dropout=0.0, gate_mode="on"
    )
    out = opened(h)
    assert out.shape == (B, N, D) and not torch.allclose(out, h)
    _all_params_get_grads(opened, out)


# --- heads ------------------------------------------------------------------------ #


def test_perturbation_head_sums_the_perturbed_tokens_per_genotype() -> None:
    """Sum pooling: reordering a set's indices leaves the output; an empty set scores
    mlp([h_CLS, 0]); the output is [B, 1] and every parameter trains.
    """
    torch.manual_seed(0)
    head = PerturbationHead(D, dropout=0.0).eval()
    h_cls, h = torch.randn(D), _h_pert()
    with torch.no_grad():
        out = head(h_cls, h, PERT_IDX, PERT_BATCH)
        reordered = head(h_cls, h, torch.tensor([2, 1, 3, 5, 0, 4]), PERT_BATCH)
        # sample 2 has no perturbed genes here: indices [1, 2 | 3 | -]
        with_empty = head(h_cls, h, torch.tensor([1, 2, 3]), torch.tensor([0, 0, 1]))
        expected_empty = head.mlp(torch.cat([h_cls, torch.zeros(D)]))
    assert out.shape == (B, 1)
    torch.testing.assert_close(reordered, out)
    torch.testing.assert_close(with_empty[2], expected_empty)
    torch.testing.assert_close(with_empty[:2], out[:2])
    _all_params_get_grads(head, head(h_cls, h, PERT_IDX, PERT_BATCH))


def test_global_head_without_gene_pool_reads_only_the_cls_token() -> None:
    """use_gene_pool=False gives every genotype the identical row; with the pool they differ."""
    torch.manual_seed(0)
    h_cls, h = torch.randn(D), _h_pert()
    cls_only = GlobalHead(D, 5, use_gene_pool=False, dropout=0.0).eval()(h_cls, h)
    assert cls_only.shape == (B, 5)
    assert torch.equal(cls_only[0], cls_only[1]) and torch.equal(
        cls_only[1], cls_only[2]
    )
    pooled = GlobalHead(D, 5, dropout=0.0).eval()(h_cls, h)
    assert not torch.equal(pooled[0], pooled[1])
    assert GlobalHead(D, 5, param_dim=2)(h_cls, h).shape == (B, 5, 2)


def test_cross_attention_head_is_invariant_to_gene_token_order() -> None:
    """Attention over the gene tokens is a set operation: a permutation leaves the output."""
    torch.manual_seed(0)
    head = CrossAttnHead(D, 5, num_heads=HEADS, dropout=0.0).eval()
    h_cls, h = torch.randn(D), _h_pert()
    perm = torch.tensor([7, 3, 5, 0, 1, 6, 2, 4])
    with torch.no_grad():
        out = head(h_cls, h)
        permuted = head(h_cls, h[:, perm])
    assert out.shape == (B, 5)
    torch.testing.assert_close(permuted, out, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(out[0], out[1])
    assert CrossAttnHead(D, 5, num_heads=HEADS, param_dim=2)(h_cls, h).shape == (
        B,
        5,
        2,
    )


def test_per_gene_head_shapes_and_film_conditioning() -> None:
    """[B, N] scalar; [B, N, 2] distributional; FiLM starts as the identity and then conditions."""
    torch.manual_seed(0)
    h = _h_pert()
    assert PerGeneHead(D)(h).shape == (B, N)
    assert PerGeneHead(D, param_dim=2)(h).shape == (B, N, 2)
    film = PerGeneHead(D, film_dim=D).eval()
    cond = torch.randn(B, D)
    with torch.no_grad():
        plain = film(h, film_cond=None)
        at_init = film(h, film_cond=cond)
        # the FiLM projection starts at zero (scale 1, shift 0), so conditioning is inert
        torch.testing.assert_close(at_init, plain)
        projection = cast(nn.Linear, cast(nn.Sequential, film.film)[2])
        projection.weight.normal_()
        projection.bias.normal_()
        conditioned = film(h, film_cond=cond)
    assert plain.shape == (B, N)
    assert not torch.allclose(plain, conditioned)


def test_per_metabolite_head_routes_genes_through_reactions() -> None:
    """Gene 0 feeds r0, which feeds metabolites 0 and 2: shifting gene 0 changes those two
    metabolites and leaves metabolite 1 (r2 <- {3, 4}) exactly unchanged.
    """
    torch.manual_seed(0)
    # reaction rows over genes: r0 <- {0, 1}, r1 <- {2}, r2 <- {3, 4}, r3 <- {5}, row-normalized
    gpr_t = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1, 2, 2, 3], [0, 1, 2, 3, 4, 5]]),
        torch.tensor([0.5, 0.5, 1.0, 0.5, 0.5, 1.0]),
        (4, 8),
    ).coalesce()
    # metabolite rows over reactions: m0 <- {r0, r1}, m1 <- {r2}, m2 <- {r3, r0}
    mr_incidence = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1, 2, 2], [0, 1, 2, 3, 0]]),
        torch.tensor([0.5, 0.5, 1.0, 0.5, 0.5]),
        (3, 4),
    ).coalesce()
    head = PerMetaboliteHead(D, dropout=0.0).eval()
    h = _h_pert()
    with torch.no_grad():
        out = head(h, gpr_t, mr_incidence)
        shifted = h.clone()
        shifted[:, 0] += 1.0
        out_shifted = head(shifted, gpr_t, mr_incidence)
    assert out.shape == (B, 3)
    assert not torch.allclose(out_shifted[:, 0], out[:, 0])
    assert not torch.allclose(out_shifted[:, 2], out[:, 2])
    torch.testing.assert_close(out_shifted[:, 1], out[:, 1])
    _all_params_get_grads(head, head(h, gpr_t, mr_incidence))


# --- the assembled model ---------------------------------------------------------- #
def test_full_model_backward_determinism_and_attention_output(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Forward gives [B, 1]; every parameter trains; a seed pins the output; attention is per layer."""

    def build() -> CellGraphTransformer:
        torch.manual_seed(7)
        return CellGraphTransformer(
            gene_num=N,
            hidden_channels=D,
            num_transformer_layers=2,
            num_attention_heads=HEADS,
            cell_graph=cell_graph,
            dropout=0.0,
        )

    model = build()
    predictions, reps = model(cell_graph, batch)
    assert predictions.shape == (B, 1)
    _all_params_get_grads(model, predictions)
    assert reps["attention_weights"] is None
    assert reps["H_genes_pert"].shape == (B, N, D)
    assert reps["h_CLS"].shape == (D,)

    again, _ = build()(cell_graph, batch)
    torch.testing.assert_close(again, predictions.detach())

    model.eval()
    with torch.no_grad():
        _, with_attention = model(cell_graph, batch, return_attention=True)
    weights = with_attention["attention_weights"]
    assert weights is not None and len(weights) == 2
    assert all(w.shape == (1, HEADS, N, N) for w in weights)
    assert len(with_attention["residual_update_ratios"]) == 2
