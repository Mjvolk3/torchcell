# tests/torchcell/models/test_cell_graph_transformer.py
# [[tests.torchcell.models.test_cell_graph_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_cell_graph_transformer.py
"""The 006 ``CellGraphTransformer`` (torchcell/models/cell_graph_transformer.py).

Fixtures: the conftest ``cell_graph`` (8 genes; a ``physical`` gene-gene relation with
edges 0->1, 1->2, 2->3, 3->4; ``gpr`` and a metabolite relation, which the model must
ignore) and ``batch`` (3 genotypes perturbing {1, 2}, {3}, {0, 4, 5}). Model: hidden 16,
4 heads, 1 layer, dropout 0, so a forward is a pure function of the weights.

A randomly initialized network carries no float contract, so what is pinned is:

* every block on hand-set weights against a numpy oracle: the encoder attention (scale
  1/sqrt(head_dim), softmax over keys, head h = feature slice [h*hd, (h+1)*hd)), the
  HyperSAGNN masked softmax (same set, never self; a two-gene set attends 1.0 to the
  partner, a singleton row is all -inf and becomes 0 after ``nan_to_num``), the
  cross-attention readout ``mlp([h_CLS || z_S])``;
* parameter counts by hand: Linear(i, o) = i*o + o, LayerNorm(d) = 2d; encoder layer
  4 * 272 + 2 * 32 + (16*64 + 64) + (64*16 + 16) = 3280; cross-attention
  in_proj 3*16*16 + 48 = 816 plus out_proj 272 = 1088; readout MLP Linear(32, 16) 528 +
  Linear(16, 1) 17 = 545; HyperSAGNN static 272 + 8 * 272 + 2 betas = 2450;
* the graph-regularization KL in closed form: with the layer's Q and K zeroed every score
  is 0, each gene row attends 1/9 to each of the 9 tokens, and a degree-1 row of the
  row-normalized adjacency contributes -log(1/9 + 1e-8) ~= log 9;
* relabeling genes permutes per-gene outputs and leaves predictions and the reg loss;
  a batch of 3 equals three single-genotype runs; seeded determinism; gradient reach.

Where the 006 model differs from its successor
(torchcell/models/equivariant_cell_graph_transformer.py) the difference is pinned: the
006 encoder never sees the perturbation (h_CLS is batch-independent), the genotype count
is ``max(batch_assignment) + 1`` (``num_graphs`` is ignored), the edge divisor counts 2
per graph, and an unknown regularized-graph name is skipped silently.
"""

import copy
import math
import re
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import nn
from torch_geometric.data import HeteroData

from torchcell.data.cell_data import GENE_GRAPH_EDGE_RELATION
from torchcell.models.cell_graph_transformer import (
    CellGraphTransformer,
    GraphRegularizedTransformerLayer,
    HyperSAGNN,
    PerturbationHead,
    calculate_weight_l2_norm,
    compute_smoothness,
)

N, D, HEADS = 8, 16, 4
PERT_IDX = [1, 2, 3, 0, 4, 5]
PERT_BATCH = [0, 0, 1, 2, 2, 2]
SETS = [[1, 2], [3], [0, 4, 5]]
PERM = torch.tensor([7, 3, 5, 0, 1, 6, 2, 4])  # new position j holds old gene PERM[j]
INVERSE = torch.argsort(PERM)  # old gene g sits at new position INVERSE[g]
LOG9 = -math.log(1.0 / 9.0 + 1e-8)  # KL of a degree-1 row against uniform 1/9


@pytest.fixture(autouse=True)
def _restore_global_rng() -> Iterator[None]:
    """Tests seed the global torch RNG (and dropout draws from it); restore it after."""
    with torch.random.fork_rng(devices=[]):
        yield


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
        graph_reg_scale=0.0,
    )
    config.update(kwargs)
    return CellGraphTransformer(**config).eval()


def _reg_config(
    heads: dict[str, dict[str, Any]], row_sampling_rate: float = 1.0
) -> dict[str, Any]:
    return {"regularized_heads": heads, "row_sampling_rate": row_sampling_rate}


def _batch_of(sets: list[list[int]]) -> HeteroData:
    batch = HeteroData()
    batch["gene"].perturbation_indices = torch.tensor(
        [g for s in sets for g in s], dtype=torch.long
    )
    batch["gene"].perturbation_indices_batch = torch.tensor(
        [b for b, s in enumerate(sets) for _ in s], dtype=torch.long
    )
    batch.num_graphs = len(sets)
    return batch


def _zero_qk(layer: nn.Module) -> None:
    assert isinstance(layer, GraphRegularizedTransformerLayer)
    with torch.no_grad():
        for proj in (layer.q_proj, layer.k_proj):
            proj.weight.zero_()
            proj.bias.zero_()


def _set_identity(linear: nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.copy_(torch.eye(linear.weight.size(0)))
        linear.bias.zero_()


def _np_softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    out: NDArray[np.float64] = e / e.sum(axis=-1, keepdims=True)
    return out


def _np_layer_norm(x: NDArray[np.float64], eps: float = 1e-5) -> NDArray[np.float64]:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    out: NDArray[np.float64] = (x - mean) / np.sqrt(var + eps)
    return out


def _no_grad(model: nn.Module) -> list[str]:
    return sorted(
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    )


# --- helpers ---------------------------------------------------------------------- #
def test_weight_l2_norm_skips_frozen_parameters() -> None:
    """Weight [[3, 4]], bias [0]: sqrt(9 + 16) = 5; a frozen bias of 12 is not counted."""
    linear = nn.Linear(2, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[3.0, 4.0]]))
        linear.bias.zero_()
    assert calculate_weight_l2_norm(linear) == pytest.approx(5.0)
    linear.bias.requires_grad_(False)
    with torch.no_grad():
        linear.bias.fill_(12.0)
    assert calculate_weight_l2_norm(linear) == pytest.approx(5.0)


def test_smoothness_is_the_frobenius_norm_of_the_deviation_from_the_mean() -> None:
    """[[0, 0], [2, 4]] has mean [1, 2]; deviations +-1, +-2: sqrt(1+4+1+4) = sqrt 10."""
    x = torch.tensor([[0.0, 0.0], [2.0, 4.0]])
    assert compute_smoothness(x) == pytest.approx(math.sqrt(10.0))
    assert compute_smoothness(torch.full((4, 3), -2.0)) == 0.0


# --- refusals --------------------------------------------------------------------- #
def test_head_count_must_divide_the_width_with_the_exact_messages() -> None:
    """Both attention blocks refuse a width that does not split into heads."""
    with pytest.raises(
        AssertionError,
        match=re.escape("hidden_dim 10 must be divisible by num_heads 4"),
    ):
        GraphRegularizedTransformerLayer(10, 4)
    with pytest.raises(
        AssertionError,
        match=re.escape("hidden_channels 10 must be divisible by num_heads 4"),
    ):
        HyperSAGNN(10, num_heads=4)


def test_default_graph_reg_scale_without_a_config_fails_at_forward(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Finding: the constructor defaults are ``graph_reg_scale=0.001`` and
    ``graph_regularization_config=None``. __init__ then stores no adjacency (the branch
    needs both, cell_graph_transformer.py:450), but the loss only short-circuits on scale
    0 (line 543), so the first forward hits the bare ``assert self.adjacency_matrices is
    not None`` (line 547) with an empty message. Pinned until __init__ refuses the
    combination or the loss returns 0 whenever no adjacency was built.
    """
    model = _model(cell_graph, graph_reg_scale=0.001)
    assert model.adjacency_matrices is None
    with pytest.raises(AssertionError, match="^$"):
        model(cell_graph, batch)


# --- encoder layer ---------------------------------------------------------------- #
def test_encoder_attention_matches_a_numpy_oracle_on_identity_projections() -> None:
    """D = 4, 2 heads (head_dim 2), seq 3 (CLS + 2 genes), Q = K = V = out = identity.

    Per head h on feature slice [2h, 2h+2): scores = x_h x_h^T / sqrt(2), softmax over the
    key axis. The returned gene attention is the [1:, 1:] block of those weights (rows do
    not renormalize after the CLS column is dropped).
    """
    layer = GraphRegularizedTransformerLayer(4, 2, dropout=0.0).eval()
    for proj in (layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj):
        _set_identity(proj)
    x = torch.tensor(
        [[[0.5, -1.0, 2.0, 0.0], [1.0, 1.0, -1.0, 0.5], [-2.0, 0.5, 0.0, 1.5]]]
    )
    _, attn = layer(x, return_attention=True)
    xn = x[0].double().numpy()
    expected = np.stack(
        [
            _np_softmax(
                xn[:, 2 * h : 2 * h + 2] @ xn[:, 2 * h : 2 * h + 2].T / math.sqrt(2)
            )
            for h in range(2)
        ]
    )[:, 1:, 1:]
    assert attn is not None
    torch.testing.assert_close(
        attn[0].double(), torch.from_numpy(expected), atol=1e-6, rtol=1e-6
    )


def test_encoder_layer_closed_form_with_uniform_attention_and_a_dead_ffn() -> None:
    """Q = K = 0 makes every score 0, so every token attends 1/3 to each of 3 tokens;
    V = out = identity gives attended = mean_t x_t. The FFN's last Linear is zeroed, so
    output = LN2(LN1(x + mean(x))) with unit-gain, zero-bias norms (numpy oracle).
    The returned gene attention is 1/3 everywhere (2 x 2 block per head), and
    ``return_attention=False`` returns None with the same output.
    """
    layer = GraphRegularizedTransformerLayer(4, 2, dropout=0.0).eval()
    _zero_qk(layer)
    _set_identity(layer.v_proj)
    _set_identity(layer.out_proj)
    last = layer.ffn[3]
    assert isinstance(last, nn.Linear)
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    x = torch.tensor(
        [[[0.5, -1.0, 2.0, 0.0], [1.0, 1.0, -1.0, 0.5], [-2.0, 0.5, 0.0, 1.5]]]
    )
    out, attn = layer(x, return_attention=True)
    xn = x[0].double().numpy()
    expected = _np_layer_norm(_np_layer_norm(xn + xn.mean(axis=0, keepdims=True)))
    torch.testing.assert_close(
        out[0].double(), torch.from_numpy(expected), atol=1e-5, rtol=1e-5
    )
    assert attn is not None
    torch.testing.assert_close(attn, torch.full((1, 2, 2, 2), 1.0 / 3.0))
    out_fast, none = layer(x, return_attention=False)
    assert none is None
    torch.testing.assert_close(out_fast, out)


def test_encoder_layer_ignores_the_adjacency_it_is_handed() -> None:
    """Finding: the layer stores ``adjacency_matrices`` and ``regularized_head_config``
    (cell_graph_transformer.py:73-74) but forward never reads them: attention is
    unmasked and identical with or without a graph. The graph enters only through the
    KL loss in ``CellGraphTransformer.compute_graph_regularization_loss``. Pinned until
    the layer either uses or stops accepting them.
    """
    torch.manual_seed(0)
    plain = GraphRegularizedTransformerLayer(D, HEADS, dropout=0.0).eval()
    with_graph = GraphRegularizedTransformerLayer(
        D,
        HEADS,
        adjacency_matrices={"physical": torch.eye(N)},
        regularized_head_config={"physical": {"layer": 0, "head": 0, "lambda": 1.0}},
        dropout=0.0,
    ).eval()
    with_graph.load_state_dict(plain.state_dict())
    x = torch.randn(1, N + 1, D)
    out_a, attn_a = plain(x, return_attention=True)
    out_b, attn_b = with_graph(x, return_attention=True)
    assert torch.equal(out_a, out_b)
    assert attn_a is not None and attn_b is not None
    assert torch.equal(attn_a, attn_b)


# --- HyperSAGNN ------------------------------------------------------------------- #
def test_hypersagnn_masked_softmax_matches_a_hand_computed_oracle() -> None:
    """One head, Q = K = V = O = identity, beta = 1: out_i = x_i + sum_j a_ij x_j with
    a_ij = softmax over allowed j of x_i . x_j / sqrt(4), allowed = same set and j != i.

    Sets {0, 1}, {2}, {3, 4, 5}. Gene 0's only allowed key is 1, so out_0 = x_0 + x_1
    exactly; the singleton gene 2 has an all -inf row, softmax gives NaN, ``nan_to_num``
    gives 0, so out_2 = x_2; genes 3..5 use a two-key softmax (numpy oracle). A mask of
    the opposite polarity would let gene 0 attend to itself and to genes 2..5.
    """
    module = HyperSAGNN(4, num_heads=1)
    x = torch.tensor(
        [
            [1.0, 0.0, -1.0, 2.0],
            [0.5, 1.5, 0.0, -1.0],
            [2.0, -2.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [-1.0, 0.5, 2.0, 0.5],
            [1.0, 1.0, -0.5, 0.0],
        ]
    )
    sets = torch.tensor([0, 0, 1, 2, 2, 2])
    for proj in (module.Q1, module.K1, module.V1, module.O1):
        _set_identity(proj)
    with torch.no_grad():
        module.beta1.fill_(1.0)
    same = sets.unsqueeze(-1) == sets.unsqueeze(0)
    mask = same & ~torch.eye(6, dtype=torch.bool)
    out = module._global_attention_layer(
        x, mask, module.Q1, module.K1, module.V1, module.O1, module.beta1
    )
    xn = x.double().numpy()
    expected = xn.copy()
    expected[0] = xn[0] + xn[1]
    expected[1] = xn[1] + xn[0]
    for i in (3, 4, 5):
        keys = [j for j in (3, 4, 5) if j != i]
        weights = _np_softmax(np.array([xn[i] @ xn[j] / 2.0 for j in keys]))
        expected[i] = xn[i] + sum(w * xn[j] for w, j in zip(weights, keys, strict=True))
    torch.testing.assert_close(
        out.detach().double(), torch.from_numpy(expected), atol=1e-6, rtol=1e-6
    )


def test_hypersagnn_forward_builds_the_same_set_not_self_mask() -> None:
    """With both ReZero gates open (beta1 = beta2 = 0.5) the attention branches matter,
    and the mask forward builds must keep sets apart: the rows of the batched call equal
    the call on each set alone ({0, 1}, {2}, {3, 4, 5}), and the singleton row equals the
    one-gene call, whose every attention row is empty (it sees only beta * O(0) = beta *
    bias). Equivalently the output equals the two layers applied with the hand-built mask
    ``same set & not self``.
    """
    torch.manual_seed(0)
    module = HyperSAGNN(D, num_heads=HEADS)
    with torch.no_grad():
        module.beta1.fill_(0.5)
        module.beta2.fill_(0.5)
    e = torch.randn(6, D)
    sets = torch.tensor([0, 0, 1, 2, 2, 2])
    with torch.no_grad():
        out = module(e, sets)
        alone = torch.cat(
            [
                module(e[:2], torch.zeros(2, dtype=torch.long)),
                module(e[2:3], torch.zeros(1, dtype=torch.long)),
                module(e[3:], torch.zeros(3, dtype=torch.long)),
            ]
        )
        mask = (sets.unsqueeze(-1) == sets.unsqueeze(0)) & ~torch.eye(
            6, dtype=torch.bool
        )
        h = module._global_attention_layer(
            e, mask, module.Q1, module.K1, module.V1, module.O1, module.beta1
        )
        h = module._global_attention_layer(
            h, mask, module.Q2, module.K2, module.V2, module.O2, module.beta2
        )
        sq = (h - module.static_embedding(e)) ** 2
        expected = torch.stack([sq[:2].mean(0), sq[2], sq[3:].mean(0)])
    torch.testing.assert_close(out, alone, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_hypersagnn_at_init_is_the_set_mean_of_squared_static_residuals() -> None:
    """beta1 = beta2 = 0 at init, so both attention layers return their input and the
    output is mean over each set of (e - ReLU(W e + b))^2, sets {0,1}, {2}, {3,4,5}.
    """
    torch.manual_seed(0)
    module = HyperSAGNN(D, num_heads=HEADS)
    e = torch.randn(6, D)
    sets = torch.tensor([0, 0, 1, 2, 2, 2])
    out = module(e, sets)
    w = module.static_embedding[0]
    assert isinstance(w, nn.Linear)
    with torch.no_grad():
        static = torch.relu(e @ w.weight.T + w.bias)
        sq = (e - static) ** 2
        expected = torch.stack([sq[:2].mean(0), sq[2], sq[3:].mean(0)])
    torch.testing.assert_close(out.detach(), expected, atol=1e-6, rtol=1e-6)


# --- perturbation head ------------------------------------------------------------ #
def test_cross_attention_head_matches_a_numpy_oracle() -> None:
    """One head, in_proj = [I; I; I], out_proj = I, zero biases. For genotype b,
    q_b = mean of its perturbed genes' rows of H[1:], z_b = softmax_j(q_b . h_j / 4) @ H[1:]
    (sqrt(16) = 4, all 8 genes are keys, perturbed ones included), and the prediction is
    W2 ReLU(W1 [h_CLS || z_b] + b1) + b2 with the module's own MLP weights read out as
    numpy arrays.
    """
    torch.manual_seed(0)
    head = PerturbationHead(D, num_heads=1, dropout=0.0).eval()
    with torch.no_grad():
        head.cross_attn.in_proj_weight.copy_(torch.eye(D).repeat(3, 1))
        head.cross_attn.in_proj_bias.zero_()
        head.cross_attn.out_proj.weight.copy_(torch.eye(D))
        head.cross_attn.out_proj.bias.zero_()
    H = torch.randn(N + 1, D)
    pred = head(H, torch.tensor(PERT_IDX), torch.tensor(PERT_BATCH))
    h = H.double().numpy()
    genes = h[1:]
    l1, l2 = head.mlp[0], head.mlp[3]
    assert isinstance(l1, nn.Linear) and isinstance(l2, nn.Linear)
    w1, b1 = l1.weight.detach().double().numpy(), l1.bias.detach().double().numpy()
    w2, b2 = l2.weight.detach().double().numpy(), l2.bias.detach().double().numpy()
    expected = []
    for s in SETS:
        q = genes[s].mean(axis=0)
        z = _np_softmax(genes @ q / 4.0) @ genes
        hidden = np.maximum(w1 @ np.concatenate([h[0], z]) + b1, 0.0)
        expected.append(w2 @ hidden + b2)
    torch.testing.assert_close(
        pred.detach().double(),
        torch.from_numpy(np.stack(expected)),
        atol=1e-5,
        rtol=1e-5,
    )


def test_hypersagnn_head_reads_the_rows_after_the_cls_token() -> None:
    """The HyperSAGNN head's z_S is HyperSAGNN(H[1:][indices]) and the MLP input is
    [h_CLS || z_S]: perturbed index 0 is row 1 of H, never the CLS row 0.
    """
    torch.manual_seed(0)
    head = PerturbationHead(D, num_heads=HEADS, dropout=0.0, use_cross_attention=False)
    H = torch.randn(N + 1, D)
    pred = head(H, torch.tensor(PERT_IDX), torch.tensor(PERT_BATCH))
    with torch.no_grad():
        z = head.hypersagnn(H[1:][torch.tensor(PERT_IDX)], torch.tensor(PERT_BATCH))
        expected = head.mlp(torch.cat([H[0].expand(3, -1), z], dim=-1))
    assert torch.equal(pred, expected)


def test_genotype_count_is_max_assignment_plus_one() -> None:
    """Finding: the head sizes the batch as ``max(batch_assignment) + 1``
    (cell_graph_transformer.py:360). This test asserts the MIDDLE-empty case
    (assignment [0, 0, 2], genotype 1 empty), which differs by head: cross-attention
    returns 3 rows, the empty genotype's from a zero query, while HyperSAGNN sizes its
    scatter by the number of distinct sets (2) and raises on set index 2. The trailing
    case (``num_graphs`` ignored, 3 rows for 4 genotypes) is asserted in
    ``test_model_drops_a_trailing_empty_genotype_despite_num_graphs``. The successor model takes ``batch_size`` from ``num_graphs`` and
    refuses an out-of-range assignment. Pinned until the 006 head is given the genotype
    count.
    """
    torch.manual_seed(0)
    cross = PerturbationHead(D, num_heads=HEADS, dropout=0.0).eval()
    hyper = PerturbationHead(D, num_heads=HEADS, dropout=0.0, use_cross_attention=False)
    H = torch.randn(N + 1, D)
    middle_empty = (torch.tensor([1, 2, 3]), torch.tensor([0, 0, 2]))
    assert cross(H, *middle_empty).shape == (3, 1)
    with pytest.raises(
        RuntimeError,
        match=re.escape("index 2 is out of bounds for dimension 0 with size 2"),
    ):
        hyper(H, *middle_empty)


def test_model_drops_a_trailing_empty_genotype_despite_num_graphs(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Same finding at model level: ``num_graphs = 4`` with three populated genotypes
    gives 3 predictions, identical to ``num_graphs = 3``.
    """
    model = _model(cell_graph)
    with torch.no_grad():
        three, _ = model(cell_graph, batch)
        batch.num_graphs = 4
        four, _ = model(cell_graph, batch)
    assert four.shape == (3, 1)
    assert torch.equal(three, four)


# --- adjacency normalization ------------------------------------------------------ #
def test_adjacency_normalization_keeps_gene_relations_only_row_normalized(
    cell_graph: HeteroData,
) -> None:
    """Only (gene, *, gene) relations are kept, keyed by relation name. A duplicate edge
    is counted once (assignment, not addition): physical edges 0->1, 1->2, 2->3, 3->4 plus
    an extra regulatory relation 5->6, 5->7, 5->7 give row 5 = [.., 1/2, 1/2] and every
    other entry 0 (zero rows stay zero: 0 / (0 + 1e-10) = 0).
    """
    cell_graph["gene", "regulatory", "gene"].edge_index = torch.tensor(
        [[5, 5, 5], [6, 7, 7]]
    )
    model = _model(
        cell_graph, graph_reg_scale=1.0, graph_regularization_config=_reg_config({})
    )
    assert model.adjacency_matrices is not None
    assert sorted(model.adjacency_matrices) == ["physical", "regulatory"]
    physical = torch.zeros(N, N)
    physical[[0, 1, 2, 3], [1, 2, 3, 4]] = 1.0
    regulatory = torch.zeros(N, N)
    regulatory[5, 6] = regulatory[5, 7] = 0.5
    torch.testing.assert_close(model.adjacency_matrices["physical"], physical)
    torch.testing.assert_close(model.adjacency_matrices["regulatory"], regulatory)


# --- graph regularization loss ---------------------------------------------------- #
def test_graph_reg_loss_closed_form_and_the_two_per_graph_edge_divisor(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Finding: the edge divisor ``len(A.nonzero()[0])`` (cell_graph_transformer.py:608-
    611) is the length of the FIRST nonzero coordinate pair, i.e. 2, not the edge count.

    Setup: Q = K = 0 in the one layer, so every gene row attends 1/9 to each token.
    ``physical`` has 4 degree-1 rows, each contributing -log(1/9 + 1e-8) = L; batchmean
    divides by the batch dimension (1), so KL = 4L, and the loss is
    lambda * 4L * scale / (total_edges / N). With lambda 0.5, scale 0.01, N = 8:
    total_edges = 2 gives 0.5 * 4L * 0.01 * 4 = 0.08 L, while the 4 real edges would give
    0.04 L. Adding an unregularized ``regulatory`` relation with one edge adds 2 more to
    the divisor (0.04 L; the true count 5 would give 0.032 L).
    Pinned until the divisor counts edges (the successor uses ``(A > 0).sum()`` per graph).
    """
    heads = {"physical": {"layer": 0, "head": 2, "lambda": 0.5}}
    model = _model(
        cell_graph, graph_reg_scale=0.01, graph_regularization_config=_reg_config(heads)
    )
    _zero_qk(model.transformer_layers[0])
    with torch.no_grad():
        _, out = model(cell_graph, batch)
    assert out["graph_reg_loss"].item() == pytest.approx(0.08 * LOG9, rel=1e-6)

    cell_graph["gene", "regulatory", "gene"].edge_index = torch.tensor([[6], [7]])
    model = _model(
        cell_graph, graph_reg_scale=0.01, graph_regularization_config=_reg_config(heads)
    )
    _zero_qk(model.transformer_layers[0])
    with torch.no_grad():
        _, out = model(cell_graph, batch)
    assert out["graph_reg_loss"].item() == pytest.approx(0.04 * LOG9, rel=1e-6)


def test_a_relation_with_no_edges_breaks_the_edge_divisor(
    cell_graph: HeteroData,
) -> None:
    """Finding (same lines): an empty gene relation has ``A.nonzero()`` of shape [0, 2],
    so ``[0]`` raises IndexError on every regularized forward. Pinned with the divisor.
    """
    cell_graph["gene", "regulatory", "gene"].edge_index = torch.zeros(
        2, 0, dtype=torch.long
    )
    heads = {"physical": {"layer": 0, "head": 0, "lambda": 1.0}}
    model = _model(
        cell_graph, graph_reg_scale=0.01, graph_regularization_config=_reg_config(heads)
    )
    with pytest.raises(
        IndexError,
        match=re.escape("index 0 is out of bounds for dimension 0 with size 0"),
    ):
        model.compute_graph_regularization_loss(torch.full((1, HEADS, N, N), 1 / 9), 0)


def test_graph_reg_layer_selection_head_selection_and_row_sampling(
    cell_graph: HeteroData,
) -> None:
    """Direct calls on a [1, 4, 8, 8] attention that is 1/9 everywhere in head 1 and 1/8
    everywhere in the other heads. Scale 1, lambda 1, divisor 2 / 8:
    * layer 0 configured, layer_idx 1 asked: 0;
    * layer [0, 1] (a list) and layer_idx 1: head 1 is read, 4 L * 4 = 16 L;
    * row_sampling_rate 0.25: int(0.25 * 8) = 2 of the 4 positive rows, every one has the
      same KL, so 2 L * 4 = 8 L whatever rows are drawn;
    * a key that is not a relation of the graph contributes 0;
    * row_sampling_rate 0.75: int(6) exceeds the 4 positive rows, so all of them: 16 L;
    * a graph with no gene-gene relation: total_edges 0 skips the divisor, loss 0.
    """
    attention = torch.full((1, HEADS, N, N), 1.0 / 8.0)
    attention[:, 1] = 1.0 / 9.0

    def loss(
        heads: dict[str, dict[str, Any]], layer_idx: int, rate: float = 1.0
    ) -> float:
        model = _model(
            cell_graph,
            graph_reg_scale=1.0,
            graph_regularization_config=_reg_config(heads, rate),
        )
        return model.compute_graph_regularization_loss(attention, layer_idx).item()

    assert loss({"physical": {"layer": 0, "head": 1, "lambda": 1.0}}, 1) == 0.0
    listed = {"physical": {"layer": [0, 1], "head": 1, "lambda": 1.0}}
    assert loss(listed, 1) == pytest.approx(16 * LOG9, rel=1e-6)
    assert loss(listed, 1, rate=0.25) == pytest.approx(8 * LOG9, rel=1e-6)
    assert loss({"tflink": {"layer": 0, "head": 1, "lambda": 1.0}}, 0) == 0.0
    # rate 0.75: int(6) exceeds the 4 positive rows, so all 4 are used: 16 L.
    assert loss(listed, 0, rate=0.75) == pytest.approx(16 * LOG9, rel=1e-6)
    # no gene-gene relation at all: no adjacency, divisor skipped (0 edges), loss 0.
    del cell_graph["gene", "physical", "gene"]
    assert loss(listed, 0) == 0.0


def test_scale_zero_disables_regularization_and_stores_no_adjacency(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """``graph_reg_scale = 0`` builds no adjacency, a reg config is ignored, and the loss
    is exactly 0 on every forward.
    """
    heads = {"physical": {"layer": 0, "head": 0, "lambda": 1.0}}
    model = _model(cell_graph, graph_regularization_config=_reg_config(heads, 0.5))
    assert model.adjacency_matrices is None
    assert model.regularized_head_config is None
    assert model.row_sampling_rate == 1.0
    with torch.no_grad():
        _, out = model(cell_graph, batch)
    assert out["graph_reg_loss"].item() == 0.0


def test_a_bare_graph_name_against_suffixed_relations_is_silently_unregularized(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Finding: a ``regularized_heads`` key with no matching relation is skipped with a
    bare ``continue`` (cell_graph_transformer.py:563-564). ``to_cell_data`` names the
    physical relation ``GENE_GRAPH_EDGE_RELATION["physical"] = "physical_interaction"``
    while every shipped 006 CGT config keys it ``physical``, so on a suffixed graph the
    physical head contributes 0 and nothing warns. The successor raises ValueError here.
    Pinned until the 006 loss raises on an unknown graph name.
    """
    relation = GENE_GRAPH_EDGE_RELATION["physical"]
    assert relation == "physical_interaction"
    edge_index = cell_graph["gene", "physical", "gene"].edge_index
    del cell_graph["gene", "physical", "gene"]
    cell_graph["gene", relation, "gene"].edge_index = edge_index
    heads = {"physical": {"layer": 0, "head": 0, "lambda": 1.0}}
    model = _model(
        cell_graph, graph_reg_scale=1.0, graph_regularization_config=_reg_config(heads)
    )
    assert model.adjacency_matrices is not None
    assert list(model.adjacency_matrices) == [relation]
    with torch.no_grad():
        _, out = model(cell_graph, batch)
    assert out["graph_reg_loss"].item() == 0.0


# --- assembled model -------------------------------------------------------------- #
def test_parameter_counts_by_hand(cell_graph: HeteroData) -> None:
    """gene_embedding 8 * 16 = 128; cls 16; one encoder layer 3280; cross head
    1088 + 545 = 1633; total 5057. Two layers add 3280. The HyperSAGNN head is
    2450 + 545 = 2995.

    Finding: ``adaptive_loss_weighting`` adds ``log_mse_weight`` and ``log_reg_weight``
    (cell_graph_transformer.py:489-490), which ``num_parameters`` omits, so its total is
    2 short of the trainable count. Pinned until num_parameters lists them.
    """
    model = _model(cell_graph)
    assert model.num_parameters == {
        "gene_embedding": 128,
        "cls_token": 16,
        "transformer_layers": 3280,
        "perturbation_head": 1633,
        "total": 5057,
    }
    assert sum(p.numel() for p in model.parameters()) == 5057
    assert _model(cell_graph, num_transformer_layers=2).num_parameters["total"] == 8337
    hyper = _model(cell_graph, perturbation_head_config={"use_cross_attention": False})
    assert hyper.num_parameters["perturbation_head"] == 2995
    adaptive = _model(cell_graph, adaptive_loss_weighting=True)
    assert adaptive.num_parameters["total"] == 5057
    assert sum(p.numel() for p in adaptive.parameters()) == 5059
    assert adaptive.log_mse_weight.item() == 0.0
    assert adaptive.log_reg_weight.item() == -3.0


def test_perturbation_head_config_keys_reach_the_head(cell_graph: HeteroData) -> None:
    """``num_heads`` and ``dropout`` from perturbation_head_config build the head; the
    model dropout is the fallback for ``dropout``.
    """
    model = _model(
        cell_graph, perturbation_head_config={"num_heads": 2, "dropout": 0.3}
    )
    assert model.perturbation_head.cross_attn.num_heads == 2
    assert model.perturbation_head.cross_attn.dropout == 0.3
    dropout = model.perturbation_head.mlp[2]
    assert isinstance(dropout, nn.Dropout) and dropout.p == 0.3
    fallback = _model(cell_graph, dropout=0.2)
    assert fallback.perturbation_head.cross_attn.dropout == 0.2
    assert fallback.perturbation_head.cross_attn.num_heads == 4


def test_forward_outputs_and_the_perturbation_blind_encoder(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Predictions are [3, 1]; h_CLS is row 0 and H_genes rows 1..8 of the encoder
    output; one gene attention [1, 4, 8, 8] per layer.

    Pinned difference from the successor: the 006 encoder input is the embedding table
    plus CLS only, so h_CLS and H_genes are IDENTICAL for any two batches; the genotype
    enters only through the head. (The successor builds per-genotype perturbed gene
    states with EquivariantPerturbationTransform.)
    """
    model = _model(cell_graph)
    with torch.no_grad():
        pred, out = model(cell_graph, batch)
        _, other = model(cell_graph, _batch_of([[7], [6, 5]]))
    assert pred.shape == (3, 1)
    assert out["h_CLS"].shape == (D,)
    assert out["H_genes"].shape == (N, D)
    assert len(out["attention_weights"]) == 1
    assert out["attention_weights"][0].shape == (1, HEADS, N, N)
    assert torch.equal(out["h_CLS"], other["h_CLS"])
    assert torch.equal(out["H_genes"], other["H_genes"])
    with torch.no_grad():
        expected = model.perturbation_head(
            torch.cat([out["h_CLS"].unsqueeze(0), out["H_genes"]]),
            batch["gene"].perturbation_indices,
            batch["gene"].perturbation_indices_batch,
        )
    assert torch.equal(pred, expected)


@pytest.mark.parametrize("cross", [True, False])
def test_a_batch_equals_one_run_per_genotype(
    cell_graph: HeteroData, batch: HeteroData, cross: bool
) -> None:
    """Prediction b of the 3-genotype batch equals the single-genotype run of set b
    (batch vector all 0), for both heads: no state leaks across genotypes.
    """
    model = _model(cell_graph, perturbation_head_config={"use_cross_attention": cross})
    with torch.no_grad():
        pred, _ = model(cell_graph, batch)
        singles = torch.cat([model(cell_graph, _batch_of([s]))[0] for s in SETS])
    torch.testing.assert_close(pred, singles, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("cross", [True, False])
def test_genotype_and_gene_order_symmetries(
    cell_graph: HeteroData, cross: bool
) -> None:
    """Reversing the genotypes reverses the predictions; reordering the genes inside each
    set ({2, 1}, {3}, {5, 0, 4}) leaves the predictions (mean query / set attention).
    """
    model = _model(cell_graph, perturbation_head_config={"use_cross_attention": cross})
    with torch.no_grad():
        pred, _ = model(cell_graph, _batch_of(SETS))
        rev, _ = model(cell_graph, _batch_of(SETS[::-1]))
        shuffled, _ = model(cell_graph, _batch_of([[2, 1], [3], [5, 0, 4]]))
    torch.testing.assert_close(rev, pred.flip(0), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(shuffled, pred, atol=1e-6, rtol=1e-6)


def _relabeled(cell_graph: HeteroData) -> HeteroData:
    graph = cell_graph.clone()
    for edge_type in graph.edge_types:
        src, _, dst = edge_type
        edge_index = graph[edge_type].edge_index.clone()
        if src == "gene":
            edge_index[0] = INVERSE[edge_index[0]]
        if dst == "gene":
            edge_index[1] = INVERSE[edge_index[1]]
        graph[edge_type].edge_index = edge_index
    return graph


@pytest.mark.parametrize("cross", [True, False])
def test_relabeling_genes_permutes_gene_states_and_keeps_predictions(
    cell_graph: HeteroData, cross: bool
) -> None:
    """Model B is model A with embedding row j = A's row PERM[j], on the relabeled graph,
    asked about the relabeled sets. The encoder has no positional term, so H_genes(B) =
    H_genes(A)[PERM], h_CLS and the predictions are unchanged, and the graph reg loss
    (adjacency relabeled the same way) is unchanged.
    """
    heads = {"physical": {"layer": 0, "head": 1, "lambda": 1.0}}
    kwargs: dict[str, Any] = dict(
        graph_reg_scale=0.1,
        graph_regularization_config=_reg_config(heads),
        perturbation_head_config={"use_cross_attention": cross},
    )
    model_a = _model(cell_graph, **kwargs)
    relabeled = _relabeled(cell_graph)
    model_b = copy.deepcopy(model_a)
    model_b.adjacency_matrices = _model(relabeled, **kwargs).adjacency_matrices
    with torch.no_grad():
        model_b.gene_embedding.weight.copy_(model_a.gene_embedding.weight[PERM])
        pred_a, out_a = model_a(cell_graph, _batch_of(SETS))
        new_sets = [[int(INVERSE[g]) for g in s] for s in SETS]
        pred_b, out_b = model_b(relabeled, _batch_of(new_sets))
    torch.testing.assert_close(
        out_b["H_genes"], out_a["H_genes"][PERM], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(out_b["h_CLS"], out_a["h_CLS"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pred_b, pred_a, atol=1e-5, rtol=1e-5)
    assert out_a["graph_reg_loss"].item() > 0.0
    assert out_b["graph_reg_loss"].item() == pytest.approx(
        out_a["graph_reg_loss"].item(), rel=1e-5
    )


def test_seeded_construction_is_deterministic(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Same seed: identical state dicts and bit-identical predictions; seed 1 differs."""
    a, b, c = _model(cell_graph, 0), _model(cell_graph, 0), _model(cell_graph, 1)
    with torch.no_grad():
        pa, pb, pc = (m(cell_graph, batch)[0] for m in (a, b, c))
    assert all(
        torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()
    )
    assert torch.equal(pa, pb)
    assert not torch.equal(pa, pc)


def test_gradient_reaches_every_parameter_except_the_adaptive_weights(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """Cross head, two layers, graph reg on layer 1, loss pred.sum() + reg loss.

    ``log_mse_weight`` / ``log_reg_weight`` are never used in forward (only by the
    training script) and are the only parameters with no grad. Every key bias (each
    encoder ``k_proj.bias`` and the K slice [16:32] of the cross-attention in_proj_bias)
    has a gradient that is 0 up to float noise: it adds q . b_k to every score of a row,
    which the row softmax cancels (measured 2.6e-10, 1.9e-8 and 8.7e-10, never exactly 0,
    so an exact ``== 0`` is not available). Every other parameter's gradient exceeds 1e-7:
    the smallest, layer 0 q_proj.weight, is ~1.4e-6 because the embeddings start at std
    0.02. The 1e-7 cut therefore has a margin of ~5x over the largest noise value and
    ~14x under the smallest real gradient; it is stable for this seeded init but would
    need re-deriving if the init changed.
    """
    heads = {"physical": {"layer": 1, "head": 0, "lambda": 1.0}}
    model = _model(
        cell_graph,
        num_transformer_layers=2,
        graph_reg_scale=0.1,
        graph_regularization_config=_reg_config(heads),
        adaptive_loss_weighting=True,
    )
    pred, out = model(cell_graph, batch)
    (pred.sum() + out["graph_reg_loss"]).backward()
    assert _no_grad(model) == ["log_mse_weight", "log_reg_weight"]
    tiny = sorted(
        n
        for n, p in model.named_parameters()
        if p.grad is not None and p.grad.abs().max().item() < 1e-7
    )
    assert tiny == [
        "transformer_layers.0.k_proj.bias",
        "transformer_layers.1.k_proj.bias",
    ]
    in_proj_grad = model.perturbation_head.cross_attn.in_proj_bias.grad
    assert in_proj_grad is not None
    assert in_proj_grad[D : 2 * D].abs().max().item() < 1e-7
    assert in_proj_grad[2 * D :].abs().max().item() > 1e-7


def test_hypersagnn_head_attention_gets_exactly_zero_gradient_at_init(
    cell_graph: HeteroData, batch: HeteroData
) -> None:
    """ReZero: beta1 = beta2 = 0, so d(out)/d(Q, K, V, O) carries a factor beta = 0 and
    those 16 tensors get a gradient of exactly 0 (not None); the betas and the static
    embedding get nonzero gradients. With beta1 = beta2 = 0.5 every parameter does.
    """
    model = _model(cell_graph, perturbation_head_config={"use_cross_attention": False})
    pred, _ = model(cell_graph, batch)
    pred.sum().backward()
    zero = sorted(
        n
        for n, p in model.named_parameters()
        if p.grad is not None and p.grad.abs().max().item() == 0.0
    )
    assert zero == sorted(
        f"perturbation_head.hypersagnn.{proj}{layer}.{kind}"
        for proj in "QKVO"
        for layer in (1, 2)
        for kind in ("weight", "bias")
    )
    assert _no_grad(model) == []
    model.zero_grad()
    with torch.no_grad():
        model.perturbation_head.hypersagnn.beta1.fill_(0.5)
        model.perturbation_head.hypersagnn.beta2.fill_(0.5)
    pred, _ = model(cell_graph, batch)
    pred.sum().backward()
    assert all(
        p.grad is not None and p.grad.abs().max().item() > 0.0
        for p in model.parameters()
    )


def test_reg_loss_reads_pre_dropout_attention_while_the_output_uses_post_dropout() -> (
    None
):
    """Finding: in training mode the layer returns ``attention_weights[:, :, 1:, 1:]``
    taken BEFORE dropout (cell_graph_transformer.py:118) while the values are mixed with
    ``self.dropout(attention_weights)`` (line 108). With dropout 0.5 the returned
    gene attention equals the eval-mode softmax exactly, and the post-dropout tensor the
    output actually used (first call of ``self.dropout``, captured by a hook) differs from
    it: entries are 0 or 2x. So the KL regularizes an attention the forward did not use.
    Pinned until the KL and the output read the same tensor (or the difference is
    documented as intended).
    """
    torch.manual_seed(0)
    layer = GraphRegularizedTransformerLayer(D, HEADS, dropout=0.5)
    x = torch.randn(1, N + 1, D)
    used: list[torch.Tensor] = []

    def hook(module: nn.Module, args: Any, output: torch.Tensor) -> None:
        used.append(output.detach().clone())

    handle = layer.dropout.register_forward_hook(hook)
    layer.train()
    with torch.no_grad():
        _, returned = layer(x, return_attention=True)
    handle.remove()
    layer.eval()
    with torch.no_grad():
        _, eval_attention = layer(x, return_attention=True)
    assert returned is not None and eval_attention is not None
    assert torch.equal(returned, eval_attention)
    post = used[0][:, :, 1:, 1:]
    assert post.shape == returned.shape
    assert not torch.allclose(post, returned)
    kept = post != 0
    torch.testing.assert_close(post[kept], 2.0 * returned[kept])
    assert 0 < int(kept.sum()) < kept.numel()
