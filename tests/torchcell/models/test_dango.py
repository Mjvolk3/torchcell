# tests/torchcell/models/test_dango.py
# [[tests.torchcell.models.test_dango]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_dango.py
"""``Dango`` (Zhang et al. 2020 reimplementation) on a four-gene, two-network graph.

Fixture graph ``_cell_graph()``: four genes, the two edge types named as the 006
configs name them, edges written source -> target (SAGEConv aggregates at the target):

* ``string12_0_neighborhood``: 0->1, 1->0, 1->2, 2->1, 3->2 (gene 3 has no in-edge);
* ``string12_0_fusion``: 0->3, 3->0.

Batches are built the way ``experiments/006-kuzmin-tmi/scripts/dango.py`` builds them:
one ``HeteroData`` per genotype holding global ``perturbation_indices`` (the
``Perturbation`` graph processor) and ``phenotype_values``, collated with
``Batch.from_data_list(..., follow_batch=["perturbation_indices"])``.

Closed forms (hand-set weights; derivations in each docstring):

* ``DangoPreTrain`` with SAGE ``lin_l = I``, ``lin_l.bias = 0``, ``lin_r = 0``: each layer
  is ``relu(mean of in-neighbor rows)`` (zero with no in-neighbor).
* ``_global_attention_layer`` with ``Q = K = 0`` (uniform attention over allowed keys),
  ``V = O = I`` and ``beta = 1``: ``out_i = x_i + mean_{j != i, same set} x_j``.
* ``HyperSAGNN`` with those weights in both layers, ``static = relu(I x)`` and a
  prediction layer of ones: a triple gives ``d2_i = x_i / 4 + 5 S / 4`` (``S`` the sum
  of the triple), a pair gives ``d2_i = 2 (x_i + x_k)``.
"""

import re
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import SAGEConv

from torchcell.models.dango import Dango, DangoPreTrain, HyperSAGNN, MetaEmbedding

NEIGH = "string12_0_neighborhood"
FUSION = "string12_0_fusion"
EDGE_TYPES = [NEIGH, FUSION]
X0 = [[1.0, 2.0], [3.0, -1.0], [-2.0, 1.0], [0.5, 0.5]]


@pytest.fixture(autouse=True)
def _isolated_rng() -> Iterator[None]:
    """Every ``torch.manual_seed`` in this file runs inside a forked RNG state."""
    with torch.random.fork_rng():
        yield


def _cell_graph() -> HeteroData:
    graph = HeteroData()
    graph["gene"].num_nodes = 4
    graph["gene", NEIGH, "gene"].edge_index = torch.tensor(
        [[0, 1, 1, 2, 3], [1, 0, 2, 1, 2]]
    )
    graph["gene", FUSION, "gene"].edge_index = torch.tensor([[0, 3], [3, 0]])
    return graph


def _genotype(indices: list[int], value: float) -> HeteroData:
    data = HeteroData()
    data["gene"].num_nodes = 4
    data["gene"].perturbation_indices = torch.tensor(indices, dtype=torch.long)
    data["gene"].phenotype_values = torch.tensor([value])
    return data


def _batch(*genotypes: list[int]) -> Batch:
    data = [_genotype(g, 0.1 * (k + 1)) for k, g in enumerate(genotypes)]
    return Batch.from_data_list(data, follow_batch=["perturbation_indices"])


def _dango(seed: int = 0) -> Dango:
    torch.manual_seed(seed)
    return Dango(gene_num=4, edge_types=EDGE_TYPES, hidden_channels=8, num_heads=2)


def _identity_sage(conv: nn.Module, bias: tuple[float, float] = (0.0, 0.0)) -> None:
    assert isinstance(conv, SAGEConv)
    with torch.no_grad():
        conv.lin_l.weight.copy_(torch.eye(2))
        assert conv.lin_l.bias is not None
        conv.lin_l.bias.copy_(torch.tensor(bias))
        conv.lin_r.weight.zero_()


def _closed_form_hyper(hidden: int = 2, heads: int = 2) -> HyperSAGNN:
    model = HyperSAGNN(hidden_channels=hidden, num_heads=heads)
    eye = torch.eye(hidden)
    with torch.no_grad():
        for q, k, v, o, beta in [
            (model.Q1, model.K1, model.V1, model.O1, model.beta1),
            (model.Q2, model.K2, model.V2, model.O2, model.beta2),
        ]:
            for lin in (q, k):
                lin.weight.zero_()
                lin.bias.zero_()
            for lin in (v, o):
                lin.weight.copy_(eye)
                lin.bias.zero_()
            beta.fill_(1.0)
        static = model.static_embedding[0]
        assert isinstance(static, nn.Linear)
        static.weight.copy_(eye)
        static.bias.zero_()
        model.prediction_layer.weight.fill_(1.0)
        model.prediction_layer.bias.zero_()
    return model


def _np_masked_mha(
    x: npt.NDArray[np.float64],
    batch: npt.NDArray[np.int64],
    layer: tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear, nn.Parameter],
    heads: int,
) -> npt.NDArray[np.float64]:
    """Independent masked multi-head attention: keys are the OTHER members of the set."""
    q_l, k_l, v_l, o_l, beta = layer

    def lin(m: nn.Linear, a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        w: npt.NDArray[np.float64] = m.weight.detach().double().numpy()
        b: npt.NDArray[np.float64] = m.bias.detach().double().numpy()
        out: npt.NDArray[np.float64] = a @ w.T + b
        return out

    n, h = x.shape
    d = h // heads
    q, k, v = lin(q_l, x), lin(k_l, x), lin(v_l, x)
    out = np.zeros((n, h))
    for head in range(heads):
        sl = slice(head * d, (head + 1) * d)
        for i in range(n):
            keys = [j for j in range(n) if j != i and batch[j] == batch[i]]
            if not keys:
                continue
            s = np.array([q[i, sl] @ k[j, sl] / np.sqrt(d) for j in keys])
            w = np.exp(s - s.max())
            w = w / w.sum()
            out[i, sl] = sum(wj * v[j, sl] for wj, j in zip(w, keys, strict=True))
    return float(beta.item()) * lin(o_l, out) + x


# --- DangoPreTrain ------------------------------------------------------------------ #
def test_pretrain_identity_sage_closed_form_embeddings_and_raw_reconstruction() -> None:
    """Each SAGE layer is relu(mean of in-neighbor rows): messages flow source->target.

    Embedding X = [[1, 2], [3, -1], [-2, 1], [.5, .5]].
    Neighborhood in-neighbors: 0 <- {1}, 1 <- {0, 2}, 2 <- {1, 3}, 3 <- {}.
      h1 = relu([X1, (X0 + X2) / 2, (X1 + X3) / 2, 0]) = [[3, 0], [0, 1.5], [1.75, 0], [0, 0]]
      layer 2 carries bias (-1, 0) for this network:
      h2 = relu([h1_1, (h1_0 + h1_2) / 2, (h1_1 + h1_3) / 2, 0] + (-1, 0))
         = relu([[-1, 1.5], [1.375, 0], [-1, 0.75], [-1, 0]])
         = [[0, 1.5], [1.375, 0], [0, 0.75], [0, 0]]
    Fusion in-neighbors: 0 <- {3}, 3 <- {0}:
      h1 = [[.5, .5], 0, 0, [1, 2]],  h2 = [[1, 2], 0, 0, [.5, .5]].
    Reconstruction = h2 W^T + b with W = [[1, 0], [0, 1], [1, 1], [1, -1]],
    b = [.1, .2, .3, .4]; gene 0 under neighborhood: [0, 1.5, 1.5, -1.5] + b =
    [.1, 1.7, 1.8, -1.1]. The row is a raw linear output (no sigmoid): -1.1 survives, and
    it has gene_num = 4 columns, the shape of the dense adjacency the loss builds.
    """
    model = DangoPreTrain(gene_num=4, edge_types=EDGE_TYPES, hidden_channels=2)
    w = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    b = torch.tensor([0.1, 0.2, 0.3, 0.4])
    with torch.no_grad():
        model.gene_embedding.weight.copy_(torch.tensor(X0))
        for et in EDGE_TYPES:
            _identity_sage(model.layer1_convs[et])
            _identity_sage(
                model.layer2_convs[et], (-1.0, 0.0) if et == NEIGH else (0.0, 0.0)
            )
            recon = model.recon_layers[et]
            assert isinstance(recon, nn.Linear)
            recon.weight.copy_(w)
            recon.bias.copy_(b)

    out = model(_cell_graph())

    assert sorted(out) == ["embeddings", "initial_embeddings", "reconstructions"]
    emb = out["embeddings"]
    recons = out["reconstructions"]
    assert isinstance(emb, dict) and isinstance(recons, dict)
    assert list(emb) == EDGE_TYPES and list(recons) == EDGE_TYPES
    torch.testing.assert_close(out["initial_embeddings"], torch.tensor(X0))
    h2_neigh = torch.tensor([[0.0, 1.5], [1.375, 0.0], [0.0, 0.75], [0.0, 0.0]])
    h2_fusion = torch.tensor([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.5, 0.5]])
    torch.testing.assert_close(emb[NEIGH], h2_neigh)
    torch.testing.assert_close(emb[FUSION], h2_fusion)
    torch.testing.assert_close(recons[NEIGH], h2_neigh @ w.T + b)
    torch.testing.assert_close(recons[FUSION], h2_fusion @ w.T + b)
    assert recons[NEIGH].shape == (4, 4)
    torch.testing.assert_close(recons[NEIGH][0], torch.tensor([0.1, 1.7, 1.8, -1.1]))


def test_pretrain_missing_edge_type_yields_zero_embeddings_and_reconstructions() -> (
    None
):
    """An edge type absent from the graph gives zeros [N, H] and zeros [N, gene_num];
    the present type is still computed (finite, not all zero for this seed).
    """
    torch.manual_seed(0)
    model = DangoPreTrain(gene_num=4, edge_types=[NEIGH, "string12_0_database"])
    out = model(_cell_graph())
    emb = out["embeddings"]
    recons = out["reconstructions"]
    assert isinstance(emb, dict) and isinstance(recons, dict)
    assert torch.equal(emb["string12_0_database"], torch.zeros(4, 64))
    assert torch.equal(recons["string12_0_database"], torch.zeros(4, 4))
    assert emb[NEIGH].shape == (4, 64) and bool(emb[NEIGH].abs().sum() > 0)


def test_pretrain_lambda_table_only_knows_string9_and_string11_names() -> None:
    """Finding: ``DangoPreTrain.lambda_values`` (dango.py:80-96) gives 0.1 only to the
    string9_1 / string11_0 neighborhood, coexpression and experimental names; every
    string12_0 network (the 006 production configs) falls to the else branch, 1.0.
    No training script reads this table (005/006 pass ``determine_lambda_values()`` to
    ``DangoLoss``); only the module's own ``main`` demo copies it.
    Pinned until the table is removed or extended to string12_0.
    """
    names = [
        "string9_1_neighborhood",
        "string11_0_coexpression",
        "string9_1_experimental",
        "string9_1_fusion",
        "string12_0_neighborhood",
        "string12_0_coexpression",
        "string12_0_experimental",
    ]
    model = DangoPreTrain(gene_num=3, edge_types=names, hidden_channels=4)
    assert model.lambda_values == {
        "string9_1_neighborhood": 0.1,
        "string11_0_coexpression": 0.1,
        "string9_1_experimental": 0.1,
        "string9_1_fusion": 1.0,
        "string12_0_neighborhood": 1.0,
        "string12_0_coexpression": 1.0,
        "string12_0_experimental": 1.0,
    }


def test_pretrain_reset_parameters_zeroes_reconstruction_biases() -> None:
    """``reset_parameters`` zeroes every reconstruction bias and draws the embedding and
    reconstruction weights from N(0, 0.1): with gene_num 200 and H 64 the embedding has
    12800 draws and each reconstruction weight 12800, so each sample std lies within
    0.1 * (1 +- 0.25) (the default nn.Linear init would give std 1 / sqrt(3 * 64) =
    0.072, outside the band).
    """
    torch.manual_seed(0)
    model = DangoPreTrain(gene_num=200, edge_types=EDGE_TYPES)
    for et in EDGE_TYPES:
        recon = model.recon_layers[et]
        assert isinstance(recon, nn.Linear)
        assert torch.equal(recon.bias, torch.zeros(200))
        assert 0.075 < recon.weight.std().item() < 0.125
    std = model.gene_embedding.weight.std().item()
    assert 0.075 < std < 0.125


# --- MetaEmbedding ------------------------------------------------------------------ #
def test_meta_embedding_softmax_over_networks_per_gene_closed_form() -> None:
    """Score = relu(e[0]) (first layer [[1, 0]], second [[1]], zero biases); the weights
    are a softmax across the TWO networks for each gene, and the output is the weighted
    sum of that gene's two embeddings.

    A = [[1, 0], [0, 5], [2, 2]], B = [[3, 1], [1, 1], [2, -2]]:
    gene 0 scores (1, 3): w = (1, e^2) / (1 + e^2); gene 1 scores (0, 1); gene 2 (2, 2):
    w = (.5, .5), output (2, 0).
    """
    meta = MetaEmbedding(hidden_channels=2)
    first, second = meta.attention_mlp[0], meta.attention_mlp[2]
    assert isinstance(first, nn.Linear) and isinstance(second, nn.Linear)
    with torch.no_grad():
        first.weight.copy_(torch.tensor([[1.0, 0.0]]))
        first.bias.zero_()
        second.weight.copy_(torch.tensor([[1.0]]))
        second.bias.zero_()
    a = torch.tensor([[1.0, 0.0], [0.0, 5.0], [2.0, 2.0]])
    b = torch.tensor([[3.0, 1.0], [1.0, 1.0], [2.0, -2.0]])

    out = meta({NEIGH: a, FUSION: b})

    scores = np.maximum(np.stack([a[:, 0].numpy(), b[:, 0].numpy()], axis=1), 0.0)
    w = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    expected = w[:, :1] * a.numpy() + w[:, 1:] * b.numpy()
    torch.testing.assert_close(out, torch.tensor(expected, dtype=torch.float32))
    torch.testing.assert_close(out[2], torch.tensor([2.0, 0.0]))


def test_meta_embedding_source_weights_sum_to_one_per_gene_and_weight_the_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture the weights the SOURCE computes (a recorder around ``F.softmax``, which
    ``MetaEmbedding.forward`` calls once): they are [5 genes, 3 networks], each gene's
    row sums to 1, they equal softmax over networks of the hooked MLP scores, and the
    output is the weighted sum of that gene's three embeddings.
    """
    torch.manual_seed(1)
    meta = MetaEmbedding(hidden_channels=8)
    embs = {f"net{k}": torch.randn(5, 8) for k in range(3)}
    scores: list[torch.Tensor] = []
    meta.attention_mlp.register_forward_hook(
        lambda module, inp, out: scores.append(out.detach().clone())
    )
    weights: list[torch.Tensor] = []
    real_softmax = torch.nn.functional.softmax

    def recording_softmax(*args: Any, **kwargs: Any) -> torch.Tensor:
        result = real_softmax(*args, **kwargs)
        weights.append(result.detach().clone())
        return result

    monkeypatch.setattr(torch.nn.functional, "softmax", recording_softmax)
    out = meta(embs)
    assert len(weights) == 1 and weights[0].shape == (5, 3)
    torch.testing.assert_close(weights[0].sum(dim=1), torch.ones(5))
    s_np = scores[0].view(5, 3).double().numpy()
    oracle = np.exp(s_np) / np.exp(s_np).sum(axis=1, keepdims=True)
    torch.testing.assert_close(
        weights[0].double(), torch.tensor(oracle), atol=1e-6, rtol=1e-6
    )
    w = weights[0].double().numpy()
    stacked = np.stack([e.double().numpy() for e in embs.values()], axis=1)
    expected = (stacked * w[:, :, None]).sum(axis=1)
    torch.testing.assert_close(
        out.double(), torch.tensor(expected), atol=1e-6, rtol=1e-6
    )


def test_meta_embedding_init_zero_biases() -> None:
    """Both MLP biases are zeroed by ``_initialize_weights``."""
    torch.manual_seed(0)
    meta = MetaEmbedding(hidden_channels=8)
    first, second = meta.attention_mlp[0], meta.attention_mlp[2]
    assert isinstance(first, nn.Linear) and isinstance(second, nn.Linear)
    assert torch.equal(first.bias, torch.zeros(4))
    assert torch.equal(second.bias, torch.zeros(1))


# --- HyperSAGNN --------------------------------------------------------------------- #
def test_attention_layer_excludes_self_closed_form() -> None:
    """Q = K = 0 makes attention uniform over the allowed keys; V = O = I, beta = 1:
    out_i = x_i + mean_{j != i, same set} x_j. Triple x = [(1, 0), (0, 2), (-1, 1)]:
    out_0 = (1, 0) + ((0, 2) + (-1, 1)) / 2 = (0.5, 1.5), out_1 = (0, 2) + (0, 0.5) =
    (0, 2.5), out_2 = (-1, 1) + (0.5, 1) = (-0.5, 2). With self-attention allowed the
    mean would include x_i and out_0 would be (1, 0) + (0, 1) = (1, 1).
    """
    model = _closed_form_hyper()
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
    batch = torch.tensor([0, 0, 0])
    mask = (batch.unsqueeze(-1) == batch.unsqueeze(0)) & ~torch.eye(3, dtype=torch.bool)
    out = model._global_attention_layer(
        x, mask, model.Q1, model.K1, model.V1, model.O1, model.beta1
    )
    torch.testing.assert_close(out, torch.tensor([[0.5, 1.5], [0.0, 2.5], [-0.5, 2.0]]))


def test_attention_layer_matches_independent_masked_mha_on_random_weights() -> None:
    """Seeded random projections, 2 heads of width 2, two sets {0, 1, 2} and {3, 4} and
    a singleton {5}; the numpy oracle attends each query to the OTHER members of its
    set with softmax(q k / sqrt(2)) per head. The singleton's attention row is empty, so
    the layer returns beta * O.bias + x there (the NaN-to-zero path).
    """
    torch.manual_seed(3)
    model = HyperSAGNN(hidden_channels=4, num_heads=2)
    with torch.no_grad():
        model.beta1.fill_(0.7)
        model.O1.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.4]))
    x = torch.randn(6, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 2])
    mask = (batch.unsqueeze(-1) == batch.unsqueeze(0)) & ~torch.eye(6, dtype=torch.bool)
    out = model._global_attention_layer(
        x, mask, model.Q1, model.K1, model.V1, model.O1, model.beta1
    )
    layer = (model.Q1, model.K1, model.V1, model.O1, model.beta1)
    expected = _np_masked_mha(x.double().numpy(), batch.numpy(), layer, heads=2)
    torch.testing.assert_close(
        out.double(), torch.tensor(expected), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(out[5], 0.7 * torch.tensor([0.1, -0.2, 0.3, 0.4]) + x[5])


def test_hyper_sagnn_closed_form_triple_and_pair_in_one_batch() -> None:
    """Triple x = [(1, 0), (0, 2), (-1, 1)], S = (0, 3): d2_i = x_i / 4 + 5 S / 4 =
    (.25, 3.75), (0, 4.25), (-.25, 4); static relu(x) = (1, 0), (0, 2), (0, 1); squared
    differences summed: 14.625, 5.0625, 9.0625; mean 28.75 / 3.
    Pair y = [(2, 0), (0, 0)]: d1 = (2, 0) for both, d2 = (4, 0); static (2, 0), (0, 0);
    scores 4 and 16, mean 10.
    """
    model = _closed_form_hyper()
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0], [2.0, 0.0], [0.0, 0.0]])
    out = model(x, torch.tensor([0, 0, 0, 1, 1]))
    torch.testing.assert_close(out, torch.tensor([28.75 / 3, 10.0]))


def test_hyper_sagnn_matches_two_layer_numpy_oracle_with_static_branch() -> None:
    """Random seeded weights, beta = 0.5 / 0.25, two sets of 3 and 2 nodes: the full
    forward equals relu-static, two oracle attention layers, squared difference,
    prediction layer and a per-set mean, computed in numpy.
    """
    torch.manual_seed(5)
    model = HyperSAGNN(hidden_channels=4, num_heads=2)
    with torch.no_grad():
        model.beta1.fill_(0.5)
        model.beta2.fill_(0.25)
    x = torch.randn(5, 4)
    batch = np.array([0, 0, 0, 1, 1])
    out = model(x, torch.tensor(batch))

    xd = x.double().numpy()
    d1 = _np_masked_mha(
        xd, batch, (model.Q1, model.K1, model.V1, model.O1, model.beta1), heads=2
    )
    d2 = _np_masked_mha(
        d1, batch, (model.Q2, model.K2, model.V2, model.O2, model.beta2), heads=2
    )
    s_lin = model.static_embedding[0]
    assert isinstance(s_lin, nn.Linear)
    static = np.maximum(
        xd @ s_lin.weight.detach().double().numpy().T
        + s_lin.bias.detach().double().numpy(),
        0.0,
    )
    node = (
        (d2 - static) ** 2
    ) @ model.prediction_layer.weight.detach().double().numpy().T
    node = node[:, 0] + model.prediction_layer.bias.item()
    expected = np.array([node[:3].mean(), node[3:].mean()])
    torch.testing.assert_close(
        out.double(), torch.tensor(expected), atol=1e-5, rtol=1e-5
    )


def test_hyper_sagnn_mask_blocks_cross_set_and_self_messages() -> None:
    """Moving gene 0's input moves set 0's score only; and with the closed-form weights
    the attention output for gene 0 (layer 1, before the residual) is independent of
    x_0: out_0 - x_0 = mean(x_1, x_2) for any x_0.
    """
    torch.manual_seed(2)
    model = HyperSAGNN(hidden_channels=4, num_heads=2)
    with torch.no_grad():
        model.beta1.fill_(0.5)
        model.beta2.fill_(0.5)
    x = torch.randn(5, 4)
    batch = torch.tensor([0, 0, 0, 1, 1])
    base = model(x, batch)
    moved = x.clone()
    moved[0] += 3.0
    after = model(moved, batch)
    assert after[1].item() == base[1].item()
    assert abs(after[0].item() - base[0].item()) > 1e-4

    cf = _closed_form_hyper()
    mask = ~torch.eye(3, dtype=torch.bool)
    for x0 in ([1.0, 0.0], [-7.0, 4.0]):
        xs = torch.tensor([x0, [0.0, 2.0], [-1.0, 1.0]])
        out = cf._global_attention_layer(xs, mask, cf.Q1, cf.K1, cf.V1, cf.O1, cf.beta1)
        torch.testing.assert_close(out[0] - xs[0], torch.tensor([-0.5, 1.5]))


def test_hyper_sagnn_noncontiguous_set_ids_raise_or_drop_sets() -> None:
    """Finding: ``HyperSAGNN.forward`` sizes the output as the NUMBER of distinct set ids
    (dango.py:307-308, 350-352), not max id + 1. Set ids [0, 0, 0, 2, 2, 2] (a genotype
    with no perturbation_indices between two triples) make ``scatter_mean`` index out of
    range. Not shown reachable in the 006 data (every Kuzmin genotype carries genes).
    Pinned until the output is sized by the batch's graph count.
    """
    model = _closed_form_hyper()
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]] * 2)
    with pytest.raises(
        RuntimeError,
        match=re.escape("index 2 is out of bounds for dimension 0 with size 2"),
    ):
        model(x, torch.tensor([0, 0, 0, 2, 2, 2]))


def test_hyper_sagnn_init_shapes_and_zero_betas() -> None:
    """head_dim = 8 // 2 = 4; both ReZero betas start at 0 before ``Dango`` re-inits."""
    model = HyperSAGNN(hidden_channels=8, num_heads=2)
    assert (model.hidden_channels, model.num_heads, model.head_dim) == (8, 2, 4)
    assert torch.equal(model.beta1, torch.zeros(1))
    assert torch.equal(model.beta2, torch.zeros(1))
    assert model.prediction_layer.weight.shape == (1, 8)


# --- Dango -------------------------------------------------------------------------- #
def test_dango_init_betas_and_zero_linear_biases_and_parameter_counts() -> None:
    """``_initialize_weights`` sets beta1 = beta2 = 0.01 and zeroes every HyperSAGNN
    Linear bias. Parameter counts with G = 4 genes, H = 8, E = 2 networks:
    pretrain G H + E (2 (2 H^2 + H) + H G + G) = 32 + 2 (272 + 36) = 648;
    meta H (H / 2) + H / 2 + H / 2 + 1 = 41;
    hyper static H^2 + H = 72, eight projections 8 * 72 = 576, two betas, prediction
    H + 1 = 9: 659; total 1348.
    """
    model = _dango()
    assert model.hyper_sagnn.beta1.item() == pytest.approx(0.01)
    assert model.hyper_sagnn.beta2.item() == pytest.approx(0.01)
    for name, module in model.hyper_sagnn.named_modules():
        if isinstance(module, nn.Linear):
            assert torch.equal(module.bias, torch.zeros_like(module.bias)), name
    g, h, e = 4, 8, 2
    pretrain = g * h + e * (2 * (2 * h * h + h) + h * g + g)
    meta = h * (h // 2) + h // 2 + h // 2 + 1
    hyper = (h * h + h) + 8 * (h * h + h) + 2 + (h + 1)
    assert model.num_parameters == {
        "pretrain_model": pretrain,
        "meta_embedding": meta,
        "hyper_sagnn": hyper,
        "total": pretrain + meta + hyper,
    }
    assert (pretrain, meta, hyper) == (648, 41, 659)


def test_dango_forward_stage_shapes_and_output_keys() -> None:
    """Two triples: per-network embeddings [4, 8], reconstructions [4, 4], integrated
    embeddings [4, 8], and one score per genotype [2]. Stage wiring: the integrated
    embeddings are ``meta_embedding`` of the per-network embeddings, and the scores are
    ``hyper_sagnn`` on the integrated rows at ``perturbation_indices`` grouped by
    ``perturbation_indices_batch``.
    """
    model = _dango()
    batch = _batch([0, 1, 2], [1, 2, 3])
    scores, outputs = model(_cell_graph(), batch)
    torch.testing.assert_close(
        outputs["integrated_embeddings"],
        model.meta_embedding(outputs["network_embeddings"]),
    )
    torch.testing.assert_close(
        scores,
        model.hyper_sagnn(
            outputs["integrated_embeddings"][batch["gene"].perturbation_indices],
            batch["gene"].perturbation_indices_batch,
        ),
    )
    assert scores.shape == (2,)
    assert sorted(outputs) == [
        "initial_embeddings",
        "integrated_embeddings",
        "interaction_scores",
        "network_embeddings",
        "reconstructions",
    ]
    assert outputs["interaction_scores"] is scores
    assert {k: tuple(v.shape) for k, v in outputs["network_embeddings"].items()} == {
        NEIGH: (4, 8),
        FUSION: (4, 8),
    }
    assert {k: tuple(v.shape) for k, v in outputs["reconstructions"].items()} == {
        NEIGH: (4, 4),
        FUSION: (4, 4),
    }
    assert outputs["integrated_embeddings"].shape == (4, 8)
    assert outputs["initial_embeddings"].shape == (4, 8)
    assert bool(torch.isfinite(scores).all())


def test_dango_forward_is_seed_deterministic() -> None:
    """Two models built under seed 0 have equal weights and equal scores; seed 1 differs."""
    batch = _batch([0, 1, 2], [1, 2, 3])
    first, _ = _dango(0)(_cell_graph(), batch)
    second, _ = _dango(0)(_cell_graph(), batch)
    other, _ = _dango(1)(_cell_graph(), batch)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_dango_batched_genotypes_equal_each_genotype_run_alone() -> None:
    """Issue #596's question for this model: with ``follow_batch=["perturbation_indices"]``
    (as the 005/006 dango scripts set it), a batch of two triples and a singleton gives
    the same three scores as each genotype collated alone; the batch>1 prediction is
    not zeroed or mixed. (#596's silent-zero mechanism lives in
    ``hetero_cell_bipartite_dango_gi``, not here.)
    """
    model = _dango()
    graph = _cell_graph()
    together, _ = model(graph, _batch([0, 1, 2], [1, 2, 3], [3]))
    alone = torch.cat([model(graph, _batch(g))[0] for g in ([0, 1, 2], [1, 2, 3], [3])])
    assert together.shape == (3,)
    torch.testing.assert_close(together, alone, atol=1e-6, rtol=0.0)
    assert bool((together != 0).all())


def test_dango_without_follow_batch_raises_attribute_error() -> None:
    """Collated without ``follow_batch`` the batch has no ``perturbation_indices_batch``
    and ``Dango.forward`` raises instead of predicting a silent zero (contrast #596).
    """
    model = _dango()
    batch = Batch.from_data_list([_genotype([0, 1, 2], 0.1), _genotype([1, 2, 3], 0.2)])
    with pytest.raises(
        AttributeError,
        match=re.escape(
            "'NodeStorage' object has no attribute 'perturbation_indices_batch'"
        ),
    ):
        model(_cell_graph(), batch)


def test_dango_trailing_empty_genotype_drops_a_prediction() -> None:
    """Finding: a genotype with no ``perturbation_indices`` at the END of a batch gives
    two predictions for three targets (dango.py:307-308 sizes the output by distinct
    set ids): the loss and metrics then receive [2] predictions against [3] targets
    (in the middle of a batch the same genotype raises, see the HyperSAGNN test). Not
    shown reachable in the 006 data.
    Pinned until HyperSAGNN sizes its output by the number of graphs in the batch.
    """
    model = _dango()
    batch = _batch([0, 1, 2], [1, 2, 3], [])
    scores, _ = model(_cell_graph(), batch)
    assert batch["gene"].phenotype_values.shape == (3,)
    assert scores.shape == (2,)
    alone = torch.cat(
        [model(_cell_graph(), _batch(g))[0] for g in ([0, 1, 2], [1, 2, 3])]
    )
    torch.testing.assert_close(scores, alone, atol=1e-6, rtol=0.0)


def test_dango_score_is_invariant_to_gene_order_within_a_triple() -> None:
    """Attention is permutation equivariant and the set score is a mean, so the order of
    the three genes does not change a genotype's score (all six orders).
    """
    model = _dango()
    graph = _cell_graph()
    orders = ([0, 1, 3], [0, 3, 1], [1, 0, 3], [1, 3, 0], [3, 0, 1], [3, 1, 0])
    scores = torch.cat([model(graph, _batch(list(o)))[0] for o in orders])
    torch.testing.assert_close(scores, scores[:1].expand(6), atol=1e-6, rtol=0.0)


def test_dango_gradient_reaches_every_parameter_except_reconstruction_heads() -> None:
    """Backpropagating the interaction scores alone reaches every parameter except the
    four reconstruction-head tensors (they feed only the reconstruction loss); adding
    the reconstruction sums reaches all of them.
    """
    model = _dango()
    scores, outputs = model(_cell_graph(), _batch([0, 1, 2], [1, 2, 3]))
    scores.sum().backward()
    no_grad = sorted(n for n, p in model.named_parameters() if p.grad is None)
    assert no_grad == [
        f"pretrain_model.recon_layers.{FUSION}.bias",
        f"pretrain_model.recon_layers.{FUSION}.weight",
        f"pretrain_model.recon_layers.{NEIGH}.bias",
        f"pretrain_model.recon_layers.{NEIGH}.weight",
    ]

    model.zero_grad(set_to_none=True)
    scores, outputs = model(_cell_graph(), _batch([0, 1, 2], [1, 2, 3]))
    total = scores.sum() + sum(r.sum() for r in outputs["reconstructions"].values())
    total.backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_hyper_sagnn_output_is_ordered_by_set_id_not_first_appearance() -> None:
    """Set ids [1, 1, 0, 0]: output[0] is the score of the set with id 0 (rows 2, 3) and
    output[1] that of id 1 (rows 0, 1), each equal to that set run alone. PyG batches
    number graphs in order, so the ids are ascending in practice.
    """
    torch.manual_seed(4)
    model = HyperSAGNN(hidden_channels=4, num_heads=2)
    with torch.no_grad():
        model.beta1.fill_(0.5)
        model.beta2.fill_(0.5)
    x = torch.randn(4, 4)
    out = model(x, torch.tensor([1, 1, 0, 0]))
    first = model(x[:2], torch.tensor([0, 0]))
    second = model(x[2:], torch.tensor([0, 0]))
    torch.testing.assert_close(out, torch.cat([second, first]), atol=1e-6, rtol=0.0)
    assert not torch.allclose(first, second)


def test_hyper_sagnn_duplicate_gene_attends_to_its_own_copy() -> None:
    """The self mask is positional, so a gene listed twice attends to its copy. With the
    closed-form weights, x = [a, a, b], a = (1, 0), b = (0, 2): layer 1 gives
    a + (a + b) / 2 = (1.5, 1) for each copy, whereas the deduplicated pair [a, b] gives
    a + b = (1, 2). Not reachable through the ``Perturbation`` processor, which builds
    ``perturbation_indices`` from a set of names (one index per gene).
    """
    model = _closed_form_hyper()
    a, b = [1.0, 0.0], [0.0, 2.0]
    dup = torch.tensor([a, a, b])
    mask = ~torch.eye(3, dtype=torch.bool)
    out = model._global_attention_layer(
        dup, mask, model.Q1, model.K1, model.V1, model.O1, model.beta1
    )
    torch.testing.assert_close(out[0], torch.tensor([1.5, 1.0]))
    torch.testing.assert_close(out[1], torch.tensor([1.5, 1.0]))
    pair = model._global_attention_layer(
        torch.tensor([a, b]),
        ~torch.eye(2, dtype=torch.bool),
        model.Q1,
        model.K1,
        model.V1,
        model.O1,
        model.beta1,
    )
    torch.testing.assert_close(pair[0], torch.tensor([1.0, 2.0]))


def test_dango_num_parameters_counts_trainable_parameters_only() -> None:
    """Freezing every HyperSAGNN parameter drops its count from 659 to 0 and the total
    from 1348 to 648 + 41 = 689.
    """
    model = _dango()
    for p in model.hyper_sagnn.parameters():
        p.requires_grad_(False)
    assert model.num_parameters == {
        "pretrain_model": 648,
        "meta_embedding": 41,
        "hyper_sagnn": 0,
        "total": 689,
    }
